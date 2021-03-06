
# Sample UDP Server - Multi threaded

# Import the necessary python modules
import socketserver
import socket
import threading
import pathlib
from sys import path, argv
path.insert(1, './tools')

from multiprocessing import Queue, Process
from queue import Empty
import atexit
from math import ceil

import tensorflow as tf
import keras
import train
 
# Create a tuple with IP Address and Port Number
ServerAddress = ("127.0.0.1", 9999)
receiveAddressPort = ("127.0.0.1", 9998)

# current working directory
CURRENT_WORKING_DIRECTORY = pathlib.Path().absolute()
DEFAULT_MODEL = 'holds_model2'

f_q = Queue()
p_q = Queue()

LABELS = [None, 'A', 'B', 'C', 'Z']

PRINT_FREQ = 30
PRED_FREQ = 5
assert PRINT_FREQ % PRED_FREQ == 0


class Message():
    def __init__(self, data, address):
        self.data = data
        self.address = address

class Error(Exception):
    pass

class MyUDPRequestHandler(socketserver.DatagramRequestHandler):
    # Override the handle() method

    def handle(self):
        # Receive and print the datagram received from client
        # print("Recieved one request from {}".format(self.client_address[0]))

        datagram = self.rfile.readline().strip()

        # print("Datagram Received from client is:".format(datagram))
        # print(datagram)

        f_q.put(datagram)
        # print(f"f_q len: {f_q.qsize()}")

        # Print the name of the thread
        # print("Thread Name:{}".format(threading.current_thread().name))


        # # Send a message to the client
        # self.wfile.write("Message from Server! Hello Client".encode())


def predict_loop(model_path):
    print("Starting prediction init")
    train.init_gpu()
    model = keras.models.load_model(model_path)

    delay = 0
    window = None
    results = None
    results_len = ceil(PRINT_FREQ / PRED_FREQ)
    print()
    print("====================Starting prediction===============")
    print()
    while True:
        row = f_q.get()
        print("got smt")
        if window is None:
            window = np.zeros((train.TIMESTEPS, len(row)))

        # Discard oldest frame and append new frame to data window
        window[:-1] = window[1:]
        window[-1] = row

        if delay >= PRED_FREQ:
            out = model(np.array([window]))
            if results is None:
                results = np.zeros((results_len, len(LABELS)))
            results[:-1] = results[1:]
            results[-1] = out

            p_q.put(np.mean(results, axis=0))
            print("predicted smt")
            delay = 0
        delay += 1


def prediction_watcher():
    print()
    print("====================Started prediction watcher===============")
    print()
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    delay = 0
    while True:
        try:
            if delay >= PRINT_FREQ:
                out = p_q.get_nowait()
                prediction = np.argmax(out)
                # send confident prediction
                if out[prediction] > .8:
                    print("{} {}%".format(
                        LABELS[prediction], out[prediction]*100))
                    tag = LABELS[prediction]
                    UDPClientSocket.sendto(str.encode(tag), receiveAddressPort)
                else:
                    print("None ({} {}% Below threshold)".format(
                        LABELS[prediction], out[prediction]*100))

                delay = 0
                if f_q.qsize() > 5:
                    print(
                        "Warning: Model feature queue overloaded - size = {}".format(f_q.qsize()))
        except Empty:
            pass

        delay += 1

# processes = []

def live_predict(model_path, use_holistic):
    predict_watcher = Process(target=prediction_watcher, args=())
    atexit.register(exit_handler, predict_watcher)
    predict_watcher.daemon = True
    predict_watcher.start()

    udp_server = Process(target=start_server, args=())
    atexit.register(exit_handler, udp_server)
    udp_server.daemon = True
    udp_server.start()
    # start_server()
    
    # processes.append(predict_watcher)

    predict_loop(model_path)
    # predict_processor = Process(target=predict_loop, args=(model_path,))
    # atexit.register(exit_handler, predict_processor)
    # predict_processor.start()
    # processes.append(predict_processor)


def exit_handler(p):
    try:
        p.kill()
    except:
        print("Couldn't kill")


def start_server():
    print()
    print("====================starting server===============")
    print()
    # Create a Server Instance
    UDPServerObject = socketserver.ThreadingUDPServer(ServerAddress, MyUDPRequestHandler)

    # Make the server wait forever serving connections
    UDPServerObject.serve_forever()

if __name__ == "__main__":
    if len(argv) < 2:
        model_path = CURRENT_WORKING_DIRECTORY/'models'/DEFAULT_MODEL
        if not model_path.exists():
            raise Error("NO MODEL CAN BE USED!")
    else:
        model_path = argv[1]    
    
    print(f"using model {model_path}")

    # udp_server = Process(target=start_server, args=())
    # atexit.register(exit_handler, udp_server)
    # udp_server.daemon = True
    # udp_server.start()

    # Use MP Hands only
    live_predict(model_path, False)

    