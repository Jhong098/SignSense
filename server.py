
import socketserver
import socket
import threading
from pathlib import Path
from sys import path, argv

path.insert(1, './tools')
import common, encrypt

from multiprocessing import Queue, Process, Manager, Value
from ctypes import c_char_p
from queue import Empty, Full
import atexit
from math import ceil
import numpy as np
import asyncio

import tensorflow as tf
import keras
import train

# import pickle

# print debug messages
DEBUG = True
 
# Create a tuple with IP Address and Port Number
SERVER_ADDR = ("0.0.0.0", 9999)
PRED_PORT = 9998

# current working directory
CURRENT_WORKING_DIRECTORY = Path().absolute()

DEFAULT_MODEL = list((CURRENT_WORKING_DIRECTORY/'models').iterdir())[-1]

LABELS = common.get_labels('data/')

PRINT_FREQ = 30
PRED_FREQ = 5
MAX_QUEUE_LEN = 25
CONFIDENCE_THRESHOLD = 0.4

class Message():
    def __init__(self, data, address):
        self.data = data
        self.address = address

class Error(Exception):
    pass

class InvalidIPException(Exception):
    pass

# TODO: store landmarks based on the client to handle multiple clients
class LandmarkReceiver(common.UDPRequestHandler):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def datagram_received(self, data, addr):
        # Receive and print the datagram received from client
        # print(f"received datagram from {addr}")

        try:
            datagram = data.decode()
            decrypted_data = encrypt.decrypt_chacha(datagram).decode()
            # print(decrypted_data.decode())
            landmark_arr = np.array([float(i.strip()) for i in decrypted_data.split(",")])
            # print(datagram)
            # print(landmark_arr.shape)
            self.f_q.put_nowait(landmark_arr)

            # set the IP address if it is empty
            if self.ip.value == "":
                common.print_debug_banner(f"SETTING IP TO: {addr[0]}")
                self.ip.value = addr[0]
        except Exception as e:
            # print(e)
            pass
        # except Full:
        #     # print("exception while receiving datagram")
        #     pass
        # except UnicodeDecodeError:
        #     print("unicode decode error")
        #     pass


def predict_loop(model_path, f_q, p_q, ip):
    train.init_gpu()
    model = keras.models.load_model(model_path)

    delay = 0
    window = None
    results = None
    results_len = ceil(PRINT_FREQ / PRED_FREQ)

    p_q.put("start")

    if DEBUG:
        common.print_debug_banner("STARTED PREDICTION")

    while True:
        row = f_q.get()
        
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
            delay = 0
    
        delay += 1


def prediction_watcher(f_q, p_q, ip):
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    delay = 0

    p_q.get()

    if DEBUG:
        common.print_debug_banner("STARTED PREDICTION WATCHER")

    while True:
        try:
            if delay >= PRINT_FREQ:
                out = p_q.get_nowait()
                prediction = np.argmax(out)

                # send confident prediction
                if out[prediction] > CONFIDENCE_THRESHOLD:
                    print("{} {}%".format(
                        LABELS[prediction], out[prediction]*100))
                    tag = LABELS[prediction]

                    # send back prediction if it is a valid class
                    if tag is not None:
                        if ip.value == "":
                            raise InvalidIPException("NO VALID IP WAS FOUND")

                        UDPClientSocket.sendto(tag.encode(), (ip.value, PRED_PORT))
                else:
                    print("None ({} {}% Below threshold)".format(
                        LABELS[prediction], out[prediction]*100))

                delay = 0

                if DEBUG and f_q.qsize() > 5:
                    print(
                        f"Warning: Model feature queue overloaded - size = {f_q.qsize()}")
                common.print_prediction_probs(out, LABELS)
        except Empty:
            pass

        delay += 1

def live_predict(model_path, use_holistic):
    # initialize shared IP string
    manager = Manager()
    client_ip = manager.Value(c_char_p, "")

    # queue containing the landmark features from the client
    f_q = Queue(MAX_QUEUE_LEN)

    # queue containing the predictions
    p_q = Queue(MAX_QUEUE_LEN)

    # launch watcher process for checking and sending predictions back
    predict_watcher = Process(target=prediction_watcher, args=(f_q, p_q, client_ip,))
    atexit.register(common.exit_handler, predict_watcher)
    predict_watcher.daemon = True
    predict_watcher.start()

    # launch process for predictions
    predict = Process(target=predict_loop, args=(model_path, f_q, p_q, client_ip,))
    atexit.register(common.exit_handler, predict)
    predict.daemon = True
    predict.start()
    
    # launch UDP server to receive landmark features
    asyncio.run(common.start_server(
        LandmarkReceiver(f_q=f_q, p_q=p_q, ip=client_ip),
        SERVER_ADDR
    ))

if __name__ == "__main__":
    if len(argv) < 2:
        model_path = CURRENT_WORKING_DIRECTORY/'models'/DEFAULT_MODEL
        if not model_path.exists():
            raise Error("NO MODEL CAN BE USED!")
    else:
        model_path = argv[1]    
    
    if DEBUG:
        print(f"using model {model_path}")

    live_predict(model_path, False)

    