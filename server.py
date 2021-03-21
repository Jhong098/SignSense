
import socket
from pathlib import Path
from sys import path, argv

path.insert(1, './tools')
import common, encrypt
from holistic import normalize_features

from multiprocessing import Queue, Process, Manager
import threading
from ctypes import c_char_p
from queue import Empty
import atexit
from math import ceil
import numpy as np
import time

DEBUG = True
LOG = False
ENCRYPT = True
GPU = True

# Create a tuple with IP Address and Port Number
SERVER_ADDR = ("0.0.0.0", common.SERVER_RECV_PORT)
BLACKLIST_ADDRS = [('192.168.1.68', 9999)] # local router heartbeat thing

# current working directory
CURRENT_WORKING_DIRECTORY = Path().absolute()

DEFAULT_MODEL = list((CURRENT_WORKING_DIRECTORY/'models').iterdir())[-1]

LABELS = common.get_labels('data/')

PRINT_FREQ = 30
PRED_FREQ = 5
MAX_QUEUE_LEN = 25
CONFIDENCE_THRESHOLD = 0.6
POLL_INTERVAL = 30

class MissingModelException(Exception):
    pass

def array_to_class(out, addr, connected):
    print(f"CONNECTED: {connected}")
    prediction = np.argmax(out)

    # send confident prediction
    if out[prediction] > CONFIDENCE_THRESHOLD:
        print(f"{LABELS[prediction]} {out[prediction]*100} - {addr}")
        tag = LABELS[prediction]

        # send back prediction if it is a valid class or if the client hasn't connected
        if tag is not None or not connected:
            print(f"prediction is {tag}")
            return encrypt.encrypt_chacha(tag) if ENCRYPT else tag.encode()
    else:
        print("None ({} {}% Below threshold)".format(
            LABELS[prediction], out[prediction]*100))

# TODO: store landmarks based on the client to handle multiple clients
class LandmarkReceiver(common.UDPRequestHandler):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.CLIENT_TIMEOUT = 30 # time allowed between messages
        self.client_to_process = {}
        self.manager = Manager()
        self.client_to_last_msg = self.manager.dict()
        self.client_to_f_q = {}
        self.client_to_p_q = {}
        self.poll_connections()
        self.cleaning_process = None
        self.client_to_connected = {}

    def periodic_task(interval, times = -1):
        def outer_wrap(function):
            def wrap(*args, **kwargs):
                stop = threading.Event()
                def inner_wrap():
                    i = 0
                    while i != times and not stop.isSet():
                        stop.wait(interval)
                        function(*args, **kwargs)
                        i += 1

                t = threading.Timer(0, inner_wrap)
                t.daemon = True
                t.start()
                return stop
            return wrap
        return outer_wrap

    def cleanup_client(self, addr):
        common.print_debug_banner(f"CLEANING UP CLIENT: {addr}")
        self.cleaning_process = addr
        del self.client_to_f_q[addr]
        del self.client_to_p_q[addr]
        del self.client_to_last_msg[addr]
        del self.client_to_connected[addr]

        process_to_del = self.client_to_process[addr]
        process_to_del.terminate()

        common.print_debug_banner("FINISHED TERMINATING")
        # process_to_del.close()
        # common.print_debug_banner("FINISHED CLOSING")
        del self.client_to_process[addr]

        common.print_debug_banner(f"FINISHED CLEANUP")
        print(f"CURRENT PROCESS COUNT: {len(self.client_to_process.keys())}")
        self.cleaning_process = None
    
    @periodic_task(POLL_INTERVAL)
    def poll_connections(self):
        common.print_debug_banner(f"POLLING CONNECTIONS")
        print(f"CURRENT PROCESS COUNT: {len(self.client_to_process.keys())}")
        for client, last_msg_ts in self.client_to_last_msg.items():
            if time.time() - last_msg_ts > self.CLIENT_TIMEOUT:
                common.print_debug_banner(f"FOUND OVERTIME CLIENT: {client}")
                self.cleanup_client(client)

    def start_process(self, addr):
        f_q = Queue(MAX_QUEUE_LEN)
        p_q = Queue(MAX_QUEUE_LEN)
        self.client_to_f_q[addr] = f_q
        self.client_to_p_q[addr] = p_q
        self.client_to_connected[addr] = False
        self.client_to_last_msg[addr] = time.time()
        predict = Process(
            target=predict_loop,
            args=(
                model_path,
                f_q,
                p_q,
            )
        )
        self.client_to_process[addr] = predict
        atexit.register(common.exit_handler, predict)
        predict.daemon = True
        predict.start()
        print(f"started new predict process for {addr}")

    def datagram_received(self, data, addr):
        if addr is None:
            return

        if addr in BLACKLIST_ADDRS:
            common.print_debug_banner(f"BLOCKED {addr}")
            return

        # new client connected
        if addr not in self.client_to_f_q and addr != self.cleaning_process:
            self.start_process(addr)
            return

        self.client_to_last_msg[addr] = time.time()

        # Receive and print the datagram received from client
        try:
            if ENCRYPT:
                data = encrypt.decrypt_chacha(data)
            # received termination signal from client
            if len(data) < 4:
                if data == "END":
                    common.print_debug_banner(f"RECEIVED 'END' FROM {addr}")
                    self.client_to_f_q[addr].put("END")
                    self.cleanup_client(addr)
                elif data == "ACK":
                    common.print_debug_banner(f"RECEIVED 'ACK' FROM {addr}")
                    self.client_to_connected[addr] = True
                return

            landmark_arr = np.array([float(i.strip()) for i in data.split(",")])
            normalized_data = normalize_features(landmark_arr)
            
            self.client_to_f_q[addr].put_nowait(normalized_data)

            pred = self.client_to_p_q[addr].get_nowait()
            tag = array_to_class(pred, addr, self.client_to_connected[addr])
            self.transport.sendto(tag, addr)

        except encrypt.DecryptionError:
            print(f"tried to decrypt {data}")
        except Exception as e:
            # print(e)
            pass


def predict_loop(model_path, f_q, p_q):
    # force predict to run on CPU
    if not GPU:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    import keras
    from train import TIMESTEPS, init_gpu

    if LOG:
        import timeit
        import logging

        LOG_FILE_NAME = "logs/predict_log"
        logging.basicConfig(
            level=logging.DEBUG,
            filemode="a+",
            filename=LOG_FILE_NAME,
            format="%(message)s"
        )
        if GPU:
            logging.info(f"\n-----USING GPU------")
        else:
            logging.info(f"\n-----USING CPU------")
        
        times = []
        time_count = 0
        TIME_FREQ = 60
    
    def slide(w, new):
        # Discard oldest frame and append new frame to data window
        w[:-1] = w[1:]
        w[-1] = new
        return w

    if GPU:
        init_gpu()
    model = keras.models.load_model(model_path)

    delay = 0
    window = None
    results = None
    results_len = ceil(PRINT_FREQ / PRED_FREQ)

    if DEBUG:
        common.print_debug_banner("STARTED PREDICTION")

    while True:
        row = f_q.get()

        if len(row) == 3 and row == "END":
            break
        
        if window is None:
            window = np.zeros((TIMESTEPS, len(row)))

        window = slide(window, row)
        
        if delay >= PRED_FREQ:
            out = model(np.array([window]))

            if results is None:
                results = np.zeros((results_len, len(LABELS)))
            
            results = slide(results, out)
            pred = np.mean(results, axis=0)
            p_q.put(pred)

            delay = 0
    
        delay += 1
    
    common.print_debug_banner("ENDING PREDICT PROCESS")

def live_predict(model_path, use_holistic):
    # launch UDP server to receive landmark features
    common.start_server(
        LandmarkReceiver(),
        SERVER_ADDR
    )

if __name__ == "__main__":
    if len(argv) < 2:
        model_path = CURRENT_WORKING_DIRECTORY/'models'/DEFAULT_MODEL
        if not model_path.exists():
            raise MissingModelException("NO MODEL CAN BE USED!")
    else:
        model_path = argv[1]    
    
    if DEBUG:
        common.print_debug_banner(f"using model {model_path}")

    live_predict(model_path, False)

    