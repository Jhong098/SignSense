
import socket
from pathlib import Path
from sys import path, argv

path.insert(1, './tools')
import common, encrypt
from holistic import normalize_features

from multiprocessing import Queue, Process, Manager
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

# current working directory
CURRENT_WORKING_DIRECTORY = Path().absolute()

DEFAULT_MODEL = list((CURRENT_WORKING_DIRECTORY/'models').iterdir())[-1]

LABELS = common.get_labels('data/')

PRINT_FREQ = 30
PRED_FREQ = 5
MAX_QUEUE_LEN = 25
CONFIDENCE_THRESHOLD = 0.6

class MissingModelException(Exception):
    pass

def array_to_class(out):
    prediction = np.argmax(out)

    # send confident prediction
    if out[prediction] > CONFIDENCE_THRESHOLD:
        print("{} {}%".format(
            LABELS[prediction], out[prediction]*100))
        tag = LABELS[prediction]

        # send back prediction if it is a valid class
        if tag is not None:
            print(f"prediction is  {tag}")
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
        self.client_to_last_msg = {}
        self.client_to_f_q = {}
        self.client_to_p_q = {}

    def cleanup_client(self, addr):
        common.print_debug_banner(f"CLEANING UP CLIENT: {addr}")
        del self.client_to_f_q[addr]
        del self.client_to_p_q[addr]
        process_to_del = self.client_to_process[addr]
        process_to_del.terminate()
        del self.client_to_process[addr]

    def check_last_msg(self, addr, new):
        time_elapsed = new - self.client_to_last_msg[addr]
        if time_elapsed > self.CLIENT_TIMEOUT:
            self.cleanup_client(addr)
        else:
            self.client_to_last_msg[addr] = new

    def start_process(self, addr):
        f_q = Queue(MAX_QUEUE_LEN)
        p_q = Queue(MAX_QUEUE_LEN)
        self.client_to_f_q[addr] = f_q
        self.client_to_p_q[addr] = p_q
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

        if addr not in self.client_to_f_q:
            self.start_process(addr)

        self.check_last_msg(addr, time.time())

        # Receive and print the datagram received from client
        try:
            if ENCRYPT:
                data = encrypt.decrypt_chacha(data)

            if len(data) < 4 and data == "END":
                self.cleanup_client(addr)

            landmark_arr = np.array([float(i.strip()) for i in data.split(",")])
            normalized_data = normalize_features(landmark_arr)
            
            self.client_to_f_q[addr].put_nowait(normalized_data)

            pred = self.client_to_p_q[addr].get_nowait()
            tag = array_to_class(pred)
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

    