
import socket
from pathlib import Path
from sys import path, argv

path.insert(1, './tools')
import common, encrypt
from holistic import normalize_features

from multiprocessing import Queue, Process, Manager, Value
from ctypes import c_char_p
from queue import Empty
import atexit
from math import ceil
import numpy as np

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

    def datagram_received(self, data, addr):
        # Receive and print the datagram received from client
        try:
            if ENCRYPT:
                data = encrypt.decrypt_chacha(data)
            landmark_arr = np.array([float(i.strip()) for i in data.split(",")])
            normalized_data = normalize_features(landmark_arr)
            
            self.f_q.put_nowait(normalized_data)

            pred = self.p_q.get_nowait()
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
            if LOG:
                start = timeit.default_timer()
            out = model(np.array([window]))

            if LOG:
                stop = timeit.default_timer()

            if results is None:
                results = np.zeros((results_len, len(LABELS)))
            elif LOG:
                times.append(stop-start)
            
            if LOG and time_count >= TIME_FREQ:
                logging.info(f"\nPREDICTION TAKES: {sum(times)/len(times)}s")
                time_count = 0
                times = []

            results = slide(results, out)
            pred = np.mean(results, axis=0)
            p_q.put(pred)

            delay = 0
    
        delay += 1
        if LOG:
            time_count += 1

def live_predict(model_path, use_holistic):
    # initialize shared IP string
    # manager = Manager()
    # client_ip = manager.Value(c_char_p, "")

    # queue containing the landmark features from the client
    f_q = Queue(MAX_QUEUE_LEN)

    # queue containing the predictions
    p_q = Queue(MAX_QUEUE_LEN)

    # launch process for predictions
    predict = Process(target=predict_loop, args=(model_path, f_q, p_q,))
    atexit.register(common.exit_handler, predict)
    predict.daemon = True
    predict.start()
    
    # launch UDP server to receive landmark features
    common.start_server(
        LandmarkReceiver(f_q=f_q, p_q=p_q),
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

    