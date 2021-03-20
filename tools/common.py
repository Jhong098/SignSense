import asyncio
from pathlib import Path
import random
import time

SERVER_RECV_PORT = 9999
CLIENT_RECV_PORT = 9998

class UDPRequestHandler(asyncio.DatagramProtocol):
    def __init__(self):
        super().__init__()
    
    def connection_made(self, transport):
        print(f"NEW CONNECTION")
        self.transport = transport


def start_server(handler, addr):
    print_debug_banner("STARTING SERVER")

    loop = asyncio.get_event_loop()
    transport = loop.create_datagram_endpoint(
        lambda: handler,
        local_addr = addr
    )

    loop.run_until_complete(transport)
    loop.run_forever()


# handle SIGKILL
def exit_handler(p):
    try:
        p.kill()
    except:
        print("Couldn't kill")


def get_labels(dirname):
    holds = [sign.name for sign in Path(dirname, 'holds_data').iterdir()]
    nonholds = [sign.name for sign in Path(dirname, 'nonholds_data').iterdir()]
    return [None] + sorted(holds + nonholds)


def print_debug_banner(msg):
    print(f"\n================={msg}===============\n")


def print_prediction_probs(out, labels):
    print("--> ", end='')
    for i, label in enumerate(out):
        print("{}:{:.2f}% | ".format(labels[i], label*100), end='')
    print("\n")


# def predict_loop(
#     model_path,
#     client_to_f_q,
#     client_to_p_q,
# ):
#     # force predict to run on CPU
#     if not GPU:
#         import os
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#     import tensorflow as tf
#     import keras
#     from train import TIMESTEPS, init_gpu

#     if LOG:
#         import timeit
#         import logging

#         LOG_FILE_NAME = "logs/predict_log"
#         logging.basicConfig(
#             level=logging.DEBUG,
#             filemode="a+",
#             filename=LOG_FILE_NAME,
#             format="%(message)s"
#         )
#         if GPU:
#             logging.info(f"\n-----USING GPU------")
#         else:
#             logging.info(f"\n-----USING CPU------")
        
#         times = []
#         time_count = 0
#         TIME_FREQ = 60
    
#     def slide(w, new):
#         # Discard oldest frame and append new frame to data window
#         w[:-1] = w[1:]
#         w[-1] = new
#         return w

#     def predict_window(window):
#         out = model(np.array([window]))

#         if results is None:
#             results = np.zeros((results_len, len(LABELS)))

#         results = slide(results, out)
#         pred = np.mean(results, axis=0)
#         return pred

#     if GPU:
#         init_gpu()
#     model = keras.models.load_model(model_path)

#     results = None
#     results_len = ceil(PRINT_FREQ / PRED_FREQ)

#     if DEBUG:
#         common.print_debug_banner("STARTED PREDICTION")

#     while True:
#         clients = client_to_f_q.keys()
#         # print(f"clients: {clients}")
#         for client in clients:
#             print(f"processing client: {client}")
#             try:
#                 row = client_to_f_q[client].get_nowait()
#             except Empty:
#                 continue
            
#             window = client_to_window[client]
#             delay = client_to_delay[client]

#             if window is None:
#                 window = np.zeros((TIMESTEPS, len(row)))
#             window = slide(window, row)
#             client_to_window[client] = window
            
#             if delay >= PRED_FREQ:
#                 pred = predict_window(window)
#                 print(f"sending prediction for client {client}")
#                 client_to_p_q[client].put(pred)
#                 client_to_delay[client] = 0
#                 continue
        
#             with client_to_delay[client].get_lock():
#                 client_to_delay[client] += 1


        # if addr not in self.client_to_f_q:
        #     common.print_debug_banner(f"REGISTERING NEW CONNECTION IN client_to_f_q")
        #     self.client_to_f_q[addr] = Queue(MAX_QUEUE_LEN)

        # if addr not in self.client_to_p_q:
        #     common.print_debug_banner(f"REGISTERING NEW CONNECTION IN client_to_p_q")
        #     self.client_to_p_q[addr] = Queue(MAX_QUEUE_LEN)

        # if addr not in self.client_to_window:
        #     common.print_debug_banner(f"REGISTERING NEW CONNECTION IN client_to_window")
        #     self.client_to_window[addr] = None

        # if addr not in self.client_to_delay:
        #     common.print_debug_banner(f"REGISTERING NEW CONNECTION IN client_to_delay")
        #     self.client_to_delay[addr] = Value('i', 0)