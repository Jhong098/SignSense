import socket
import threading
from sys import argv
import cv2
import mediapipe as mp
import itertools
import numpy as np
import time
import signal
import sys
from multiprocessing import Queue, Process
from queue import Empty
import atexit
from math import ceil

sys.path.insert(1, './tools')
import holistic, common, encrypt

PRINT_FREQ = 30
SERVER_ADDR = "35.243.169.18"

# Server IP address and Port number
serverAddressPort = (SERVER_ADDR, 9999)
receiveAddressPort = ("0.0.0.0", 9998)

APP_NAME = "SignSense"

class PredictionReceiver(common.UDPRequestHandler):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def datagram_received(self, data, addr):
        # Receive and print the datagram received from client
        # print(f"received datagram from {addr}")
        try:
            datagram = encrypt.decrypt_chacha(data)
            print(f"prediction received from server is: {datagram}")
            self.p_q.put(datagram)
        except:
            print("exception while receiving datagram")
            pass

# send UDP datagram
def Connect2Server(message):
    #Create a socket instance - A datagram socket
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Send message to server using created UDP socket
    UDPClientSocket.sendto(message, serverAddressPort)

    # Receive message from the server
    # msgFromServer = UDPClientSocket.recvfrom(bufferSize)
    # msg = "Message from Server {}".format(msgFromServer[0])


def video_loop(p_q, use_holistic=False):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    if not cap.isOpened():
        print("Error opening Camera")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Webcam FPS = {}".format(fps))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp_drawing = mp.solutions.drawing_utils

    timestamp = None
    predicted = None
    delay = 0
    tag = ''
    print("starting image cap")

    for image, results in holistic.process_capture(cap, use_holistic):
        newtime = time.time()
        if timestamp is not None:
            diff = newtime - timestamp
            # Uncomment to print time between each frame
            # print(diff)
        timestamp = newtime

        row = holistic.to_landmark_row(results, use_holistic)

        landmark_str = ','.join(np.array(row).astype(np.str))
        encrypted_str = encrypt.encrypt_chacha(landmark_str)

        # send comma delimited str of flattened landmarks in bytes to server
        send_feature_thread = threading.Thread(
            target=Connect2Server(encrypted_str)
        )
        send_feature_thread.start()
        send_feature_thread.join()

        try:
            out = p_q.get_nowait()
            if delay >= PRINT_FREQ:
                if out:
                    predicted = out
                delay = 0
        except Empty:
            pass

        delay += 1

        holistic.draw_landmarks(image, results, use_holistic, predicted)
        cv2.imshow(APP_NAME, image)

if __name__ == "__main__":
    # queue containing the returned predictions from the server
    p_q = Queue()

    # start separate process for the webcam video GUI
    p = Process(target=video_loop, args=(p_q,))
    p.daemon = True
    atexit.register(common.exit_handler, p)
    p.start()

    # run UDP server to receive predictions
    common.start_server(
        PredictionReceiver(p_q=p_q),
        receiveAddressPort
    )