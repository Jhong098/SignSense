

import socket
import socketserver
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
import holistic

# Define the message to the server
# msgFromClient       = "Hello UDP Server"
# bytesToSend         = str.encode(msgFromClient)

# Buffer size for receiving the datagrams from server
bufferSize          = 1024

# Server IP address and Port number
serverAddressPort   = ("127.0.0.1", 9999)
receiveAddressPort = ("127.0.0.1", 9998)

# Connect2Server forms the thread - for each connection made to the server

class MyUDPRequestHandler(socketserver.DatagramRequestHandler):
    # Override the handle() method

    def handle(self):
        # Receive and print the datagram received from client
        print("Recieved one request from {}".format(self.client_address[0]))

        datagram = self.rfile.readline().strip()

        print("Datagram Received from server is:".format(datagram))
        print(datagram)

def Connect2Server(message):
    #Create a socket instance - A datagram socket
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    # Send message to server using created UDP socket
    UDPClientSocket.sendto(message, serverAddressPort)

    # Receive message from the server
    # msgFromServer = UDPClientSocket.recvfrom(bufferSize)
    # msg = "Message from Server {}".format(msgFromServer[0])

    # print(msg)


# print("Client - Main thread started")

# ThreadList  = []
# ThreadCount = 20
 

# # Create as many connections as defined by ThreadCount

# for index in range(ThreadCount):
#     ThreadInstance = threading.Thread(target=Connect2Server())
#     ThreadList.append(ThreadInstance)
#     ThreadInstance.start()

# # Main thread to wait till all connection threads are complete

# for index in range(ThreadCount):
#     ThreadList[index].join()

def video_loop(use_holistic):
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
    print("Awaiting start signal from predict")
    # prediction_q.get()
    timestamp = None
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
        # feature_q.put(np.array(row))
        send_feature_thread = threading.Thread(target=Connect2Server(np.array(row).tobytes()))
        send_feature_thread.start()
        send_feature_thread.join()

        # try:
        #     out = prediction_q.get_nowait()
        #     prediction = np.argmax(out)
        #     if delay >= PRINT_FREQ:
        #         if out[prediction] > .8:
        #             print("{} {}%".format(
        #                 LABELS[prediction], out[prediction]*100))
        #             tag = LABELS[prediction]
        #         else:
        #             print("None ({} {}% Below threshold)".format(
        #                 LABELS[prediction], out[prediction]*100))

        #         delay = 0
        #         if feature_q.qsize() > 5:
        #             print(
        #                 "Warning: Model feature queue overloaded - size = {}".format(feature_q.qsize()))
        # except Empty:
        #     pass

        delay += 1

        holistic.draw_landmarks(image, results, use_holistic, tag)
        cv2.imshow("MediaPipe", image)

def start_server():
    # Create a Server Instance
    UDPServerObject = socketserver.ThreadingUDPServer(receiveAddressPort, MyUDPRequestHandler)

    # Make the server wait forever serving connections
    UDPServerObject.serve_forever()

if __name__ == "__main__":
    # Use MP Hands only
    p = Process(target=video_loop(False))
    p.daemon = True
    atexit.register(exit_handler, p)
    p.start()

    start_server()