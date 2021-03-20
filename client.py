import socket
from sys import argv
import cv2
import mediapipe as mp
import itertools
import numpy as np
import time
import sys
from multiprocessing import Queue, Process
from queue import Empty
import atexit
from math import ceil
from collections import deque

sys.path.insert(1, './tools')
import holistic, common, encrypt

PRINT_FREQ = 30
# SERVER_ADDR = "35.243.169.18"
SERVER_ADDR = "127.0.0.1"

# Server IP address and Port number
serverAddressPort = (SERVER_ADDR, 9999)

APP_NAME = "SignSense"

# send landmarks and receive predictions from server continuously
def server(landmark_queue, prediction_queue):
    common.print_debug_banner("STARTED SERVER")
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPClientSocket.setblocking(0)
    while True:
        try:
            landmark = landmark_queue.get()
            encrypted_landmark = encrypt.encrypt_chacha(landmark)

            # Send message to server using created UDP socket
            UDPClientSocket.sendto(encrypted_landmark, serverAddressPort)

            # Receive message from the server
            msgFromServer = UDPClientSocket.recvfrom(1024)[0]
            raw_data = encrypt.decrypt_chacha(msgFromServer)
            prediction_queue.put(raw_data)
        except encrypt.DecryptionError:
            print(f"tried to decrypt {msgFromServer}")
        except socket.error:
            pass
        except Exception as e:
            print(f"SERVER EXCEPTION: {e}")
            pass


def video_loop(landmark_queue, prediction_queue, use_holistic=False):
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
    started = False
    predicted = None
    delay = 0
    pred_history = deque([" "]*5, 5)
    pdecay = time.time()

    print("starting image cap")

    for image, results in holistic.process_capture(cap, use_holistic):
        window_state = cv2.getWindowProperty(APP_NAME, 0)
        if started and window_state == -1:
            print("QUITTING")
            break

        started = True

        newtime = time.time()
        if timestamp is not None:
            diff = newtime - timestamp
            # Uncomment to print time between each frame
            # print(diff)
        timestamp = newtime

        row = holistic.to_landmark_row(results, use_holistic)

        landmark_str = ','.join(np.array(row).astype(np.str))

        # send comma delimited str of flattened landmarks in bytes to server
        try:
            landmark_queue.put_nowait(landmark_str)
        except Exception as e:
            print(e)

        try:
            out = prediction_queue.get_nowait()
            if delay >= PRINT_FREQ:
                if out and out != pred_history[-1]:
                    pred_history.append(out)
                    pdecay = time.time()
                delay = 0
        except Empty:
            pass

        delay += 1
        if time.time() - pdecay > 7:
            pred_history = deque([" "]*5, 5)
        holistic.draw_landmarks(image, results, use_holistic, ' '.join(pred_history))
        cv2.imshow(APP_NAME, image)
    cap.release()
    cv2.destroyAllWindows()

    # send termination message to server
    landmark_queue.put("END")

if __name__ == "__main__":
    # queue containing the returned predictions from the server
    landmark_queue, prediction_queue = Queue(), Queue()

    # start separate process for the webcam video GUI
    server_p = Process(target=server, args=(landmark_queue, prediction_queue, ))
    server_p.daemon = True
    atexit.register(common.exit_handler, server_p)
    server_p.start()

    video_p = Process(target=video_loop, args=(landmark_queue, prediction_queue, ))
    video_p.daemon = True
    atexit.register(common.exit_handler, video_p)
    video_p.start()

    video_p.join()
