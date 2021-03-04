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

import holistic


LABELS = [None, 'A', 'B', 'C', 'Z']


PRINT_FREQ = 30
PRED_FREQ = 5

def video_loop(feature_q, prediction_q, use_holistic):
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
        prediction_q.get()
        timestamp = None
        delay = 0
        tag = ''
        print("starting image cap")
        for image, results in holistic.process_capture(cap, use_holistic):
            newtime = time.time()
            if timestamp is not None:
                diff = newtime - timestamp
                # Uncomment to print time between each frame
                #print(diff)
            timestamp = newtime

            row = holistic.to_landmark_row(results, use_holistic)
            feature_q.put(np.array(row))


            try:
                out = prediction_q.get_nowait()
                prediction = np.argmax(out)
                if delay >= PRINT_FREQ:
                    print("{} {}%".format(LABELS[prediction], out[0][prediction]*100))
                    tag = LABELS[prediction]
                    delay = 0
                    if feature_q.qsize() > 5:
                        print("Warning: Model feature queue overloaded - size = {}".format(feature_q.qsize()))
            except Empty:
                pass

            delay += 1

            holistic.draw_landmarks(image, results, use_holistic, tag)
            cv2.imshow("MediaPipe", image)


def predict_loop(feature_q, prediction_q):
        import tensorflow as tf
        import keras
        import train
        print("Starting prediction init")
        train.init_gpu()
        model = keras.models.load_model(model_path)
        print("Sending ready to video loop")
        prediction_q.put("start")

        delay = 0
        window = None
        print("Starting prediction")
        while True:
            row = feature_q.get()
            if window is None:
                window = np.zeros((train.TIMESTEPS, len(row)))

            # Discard oldest frame and append new frame to data window
            window[:-1] = window[1:]
            window[-1] = row

            if delay >= PRED_FREQ:
                out = model(np.array([window]))
                prediction_q.put(out)
                delay = 0

            delay += 1

def live_predict(model_path, use_holistic):

    f_q = Queue()
    p_q = Queue()

    p = Process(target=video_loop, args = (f_q, p_q,use_holistic,))
    atexit.register(exit_handler, p)
    p.start()
    predict_loop(f_q, p_q)

def exit_handler(p):
    try:
        p.kill()
    except:
        print("Couldn't kill video_loop")


if __name__ == "__main__":
    model_path = argv[1]

    # Use MP Hands only
    live_predict(model_path, False)
