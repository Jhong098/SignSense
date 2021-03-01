from sys import argv
import cv2
import mediapipe as mp
import itertools
import numpy as np
import time
import signal
import sys
import keras
from multiprocessing import Queue
from queue import Empty
import os

import holistic
import train

labels = [None, 'A', 'B', 'C', 'Z']
feature_q = Queue()
prediction_q = Queue()


def live_predict(model, use_holistic):
    if os.fork() != 0:
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
        delay = 0
        for image, results in holistic.process_capture(cap, use_holistic):
            newtime = time.time()
            if timestamp is not None:
                diff = newtime - timestamp
                # Uncomment to print time between each frame
                #print(diff)
            timestamp = newtime

            row = holistic.to_landmark_row(results, use_holistic)
            feature_q.put(np.array(row))

            holistic.draw_landmarks(image, results, use_holistic)
            cv2.imshow("MediaPipe", image)

            try:
                out = prediction_q.get_nowait()
                prediction = np.argmax(out)
                if delay >= 30:
                    print("{} {}%".format(labels[prediction], out[0][prediction]))
                    delay = 0
            except Empty:
                if feature_q.qsize() > 5:
                    print("Warning: Model feature queue overloaded - size = {}".format(feature_q.qsize()))

            delay += 1

    else:
        delay = 0
        window = None
        while True:
            row = feature_q.get()
            if window is None:
                window = np.zeros((train.TIMESTEPS, len(row)))

            # Discard oldest frame and append new frame to data window
            window[:-1] = window[1:]
            window[-1] = row

            if delay >= 5:
                out = model(np.array([window]))
                prediction_q.put(out)
                delay = 0

            delay += 1

if __name__ == "__main__":
    model_path = argv[1]
    model = keras.models.load_model(model_path)

    # Use MP Hands only
    live_predict(model, False)
