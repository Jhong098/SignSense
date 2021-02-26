from sys import argv
import cv2
import mediapipe as mp
import itertools
import numpy as np
import time
import signal
import sys
import keras

import holistic
import train

labels = ['A', 'B', 'C', 'Z']


def live_predict(model, use_holistic):
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

    delay = 0
    window = None
    for image, results in holistic.process_capture(cap, use_holistic):
        row = holistic.to_landmark_row(results, use_holistic)
        if window is None:
            window = np.zeros((train.TIMESTEPS, len(row)))
        # Discard oldest frame and append new frame to data window
        window[:-1] = window[1:]
        window[-1] = np.array(row)

        out = model(np.array([window]))
        prediction = np.argmax(out)

        try:
            if delay == 30:
                print("{} {}%".format(labels[prediction], out[0][prediction]))
                delay = 0
            delay += 1
        except:
            continue

        holistic.draw_landmarks(image, results, use_holistic)
        cv2.imshow("MediaPipe", image)


if __name__ == "__main__":
    model_path = argv[1]
    model = keras.models.load_model(model_path)

    # Use MP Hands only
    live_predict(model, False)
