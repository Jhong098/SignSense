from sys import argv
import cv2
import mediapipe as mp
import threading
import itertools
import numpy as np
import time
import signal
import sys
import holistic


HAND_LANDMARK_COUNT = 21

start_recording = threading.Event()
stop_recording = threading.Event()
ready = threading.Event()
quit = threading.Event()

def signal_handler(sig, frame):
    sys.exit(0)

def recorder():
    # change the file path below to the video you want to output
    recording = False
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
    i = 0

    ready.set()
    for image, results in holistic.process_capture(cap, True):
        if quit.is_set():
            break

        if start_recording.is_set():
            print("Recording")
            try:
                out = cv2.VideoWriter("vid{}.mp4".format(i), fourcc, fps, (width, height))

            except e:
                print(e)
                exit("Error opening output file")

            outfile = "vid{}".format(i)

            data = list()
            i += 1
            recording = True
            start_recording.clear()

        if stop_recording.is_set():
            print("Stopped")

            np.save(outfile, np.array(data))
            recording = False
            out.release()
            stop_recording.clear()

        if recording:
            out.write(image)
            data.append(holistic.to_landmark_row(results))

        cv2.imshow('Input', image)
        holistic.draw_landmarks(image, results, True)
        cv2.imshow("MediaPipe", image)
    return


def close(thread):
    print("Cleaning up")
    quit.set()
    thread.join()
    exit()


if __name__ == "__main__":

    # signal.signal(signal.SIGINT, signal_handler)

    capture = threading.Thread(target=recorder)  # , daemon=True)
    capture.start()
    print("Hold on while I start the camera")
    ready.wait()
    print("Ready when you are :) (Press enter to start/stop recording)")
    try:
        while (1):
            if input() == 'q':
                close(capture)
            print("starting in:")
            for i in range(3, 0, -1):
                print(i)
                time.sleep(1)
            start_recording.set()
            input()
            stop_recording.set()
    except KeyboardInterrupt:
        close(capture)
