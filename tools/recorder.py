from sys import argv
import cv2
import mediapipe as mp
import threading
import itertools
import numpy as np
import time
import signal
import sys

# mp_hands = mp.solutions.hands.Hands(
#     min_detection_confidence=0.5, min_tracking_confidence=0.3)

# mp_drawing = mp.solutions.drawing_utils

HAND_LANDMARK_COUNT = 21

start_recording = threading.Event()
stop_recording = threading.Event()
ready = threading.Event()
quit = threading.Event()

# def process_frame(frame):

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     frame.flags.writeable = False
#     landmarks = mp_hands.process(frame)
#     frame.flags.writeable = True

#     if landmarks.multi_hand_landmarks:
#         for hand_landmarks in landmarks.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame, hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)
#     cv2.imshow("MediaPipe", frame)

#     return landmarks


def signal_handler(sig, frame):
    sys.exit(0)


def process_video(cap):
    # change the file path below to the video you want to output

    # below is all choppy??
    # width = 720
    # height = 1280

    # below works at 30 FPS

    ready.set()
    mp_hands = mp.solutions.hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.3)

    while(cap.isOpened()):
        ret, image = cap.read()
        if ret == True:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            landmarks = mp_hands.process(image)
            image.flags.writeable = True
            yield (image, landmarks)

            # this makes the software wait before reading the next frame. Effectively sets the frame rate of the output video (lower the number faster it reads through the frames)
            # for a webcam it just limits the polling of the webcam
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if quit.is_set():
                break

        else:
            break

    mp_hands.close()
    cap.release()


def to_list(landmark_list, list_size):
    if landmark_list is None:
        return itertools.repeat(0.0, list_size * 3)
    return (c for landmark in landmark_list.landmark for c in [landmark.x, landmark.y, landmark.z])


def recorder():

    # change the file path below to the video you want to output
    recording = False
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    if not cap.isOpened():
        print("Error opening Camera")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp_drawing = mp.solutions.drawing_utils
    i = 0

    for image, results in process_video(cap):
        if start_recording.is_set():
            print("Recording")
            try:
                # out = cv2.VideoWriter("test/test1.mp4", cv2.VideoWriter_fourcc(
                #     'M', 'P', '4', 'V'), 30, (640, 480))

                out = cv2.VideoWriter("vid{}.mp4".format(i), fourcc, 30, (width, height))

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

        cv2.imshow('Input', image)
        if recording:
            out.write(image)
            if results.multi_hand_landmarks:
                # for hand_landmarks in results.multi_hand_landmarks:
                #     print(
                #         to_list(results.multi_hand_landmarks[0], HAND_LANDMARK_COUNT), '\n')
                detections = len(results.multi_hand_landmarks)
                landmarks = list()
                # Sometimes MP Hands detects more than 2 hands, which messes up the data format, so we limit # of hands to 2
                for hand in results.multi_hand_landmarks[:2]:
                    landmarks.extend(to_list(hand, HAND_LANDMARK_COUNT))
                landmarks.extend(itertools.repeat(0.0, HAND_LANDMARK_COUNT*2*3 - len(landmarks)))
                if len(landmarks) != HAND_LANDMARK_COUNT*2*3:
                    print("WARNING: expected row length {} but got {} instead".format(HAND_LANDMARK_COUNT*2*3, len(landmarks)))
                data.append(landmarks)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)
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
