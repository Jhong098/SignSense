import pprint
import itertools
from sys import argv
import cv2
from cv2 import data
import mediapipe as mp
import numpy as np
import os

HAND_LANDMARK_COUNT = 21

# tested and working, simply pip install mediapipe, numpy, and cv2

# For each video frame, yield the image and landmarks


def process_video(infile, solution_cls):
    # change the file path below to the video you want to output
    cap = cv2.VideoCapture(infile)

    solution = solution_cls(
        min_detection_confidence=0.6, min_tracking_confidence=0.3)

    if not cap.isOpened():
        print("Error opening {}".format(infile))

    while(cap.isOpened()):
        ret, image = cap.read()
        if ret == True:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            landmarks = solution.process(image)
            image.flags.writeable = True
            yield (image, landmarks)

            # this makes the software wait before reading the next frame. Effectively sets the frame rate of the output video (lower the number faster it reads through the frames)
            # for a webcam it just limits the polling of the webcam
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    solution.close()
    cap.release()

# Add landmarks onto input video and show the result


def convert_video(infile, outfile, use_holistic):
    mp_drawing = mp.solutions.drawing_utils
    # first argument is the ouput file. Set to AVI, but doesn't matter since this is only for visualization
    out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 60, (640, 480))

    if use_holistic:
        solution_cls = mp.solutions.holistic.Holistic
    else:
        solution_cls = mp.solutions.hands.Hands

    for image, results in process_video(infile, solution_cls):
        # both cv2.imshow's can be omitted if you don't want to see the software work in real time.
        cv2.imshow('Frame', image)
        # Draw landmark annotation on the image.
        if use_holistic:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp.python.solutions.holistic.POSE_CONNECTIONS)
        else:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe', image)

        out.write(image)

    out.release()
    cv2.destroyAllWindows()

# Store landmarks into a np array


def convert_array(infile):
    def to_list(landmark_list, list_size):
        if landmark_list is None:
            return itertools.repeat(0.0, list_size * 3)
        return (c for landmark in landmark_list.landmark for c in [landmark.x, landmark.y, landmark.z])

    # Each frame represents a row in the data.
    # Each row contains x, y, and z of all landmarks(hands and pose) associated with the frame.
    # process as holistic
    data = np.array([
        list(itertools.chain(
            to_list(results.left_hand_landmarks, HAND_LANDMARK_COUNT),
            to_list(results.right_hand_landmarks, HAND_LANDMARK_COUNT)
            # remove pose landmarks since prod will only use MP hands
            # to_list(results.pose_landmarks, 33)
        )) for _, results in process_video(infile, mp.solutions.holistic.Holistic)
    ])

    return data


# Store landmarks as .npy file
def convert_datafile(infile, outfile):
    np.save(outfile, convert_array(infile))


def read_datafile(infile):
    return np.load(infile)


def print_datafile(infile, rows):
    data = read_datafile(infile)
    with np.printoptions(threshold=np.inf):
        print(data[: rows])


def convert_dataset(indir, outdir):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for sign in os.listdir(indir):
        signPath = indir+'/'+sign
        dataPath = outdir+'/'+sign
        if os.path.exists(dataPath) and len(os.listdir(dataPath)) != 0:
            exit(
                "Datapath for sign {} is not empty, please ensure all data paths are removed or empty")
        elif not os.path.exists(dataPath):
            os.mkdir(dataPath)

        for video in os.listdir(signPath):
            convert_datafile(signPath+'/'+video,  dataPath +
                             '/'+video.split('.')[0])


if __name__ == "__main__":
    cmd = argv[1]
    arg1 = argv[2]
    arg2 = argv[3]

    if cmd == 'holistic_video':
        convert_video(arg1, arg2, True)
    elif cmd == 'hands_video':
        convert_video(arg1, arg2, False)
    elif cmd == 'write':
        convert_datafile(arg1, arg2)
    elif cmd == "read":
        print_datafile(arg1, int(arg2))
    elif cmd == "dataset":
        convert_dataset(indir=arg1, outdir=arg2)
    else:
        print("Wrong command")
