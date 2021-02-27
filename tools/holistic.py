import pprint
import itertools
from sys import argv
import cv2
from cv2 import data
import mediapipe as mp
import numpy as np
import os
from google.protobuf.json_format import MessageToDict


HAND_LANDMARK_COUNT = 21
POSE_LANDMARK_COUNT = 25
LANDMARK_COUNT = HAND_LANDMARK_COUNT * 2 + POSE_LANDMARK_COUNT

TARGET_FPS = 30

# POSE_CONNECTIONS only works for whole-body pose data, not upper body
# This is only necessary for drawing landmarks not for training
UPPER_BODY_CONNECTIONS = frozenset([
    (mp.solutions.pose.PoseLandmark.NOSE,
     mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER),
    (mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER,
     mp.solutions.pose.PoseLandmark.RIGHT_EYE),
    (mp.solutions.pose.PoseLandmark.RIGHT_EYE,
     mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER),
    (mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER,
     mp.solutions.pose.PoseLandmark.RIGHT_EAR),
    (mp.solutions.pose.PoseLandmark.NOSE,
     mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER),
    (mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER,
     mp.solutions.pose.PoseLandmark.LEFT_EYE),
    (mp.solutions.pose.PoseLandmark.LEFT_EYE,
     mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER),
    (mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER,
     mp.solutions.pose.PoseLandmark.LEFT_EAR),
    (mp.solutions.pose.PoseLandmark.MOUTH_RIGHT,
     mp.solutions.pose.PoseLandmark.MOUTH_LEFT),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
     mp.solutions.pose.PoseLandmark.LEFT_SHOULDER),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
     mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
     mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
     mp.solutions.pose.PoseLandmark.RIGHT_PINKY),
    (mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
     mp.solutions.pose.PoseLandmark.RIGHT_INDEX),
    (mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
     mp.solutions.pose.PoseLandmark.RIGHT_THUMB),
    (mp.solutions.pose.PoseLandmark.RIGHT_PINKY,
     mp.solutions.pose.PoseLandmark.RIGHT_INDEX),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
     mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
     mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_WRIST,
     mp.solutions.pose.PoseLandmark.LEFT_PINKY),
    (mp.solutions.pose.PoseLandmark.LEFT_WRIST,
     mp.solutions.pose.PoseLandmark.LEFT_INDEX),
    (mp.solutions.pose.PoseLandmark.LEFT_WRIST,
     mp.solutions.pose.PoseLandmark.LEFT_THUMB),
    (mp.solutions.pose.PoseLandmark.LEFT_PINKY,
     mp.solutions.pose.PoseLandmark.LEFT_INDEX),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
     mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
     mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP,
     mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP,
     mp.solutions.pose.PoseLandmark.LEFT_HIP),
])

# tested and working, simply pip install mediapipe, numpy, and cv2

# For each video frame, yield the image and landmarks


def process_video(infile, use_holistic):
    # change the file path below to the video you want to output
    cap = cv2.VideoCapture(infile)
    if not cap.isOpened():
        print("Error opening {}".format(infile))
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return process_capture(cap, use_holistic)


def process_capture(cap, use_holistic):
    if use_holistic:
        solution = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.2, upper_body_only=True)
    else:
        solution = mp.solutions.hands.Hands(
            min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=2)

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


def draw_landmarks(image, landmarks, use_holistic):
    mp_drawing = mp.solutions.drawing_utils
    # Draw landmark annotation on the image.
    if use_holistic:
        mp_drawing.draw_landmarks(
            image, landmarks.left_hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, landmarks.right_hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, landmarks.pose_landmarks, UPPER_BODY_CONNECTIONS)
    else:

        if landmarks.multi_hand_landmarks:
            for idx, hand_handedness in enumerate(landmarks.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                hand = handedness_dict['classification'][0]["label"]
                cv2.putText(image, hand, (int(1280*landmarks.multi_hand_landmarks[idx].landmark[0].x), int(720*landmarks.multi_hand_landmarks[idx].landmark[0].y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            for hand_landmarks in landmarks.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)


def convert_video(infile, outfile, use_holistic):

    # change the file path below to the video you want to output
    cap = cv2.VideoCapture(infile)
    if not cap.isOpened():
        print("Error opening {}".format(infile))
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # first argument is the ouput file. Set to AVI, but doesn't matter since this is only for visualization
    out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
        'M', 'P', '4', 'V'), fps, (width, height))

    for image, results in process_capture(cap, use_holistic):
        # both cv2.imshow's can be omitted if you don't want to see the software work in real time.
        # cv2.imshow('Frame', image)
        draw_landmarks(image, results, use_holistic)
        cv2.putText(image, infile.split('/')[-1], (0,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('MediaPipe', image)
        out.write(image)

    out.release()
    cv2.destroyAllWindows()

# Store landmarks into a np array


def to_landmark_row(results, use_holistic):
    def to_list(landmark_list, list_size):
        if landmark_list is None:
            return itertools.repeat(0.0, list_size * 3)
        return (c for landmark in landmark_list.landmark for c in [landmark.x, landmark.y, landmark.z])

    if use_holistic:
        return list(itertools.chain(
            to_list(results.left_hand_landmarks, HAND_LANDMARK_COUNT),
            to_list(results.right_hand_landmarks, HAND_LANDMARK_COUNT),
            to_list(results.pose_landmarks, POSE_LANDMARK_COUNT),
        ))
    else:
        # NOTE THESE ARE CAMERA RELATIVE POSITIONS, RIGHT IS ACTUALLY THE SUBJECT's LEFT HAND, and vice versa
        hand_landmarks = {"Left": None, "Right": None}
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)

        # if num_hands == 0:
        # we do nothing

            if num_hands == 1:
                # get which hand was found
                hand = MessageToDict(results.multi_handedness[0])[
                    'classification'][0]["label"]
                hand_landmarks[hand] = results.multi_hand_landmarks[0]

            elif num_hands == 2:
                # this order is how 2 hands are output from mp
                hand_landmarks["Left"] = results.multi_hand_landmarks[0]
                hand_landmarks["Right"] = results.multi_hand_landmarks[1]
            elif num_hands > 2:
                exit("Too many hands")

            # output format, always paired, in order Right, Left
        return list(itertools.chain(
            # PoV left = right hand
            to_list(hand_landmarks["Left"], HAND_LANDMARK_COUNT),
            # PoV right = left hand
            to_list(hand_landmarks["Right"], HAND_LANDMARK_COUNT),
        ))


def convert_array(infile):
    # Each frame represents a row in the data.
    # Each row contains x, y, and z of all landmarks(hands and pose) associated with the frame.
    # process as holistic

    data = np.array([
        to_landmark_row(results, False) for _, results in process_video(infile,  False)
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
            vid_path = signPath+'/'+video
            print("Processing {}".format(vid_path))
            convert_datafile(vid_path,  dataPath +
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
    elif cmd == "multi":
        dir = arg1+'/mp'
        start_vid = arg2
        try:
            os.mkdir(dir)
        except:
            print("out dir already exists")
        for i, f in enumerate(os.listdir(arg1)):
            if i < int(start_vid):
                continue
            if os.path.isdir(arg1+'/'+f):
                continue
            print("showing ", i)
            convert_video(arg1+'/'+f, dir+'/mp.'+f, False)
    else:
        print("Wrong command")
