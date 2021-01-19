import pprint
import itertools
from sys import argv
import cv2
import mediapipe as mp
import numpy as np
#tested and working, simply pip install mediapipe, numpy, and cv2

# For each video frame, yield the image and landmarks
def process_video(infile):
    # change the file path below to the video you want to output
    cap = cv2.VideoCapture(infile)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.8, smooth_landmarks=True)

    if not cap.isOpened():
        print("Error opening")

    while(cap.isOpened()):
        ret, image = cap.read()
        if ret == True:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            landmarks = holistic.process(image)
            image.flags.writeable = True
            yield (image, landmarks)

            # this makes the software wait before reading the next frame. Effectively sets the frame rate of the output video (lower the number faster it reads through the frames)
            # for a webcam it just limits the polling of the webcam
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    holistic.close()
    cap.release()

# Add landmarks onto input video and show the result
def convert_video(infile, outfile):
    mp_drawing = mp.solutions.drawing_utils
    # first argument is the ouput file. Set to AVI, but doesn't matter since this is only for visualization
    out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('M','J','P','G'), 60, (640,480))

    for image, results in process_video(infile):
        #both cv2.imshow's can be omitted if you don't want to see the software work in real time.
        cv2.imshow('Frame', image)
        # Draw landmark annotation on the image.
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp.python.solutions.holistic.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Holistic', image)

        out.write(image)

    out.release()
    cv2.destroyAllWindows()

# Store landmarks as .npy file
def convert_datafile(infile, outfile):
    def to_list(landmark_list, list_size):
        if landmark_list is None:
            return itertools.repeat([0.0, 0.0], list_size)
        return ([landmark.x, landmark.y] for landmark in landmark_list.landmark)

    # Each frame represents a row in the data.
    # Each row contains all landmarks (hands and pose) associated with the frame.
    # Each landmark is a [x, y] pair
    data = np.array([
       list(itertools.chain(
           to_list(results.left_hand_landmarks, 21),
           to_list(results.right_hand_landmarks, 21),
           to_list(results.pose_landmarks, 33)
        )) for _, results in process_video(infile)
    ])

    np.save(outfile, data)

def read_datafile(infile, rows):
    data = np.load(infile)
    with np.printoptions(threshold=np.inf):
        print(data[:rows])

if __name__ == "__main__":
    cmd = argv[1]
    arg1 = argv[2]
    arg2 = argv[3]

    if cmd == 'video':
        convert_video(arg1, arg2)
    elif cmd == 'write':
        convert_datafile(arg1, arg2)
    elif cmd == "read":
        read_datafile(arg1, int(arg2))
    else:
        print("Wrong command")
