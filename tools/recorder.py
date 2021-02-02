from sys import argv
import cv2
import mediapipe as mp
import threading


mp_hands = mp.solutions.hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.3)

mp_drawing = mp.solutions.drawing_utils

start_recording = threading.Event()
stop_recording = threading.Event()
ready = threading.Event()


def process_video(frame):

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    landmarks = mp_hands.process(frame)
    frame.flags.writeable = True

    if landmarks.multi_hand_landmarks:
        for hand_landmarks in landmarks.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp.python.solutions.holistic.HAND_CONNECTIONS)
    cv2.imshow("MediaPipe", frame)


def test():
    try:
        # change the file path below to the video you want to output
        recording = False
        try:
            cap = cv2.VideoCapture(0,  cv2.CAP_DSHOW)
            # cap = cv2.VideoCapture(0)
        except Exception as e:
            print("Error opening input source")
            return

        if not cap.isOpened():
            print("Error opening camera")

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # below is all choppy??
        # width = 720
        # height = 1280

        # below works at 30 FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 60)

        i = 0
        ready.set()
        while(cap.isOpened()):
            if start_recording.is_set():
                print("Recording")
                try:
                    # out = cv2.VideoWriter("test/test1.mp4", cv2.VideoWriter_fourcc(
                    #     'M', 'P', '4', 'V'), 30, (640, 480))
                    out = cv2.VideoWriter("test{}.mp4".format(i), cv2.VideoWriter_fourcc(
                        *'MJPG'), 60, (width, height))
                    i += 1
                except:
                    exit("Error opening output file")
                recording = True
                start_recording.clear()

            if stop_recording.is_set():
                print("Stopped")

                recording = False
                out.release()
                stop_recording.clear()

            ret, image = cap.read()
            if ret == True:
                image.flags.writeable = False
                image.flags.writeable = True
                cv2.imshow('Input', image)
                process_video(image)
                # get mediapipe results

                # show media pipe window

                # this makes the software wait before reading the next frame. Effectively sets the frame rate of the output video (lower the number faster it reads through the frames)
                # for a webcam it just limits the polling of the webcam
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if recording:
                    out.write(image)

            else:
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    capture = threading.Thread(target=test, daemon=True)
    capture.start()
    print("Hold on while I start the camera")
    ready.wait()
    print("Ready when you are :)")
    try:
        while (1):
            input()
            start_recording.set()
            input()
            stop_recording.set()
    except KeyboardInterrupt:
        mp_hands.close()
        exit()
