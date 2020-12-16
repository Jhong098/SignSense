import cv2
import mediapipe as mp
#tested and working, simply pip install mediapipe and cv2

# change the file path below to the video you want to output
cap = cv2.VideoCapture('/home/jack/Downloads/scene6-camera1(1).mov')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.8, smooth_landmarks=True)

if not cap.isOpened():
    print("Error opening")

# first argument is the ouput file. Set to AVI, but doesn't matter since this is only for visualization
out = cv2.VideoWriter('output2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 60, (640,480))

while(cap.isOpened()):
    ret, image = cap.read()
    if ret == True:
        #both cv2.imshow's can be omitted if you don't want to see the software work in real time.
        cv2.imshow('Frame', image)
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        # mp_drawing.draw_landmarks(
        #     image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Holistic', image)
        
        out.write(image)

        # this makes the software wait before reading the next frame. Effectively sets the frame rate of the output video (lower the number faster it reads through the frames)
        # for a webcam it just limits the polling of the webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
    
    else:
        break

holistic.close()
cap.release()
out.release
cv2.destroyAllWindows()