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
            # print(diff)
        timestamp = newtime

        row = holistic.to_landmark_row(results, use_holistic)
        feature_q.put(np.array(row))

        try:
            out = prediction_q.get_nowait()
            prediction = np.argmax(out)
            if delay >= PRINT_FREQ:
                if out[prediction] > .8:
                    print("{} {}%".format(
                        LABELS[prediction], out[prediction]*100))
                    tag = LABELS[prediction]
                else:
                    print("None ({} {}% Below threshold)".format(
                        LABELS[prediction], out[prediction]*100))

                delay = 0
                if feature_q.qsize() > 5:
                    print(
                        "Warning: Model feature queue overloaded - size = {}".format(feature_q.qsize()))
        except Empty:
            pass

        delay += 1

        holistic.draw_landmarks(image, results, use_holistic, tag)
        cv2.imshow("MediaPipe", image)

if __name__ == "__main__":
    # Use MP Hands only
    f_q = Queue()
    p = Process(target=video_loop, args=(f_q, p_q, use_holistic,))
    atexit.register(exit_handler, p)
    p.start()