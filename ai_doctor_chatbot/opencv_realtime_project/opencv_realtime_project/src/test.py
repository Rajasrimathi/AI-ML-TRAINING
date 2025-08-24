import cv2

for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        print("Testing backend:", backend)
        for i in range(50):  # show 50 frames
            ok, frame = cap.read()
            if ok:
                cv2.imshow("Test Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
