import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Camera Test - press q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()