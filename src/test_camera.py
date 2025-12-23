# test_camera.py (Optimized for better FPS with lower resolution)
import cv2

def main():
    cap = cv2.VideoCapture(0)
    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Attempt to set FPS (may not work on all hardware)
    
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