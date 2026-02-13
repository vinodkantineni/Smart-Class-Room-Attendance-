import cv2
import sys

print("Testing camera access...")
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera 0")
    else:
        print("Camera 0 opened successfully")
        ret, frame = cap.read()
        if ret:
            print(f"Frame captured: {frame.shape}")
        else:
            print("Failed to read frame")
        cap.release()
except Exception as e:
    print(f"Exception: {e}")

print("Testing Haar Cascade load...")
try:
    path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Loading cascade from: {path}")
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        print("Failed to load cascade")
    else:
        print("Cascade loaded successfully")
except Exception as e:
    print(f"Exception loading cascade: {e}")
