import cv2
import numpy as np
import sys
import os

# Add root to path so we can import Main
sys.path.append(os.getcwd())

try:
    from Main.liveness import LivenessDetector
    print("Success: Imported LivenessDetector")
except ImportError as e:
    print(f"Error: Failed to import LivenessDetector: {e}")
    sys.exit(1)

def create_dummy_face():
    # Create a 200x200 grey image simulating a face
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (200, 200, 200), -1) # Face
    return img

def test_logic():
    # Initialize with a dummy model path (it will fail to load model and go to fallback, which is fine for logic testing)
    detector = LivenessDetector("dummy/path/to/model.pth")
    
    print("\n--- Testing Logic (Fallback Mode) ---")
    
    # 1. Test Low Quality (Black Image)
    black_img = np.zeros((300, 300, 3), dtype=np.uint8)
    bbox = (50, 50, 250, 250)
    label, score, smoothed, details = detector.predict(black_img, bbox, face_id="test1")
    print(f"Test 1 (Black Image): Label={label} (Expected: UNCERTAIN/UNKNOWN), Reason={details.get('quality_reason')}")
    
    # 2. Test Consecutive Logic (Simulate Blink)
    # Since we don't have a model, it relies on blink.
    # We can't easily simulate a blink for MediaPipe without a real face image that MP recognizes.
    # However, we can check if the code runs without crashing.
    
    face_img = create_dummy_face()
    print("\nTest 2 (Dummy Face - Run Cycle)")
    for i in range(5):
        label, score, smoothed, details = detector.predict(face_img, bbox, face_id="test2")
        print(f"Frame {i+1}: Label={label}, Consecutive={details.get('consecutive_frames')}")

    print("\nLogic checks passed (Code executed without crash).")

if __name__ == "__main__":
    test_logic()
