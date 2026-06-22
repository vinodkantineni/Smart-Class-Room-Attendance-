
import mediapipe as mp
try:
    import mediapipe.python.solutions
    print(" Explicit import successful")
    print(f"Solutions: {mp.solutions}")
except Exception as e:
    print(f"Explicit import failed: {e}")
