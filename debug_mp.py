
import mediapipe as mp
import os
try:
    print(f"MediaPipe Version: {mp.__version__}")
    print(f"Dir(mp): {dir(mp)}")
    print(f"Location: {mp.__file__}")
except Exception as e:
    print(f"Error: {e}")
