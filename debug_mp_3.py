
try:
    import mediapipe.solutions
    print("Success: import mediapipe.solutions")
except ImportError as e:
    print(f"Fail: import mediapipe.solutions: {e}")

try:
    from mediapipe import solutions
    print("Success: from mediapipe import solutions")
except ImportError as e:
    print(f"Fail: from mediapipe import solutions: {e}")
