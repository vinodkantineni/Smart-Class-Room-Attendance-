# Project Libraries

This document lists all Python libraries used in the **Smart Attendance System**, including external dependencies and key standard libraries.

## 1. External Libraries (Installable via pip)

| Library Name | Work / Function | Where Used (Files) |
| :--- | :--- | :--- |
| **opencv-python** (`cv2`) | **Computer Vision**: Handles video capture, image manipulation (resize, color conversion), and drawing graphics (boxes/text) on frames. | `Main/app_ui.py`, `Main/liveness.py`, `student_enrollment/ui/layout.py`, `Main/detector.py` |
| **customtkinter** | **UI Framework**: A modern, dark-mode friendly wrapper around Tkinter. Used for the main application dashboard. | `Main/app_ui.py` |
| **mediapipe** | **Face Analysis**: Provides ready-to-use solutions for Face Mesh. Used here for extracting facial landmarks (eyes) for blink detection. | `Main/liveness.py` |
| **torch** (PyTorch) | **Deep Learning**: The engine that runs the MiniFASNet neural network for anti-spoofing (liveness) checks. | `Main/liveness.py`, `Main/minifasnet.py` |
| **onnxruntime** | **Inference Engine**: A high-performance engine for running the SCRFD face detection model (which is in .onnx format). | `Main/detector.py` |
| **numpy** | **Data Processing**: Fundamental library for matrix operations. Images in OpenCV are numpy arrays, so it's used for cropping, reshaping, and math. | `Main/liveness.py`, `Main/app_ui.py`, `Main/detector.py`, `Main/minifasnet.py` |
| **Pillow** (`PIL`) | **Image Handling**: Python Imaging Library. Converts OpenCV images into a format that Tkinter/CustomTkinter can display on the screen. | `Main/app_ui.py`, `student_enrollment/ui/layout.py` |
| **requests** | **HTTP Client**: Used for making web requests (e.g., downloading models if missing). | `Main/download_liveness.py` (if present/used) |

## 2. Key Standard Libraries (Built-in)

| Library Name | Work / Function | Where Used (Files) |
| :--- | :--- | :--- |
| **tkinter** | **Standard UI**: The built-in GUI library. Used for the Student Enrollment window (legacy module). | `student_enrollment/main.py`, `student_enrollment/ui/layout.py` |
| **winsound** | **Audio**: Windows-specific library to play simple beep sounds. Used for feedback when a photo is captured. | `student_enrollment/ui/layout.py` |
| **threading** | **Concurrency**: Runs tasks in the background to prevent the UI from freezing (e.g., during blocking operations). | `Main/app_ui.py` |
| **os / sys** | **System Operations**: File path handling, directory creation, and system path manipulation for imports. | All files (`app_ui.py`, `liveness.py`, `create_folders.py`, etc.) |
| **datetime** | **Time Handling**: Generates timestamps for logs and filenames. | `Main/app_ui.py`, `student_enrollment/ui/create_folders.py` |
