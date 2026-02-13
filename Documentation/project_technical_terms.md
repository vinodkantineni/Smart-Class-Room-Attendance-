# Technical Terms & Concepts

This document explains the specific technical terms and concepts used within the **Smart Attendance System** codebase.

## 1. Computer Vision Concepts

| Term | Definition | Work in this Project | Where Used |
| :--- | :--- | :--- | :--- |
| **Bounding Box (BBox)** | A rectangular box defined by coordinates (x1, y1, x2, y2) that encloses a detected object. | Used to define exactly where a face is located in the video frame so we can crop it and analyze it. | `Main/detector.py` (output), `Main/app_ui.py` (drawing), `Main/liveness.py` (input) |
| **Landmarks** | Specific key points on a face (e.g., corners of eyes, tip of nose). | Used to calculate the "Eye Aspect Ratio" to detect if a person is blinking (Active Liveness). | `Main/liveness.py` (via MediaPipe) |
| **Eye Aspect Ratio (EAR)** | A mathematical formula that calculates the openness of an eye based on the distance between vertical and horizontal eye landmarks. | Determines if an eye is open or closed. A sudden drop in EAR indicates a blink, verifying the user is live. | `Main/liveness.py` (logic inside `predict`) |
| **Preprocessing** | The manipulation of raw data (images) to make it suitable for a model (e.g., resizing, normalizing colors). | Converts the raw camera image into the specific 80x80 format required by the MiniFASNet model. | `Main/liveness.py` (resize, transpose), `Main/detector.py` |

## 2. AI / Model Terminology

| Term | Definition | Work in this Project | Where Used |
| :--- | :--- | :--- | :--- |
| **Inference** | The process of using a trained machine learning model to make a prediction on new data. | The system "infers" whether a face is Real or Fake every time a video frame is processed. | `Main/liveness.py` (`self.model(tensor_input)`), `Main/detector.py` |
| **Tensor** | A multi-dimensional array used as the standard input format for Deep Learning models (like PyTorch). | The image data is converted from a NumPy array to a PyTorch Tensor before being sent to the Liveness model. | `Main/liveness.py` |
| **Logits / Softmax** | Raw output scores from a model (logits) and the function to convert them into probabilities (softmax). | Converts the model's raw numbers into a readable percentage (e.g., "98% Real"). | `Main/liveness.py` (`F.softmax`) |
| **Temporal Smoothing** | A technique to stabilize predictions by averaging results over several frames instead of relying on a single frame. | Prevents the "Real/Fake" label from flickering rapidly by keeping a history of the last 5 frames. | `Main/liveness.py` (`deque` usage) |
| **ONNX** (Open Neural Network Exchange) | A cross-platform file format for AI models. | Allows the **SCRFD** face detector to run very fast on the CPU without needing heavy dependencies. | `Main/models/scrfd...onnx`, `Main/detector.py` |

## 3. Deployment / System Terms

| Term | Definition | Work in this Project | Where Used |
| :--- | :--- | :--- | :--- |
| **GUI** (Graphical User Interface) | The visual part of the app (buttons, windows) that allows users to interact with the code. | The "Dashboard" the user sees, built with CustomTkinter. | `Main/app_ui.py` |
| **Threading** | Running multiple tasks at the same exact time. | Although not heavily used for the AI logic itself, the UI loop runs separately from the OS listeners to keep the window responsive. | `Main/app_ui.py` |
| **FPS** (Frames Per Second) | The frequency at which consecutive images (frames) appear on the display. | The speed of the video feed. The system tries to process frames as fast as possible to maintain smooth video. | `Main/app_ui.py` (implicit in `update_video_feed`) |
