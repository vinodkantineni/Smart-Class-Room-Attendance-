# Tools and Technologies Used

This document details every major tool, library, and framework used in the **Smart Attendance System**, explaining what it does and where it is implemented in the codebase.

## 1. Core Libraries & Frameworks

| Tool Name | Type | Function/Work | Where it is Used (Files) |
| :--- | :--- | :--- | :--- |
| **OpenCV** (`cv2`) | Computer Vision Library | - Captures video from camera.<br>- Reads/Writes images.<br>- Draws bounding boxes and text on frames.<br>- Image color conversion (BGR to RGB). | - `Main/app_ui.py` (Video loop, drawing)<br>- `Main/liveness.py` (Image preprocessing)<br>- `student_enrollment/ui/layout.py` (Camera handling)<br>- `Main/detector.py` (Image handling) |
| **CustomTkinter** (`ctk`) | UI Framework | - Provides the modern, dark-mode friendly Graphical User Interface (GUI) for the main application.<br>- Creates windows, buttons, labels, and frames. | - `Main/app_ui.py` (Main Dashboard interface) |
| **Tkinter** (`tk`) | UI Framework | - Standard Python GUI library used for the Enrollment module's interface (legacy/separate window). | - `student_enrollment/main.py`<br>- `student_enrollment/ui/layout.py` |
| **NumPy** (`numpy`) | Math Library | - Handles image data as multi-dimensional arrays (matrices).<br>- Performs efficient mathematical operations for model input preparation. | - `Main/liveness.py`<br>- `Main/app_ui.py`<br>- Everywhere image data is processed. |
| **Pillow** (`PIL`) | Image Library | - Converts OpenCV images (arrays) into a format compatible with the UI (Tkinter/CustomTkinter) for display. | - `Main/app_ui.py`<br>- `student_enrollment/ui/layout.py` |
| **Winsound** | System Library | - Generates beep sounds for audio feedback during enrollment. | - `student_enrollment/ui/layout.py` |

## 2. AI & Machine Learning Tools

| Tool Name | Type | Function/Work | Where it is Used (Files) |
| :--- | :--- | :--- | :--- |
| **ONNX Runtime** | Inference Engine | - Runs the **SCRFD** face detection model efficiently on the CPU. | - `Main/detector.py` (Implied usage for loading `.onnx` models) |
| **PyTorch** (`torch`) | ML Framework | - Loads and runs the **MiniFASNetV2** deep learning model.<br>- Performs the actual "Real vs Fake" classification inference. | - `Main/liveness.py`<br>- `Main/minifasnet.py` (Model Architecture definition) |
| **MediaPipe** | ML Solution | - **Face Mesh**: Detects 468 facial landmarks.<br>- Used specifically to calculate Eye Aspect Ratio (EAR) for **Blink Detection** (Active Liveness). | - `Main/liveness.py` |

## 3. Specific Models

| Model Name | Type | Work | File Location |
| :--- | :--- | :--- | :--- |
| **SCRFD** | AI Model (ONNX) | - **Face Detector**: Highly efficient model to locate faces in an image. Returns bounding box coordinates. | `Main/models/scrfd_2.5g_bnkps.onnx` |
| **MiniFASNetV2** | AI Model (PTH) | - **Liveness Detector**: Analyzes the texture of a cropped face to determine if it is a real skin surface or a screen/paper spoof. | `Main/models/liveness.pth` (Weights)<br>`Main/models/2.7_80x80_MiniFASNetV2.pth` (Pre-trained source) |

## 4. Hardware Interaction

| Tool | Work | Usage |
| :--- | :--- | :--- |
| **Webcam / USB Camera** | - Source of raw video data. Accessed via OpenCV VideoCapture. | Index `0` (Laptop default) or `1` (USB) in `Main/app_ui.py`. |
