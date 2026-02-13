# Smart Attendance System - Project Documentation

## 1. Project Overview
The **Smart Attendance System** is a computer vision-based application designed to manage student attendance using face detection and liveness verification. It consists of two main modules: **Student Enrollment** (for data collection) and **Attendance Monitoring** (real-time liveness checking).

**Current Status:**
- **Enrollment Module:** functional (captures and saves datasets).
- **Attendance Module:** functional for **Face Detection** and **Liveness Assurance** (Anti-Spoofing).
- **Note:** The specific *Face Recognition* (Identification) logic linking a live face to a stored name is currently a placeholder or pending integration.

---

## 2. Detailed Workflow

### A. Student Enrollment Workflow
This module handles the registration of new students by capturing their face data.
1.  **Initialization**: User clicks "+ New Student" in the Main Dashboard or runs `student_enrollment/main.py`.
2.  **Input**: User enters **Name** and **Registration Number**.
3.  **folder Creation**: System creates a dedicated directory structure:
    `student_enrollment/dataset/[Name_RegNo]/[Pose]/`
4.  **Data Capture**:
    - The system guides the user through multiple poses (e.g., *Front, Up, Down, Left, Right*).
    - For each pose, it captures a set number of images (e.g., 50 images).
    - Checks for face visibility before saving.
    - Provides audio feedback (beep) on successful capture.
5.  **Completion**: Once all poses are captured, the student is considered "Enrolled" (data is saved for training/embedding generation).

### B. Attendance Monitoring Workflow (Main System)
This is the core real-time monitoring interface.
1.  **Launch**: User runs `Main/app_ui.py`.
2.  **Configuration**: User selects Camera Source (Laptop or USB).
3.  **Start System**:
    - Initializes **SCRFD** (Face Detector).
    - Initializes **MiniFASNetV2** (Liveness Detector).
4.  **Real-Time Processing Loop**:
    - **Face Detection**: Locates all faces in the video frame.
    - **Liveness Analysis**:
        - **Static Analysis (MiniFASNetV2)**: Analyzes the face texture/quality to detect screens or paper masks.
        - **Active Analysis (MediaPipe)**: Checks for eye blinks and natural movement.
        - **Fusion**: Combines scores to classify face as **REAL**, **FAKE**, or **UNKNOWN**.
5.  **Visual Feedback**:
    - **Green Box**: Real Human Detected.
    - **Red Box**: Spoof/Fake Face Detected.
    - **Orange Box**: Unknown/Poor Quality.
6.  **Stats**: Updates counters for Total Faces, Real (Verified), and Spoofer (Fake).

---

## 3. Project Structure & Components

### Directory Structure
```
Smart_Attendance_System/
├── Main/                       # Core Attendance Application
│   ├── app_ui.py               # Main Graphical User Interface (CustomTkinter)
│   ├── liveness.py             # Liveness Logic (MiniFASNet + MediaPipe)
│   ├── minifasnet.py           # Anti-spoofing model architecture
│   ├── detector.py             # SCRFD Face Detection wrapper
│   └── models/                 # Model weights (.onnx, .pth)
│       ├── scrfd_2.5g_bnkps.onnx
│       └── liveness.pth
├── student_enrollment/         # Enrollment Sub-system
│   ├── main.py                 # Enrollment Entry Point
│   ├── ui/                     # Enrollment UI Logic
│   └── dataset/                # Stored Student Images
├── requirements.txt            # Python Dependencies
└── README.md
```

### Key Components

#### 1. Face Detector (SCRFD)
- **Model**: `scrfd_2.5g_bnkps.onnx`
- **Purpose**: High-efficiency face detection.
- **Library**: `onnxruntime`, `opencv-python`.

#### 2. Liveness Detector (Anti-Spoofing)
- **Model**: `MiniFASNetV2` (`liveness.pth`) + `MediaPipe FaceMesh`.
- **Purpose**: Distinguish between a live person and a photo/video attack.
- **Logic**:
    - Uses a deep learning model (MiniFASNet) to analyze texture.
    - Uses distinct feature analysis (Blink Detection) via FaceMesh landmarks.

#### 3. User Interface
- **Library**: `CustomTkinter` (Modern, dark-mode friendly wrapper for Tkinter).
- **Features**: Real-time video feed, live statistics, control panel, and enrollment wizard.

---

## 4. Dependencies
Based on `requirements.txt` and code analysis:

| Library | Purpose |
| :--- | :--- |
| **opencv-python** (`cv2`) | Image processing, camera handling, drawing bounding boxes. |
| **numpy** | Matrix operations for image data. |
| **onnxruntime** | Running the SCRFD face detection model. |
| **mediapipe** | Face Mesh landmarks for blink detection. |
| **torch** (via liveness.py) | Running the MiniFASNet pytorch model. |
| **customtkinter** | Modern GUI components. |
| **Pillow** (`PIL`) | Image manipulation for UI display. |

## 5. Technical Notes
- **Liveness Tracking**: The system tracks faces across frames (using simple coordinate matching) to smooth liveness scores over time (Temporal Smoothing), reducing flicker.
- **Model fallback**: If `liveness.pth` is missing, the system falls back to a simpler "Blink Detection Only" mode.
- **Enrollment Data**: Currently saves raw `.jpg` images. For full recognition, a training step (e.g., generating embeddings) would be required to consume this data.
