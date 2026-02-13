# Smart Attendance System Using Face Recognition

A modern desktop application for attendance tracking using face recognition technology. This system detects faces from a live camera feed, verifies them against a registered database, and marks attendance automatically.

## 🚀 Key Features
- **Modern UI**: Clean, professional interface built with `customtkinter`.
- **Live Face Recognition**: Real-time detection and verification.
- **Attendance Logging**: Automatically records verified students with timestamps.
- **Student Enrollment**: dedicated module for capturing training images.
- **Dual Camera Support**: Switch between Laptop and USB cameras.
- **Live Statistics**: Real-time counters for detected, verified, and unknown faces.

## 📂 Project Structure (File-Wise Description)

### `Main/`
The core application logic and modern User Interface.
- **`app_ui.py`**: **(Entry Point)** The main application file containing the modern dashboard, camera integration, and attendance logic. Run this file to start the system.
- **`Camera.py`**: Legacy/Alternative camera handling script.

### `student_enrollment/`
Module responsible for registering new students and capturing face data.
- **`main.py`**: Entry point for the enrollment tool.
- **`ui/`**: Contains layout and widgets for the enrollment interface.
- **`camera/`**: Camera capture utilities for enrollment.
- **`dataset/`**: Directory where student face images are stored.
- **`utils/`**: Helper scripts for folder creation and configuration.

### Root Files
- **`requirements.txt`** (if applicable): List of Python dependencies.
- **`README.md`**: This documentation file.

## 🛠️ Installation & Setup

1. **Prerequisites**
   - Python 3.8 or higher
   - Webcam

2. **Install Dependencies**
   Run the following command in your terminal:
   ```bash
   pip install opencv-python pillow customtkinter numpy
   ```

## ▶️ How to Run

### 1. Run the Main Attendance System
To launch the modern dashboard:
```bash
python run_app.py
```

### 2. Enroll a New Student
To capture images for a new student:
```bash
cd student_enrollment
python main.py
```

## 📝 Usage Guide
1. **Select Camera**: Choose between Laptop or USB Camera from the dropdown.
2. **Start System**: Click the green **START SYSTEM** button.
3. **Attendance**: As faces are recognized, they will appear in the "Attendance Log" on the right.
4. **Stop System**: Click **STOP SYSTEM** to release the camera resource.

## 🎨 UI Overview
- **Left Panel**: Controls and Live Statistics.
- **Center**: Real-time Camera Feed.
- **Right**: Scrollable Attendance Log and "New Student" shortcut.
