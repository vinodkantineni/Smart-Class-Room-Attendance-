# Project Dependencies

To run the **Smart Attendance System**, you need to install the following Python libraries. 

## 1. Primary Dependencies (from `requirements.txt`)
These are explicitly listed in the project's configuration files.

| Dependency | Pip Package Name | Function |
| :--- | :--- | :--- |
| **OpenCV** | `opencv-python` | Core image processing and camera access. |
| **NumPy** | `numpy` | Matrix and array mathematics (required by OpenCV/PyTorch). |
| **ONNX Runtime** | `onnxruntime` | Optimized engine to run the Face Detection model (SCRFD). |
| **MediaPipe** | `mediapipe` | Google's framework used here for Blink Detection (Face Mesh). |
| **CustomTkinter** | `customtkinter` | Modern UI styling for the main application window. |
| **Pillow** | `Pillow` | Image format handling for GUI display. |
| **Requests** | `requests` | HTTP library (likely for downloading assets). |

## 2. Critical Implied Dependencies
These are **NOT** in the `requirements.txt` but are **REQUIRED** for the code to work (based on imports in `liveness.py`).

| Dependency | Pip Package Name | Function |
| :--- | :--- | :--- |
| **PyTorch** | `torch torchvision` | **CRITICAL**: The Liveness model (`liveness.pth`) is a PyTorch model. The system tries to import `torch`, so you MUST have this installed or the Liveness feature will fail. |

## 3. Installation Command

To install everything at once, run this command in your terminal:

```bash
pip install opencv-python numpy onnxruntime mediapipe customtkinter Pillow requests torch torchvision
```

> [!NOTE]
> If you have a dedicated NVIDIA GPU, you may want to install the GPU-enabled version of PyTorch for faster performance, though the CPU version is sufficient for this project's lightweight models.
