import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import time
import datetime
import os
import numpy as np
import subprocess
import sys

from Main.detector import SCRFD
from Main.liveness import LivenessDetector
from Main.recognition import FaceRecognizer
from Main.camera_manager import CameraSource

import concurrent.futures

# Append student_enrollment to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
enrollment_path = os.path.join(project_root, "student_enrollment")
if enrollment_path not in sys.path:
    sys.path.append(enrollment_path)

# --- Configuration ---
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

APP_WIDTH = 1200
APP_HEIGHT = 700

class SmartAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Smart Attendance System")
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.minsize(1000, 600)
        
        # State
        self.debug_mode = ctk.BooleanVar(value=False)
        self.marked_students = set() # Track attendance for this session

        # Initialize Detector
        model_file = os.path.join("Main", "models", "scrfd_2.5g_bnkps.onnx")
        self.detector = SCRFD(model_path=model_file)
        
        # Initialize Liveness
        # Initialize Liveness
        liveness_model_file = os.path.join("Main", "models", "liveness.pth")
        self.liveness_detector = LivenessDetector(model_path=liveness_model_file)

        # Initialize Recognition (Dataset Scan)
        dataset_dir = os.path.join("dataset") # or project_root/dataset
        if not os.path.exists(dataset_dir) and os.path.exists(os.path.join(project_root, "dataset")):
             dataset_dir = os.path.join(project_root, "dataset")
             
        self.face_recognizer = FaceRecognizer(dataset_path=dataset_dir)

        # Layout Configuration (3 Columns)
        self.grid_columnconfigure(1, weight=1) # Center expands
        self.grid_rowconfigure(1, weight=1)    # Content row expands

        # --- Header ---
        self.header_frame = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="#E0E0E0")
        self.header_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        
        self.title_label = ctk.CTkLabel(self.header_frame, text="Smart Attendance System", 
                                        font=ctk.CTkFont(family="Roboto", size=24, weight="bold"))
        self.title_label.pack(pady=10)

        # --- Left Panel (Control Panel) ---
        self.left_panel = ctk.CTkFrame(self, width=250, corner_radius=10)
        self.left_panel.grid(row=1, column=0, sticky="ns", padx=10, pady=10)
        self.left_panel.grid_propagate(False)

        # Left Panel Components
        self.lbl_control = ctk.CTkLabel(self.left_panel, text="Control Panel", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_control.pack(pady=(20, 10))

        # Camera status labels
        self.lbl_cam_status = ctk.CTkLabel(self.left_panel, text="Wait for Start...", text_color="gray")
        self.lbl_cam_status.pack(pady=5)
        
        self.btn_start = ctk.CTkButton(self.left_panel, text="START SYSTEM", fg_color="green", hover_color="#006400", command=self.start_system)
        self.btn_start.pack(pady=10, padx=20, fill="x")

        self.btn_stop = ctk.CTkButton(self.left_panel, text="STOP SYSTEM", fg_color="red", hover_color="#8B0000", command=self.stop_system, state="disabled")
        self.btn_stop.pack(pady=10, padx=20, fill="x")
        
        # Debug Toggle
        self.chk_debug = ctk.CTkCheckBox(self.left_panel, text="Visual Debug Mode", variable=self.debug_mode)
        self.chk_debug.pack(pady=10, padx=20, anchor="w")

        # Live Stats
        self.stats_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.stats_frame.pack(pady=30, padx=10, fill="x")
        
        ctk.CTkLabel(self.stats_frame, text="Live Stats", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(0, 5))
        self.lbl_total_faces = ctk.CTkLabel(self.stats_frame, text="Total Faces: 0", anchor="w")
        self.lbl_total_faces.pack(fill="x", pady=2)
        self.lbl_verified = ctk.CTkLabel(self.stats_frame, text="Verified: 0", text_color="green", anchor="w")
        self.lbl_verified.pack(fill="x", pady=2)
        self.lbl_fake = ctk.CTkLabel(self.stats_frame, text="Fake/Spoof: 0", text_color="red", anchor="w")
        self.lbl_fake.pack(fill="x", pady=2)
        self.lbl_unknown = ctk.CTkLabel(self.stats_frame, text="Unknown: 0", text_color="orange", anchor="w")
        self.lbl_unknown.pack(fill="x", pady=2)


        # --- Center Panel (Top Buttons + Camera Preview) ---
        self.center_panel = ctk.CTkFrame(self, corner_radius=10, fg_color="#F0F0F0")
        self.center_panel.grid(row=1, column=1, sticky="nsew", padx=5, pady=10)
        
        # Grid layout: Row 0 = Camera Toggle, Row 1 = Video (Expand)
        self.center_panel.columnconfigure(0, weight=1)
        self.center_panel.rowconfigure(0, weight=0) # Control Bar
        self.center_panel.rowconfigure(1, weight=1) # Video
        
        # 1. Camera Toggle Bar (Row 0)
        self.camera_select_frame = ctk.CTkFrame(self.center_panel, height=50, corner_radius=5, fg_color="transparent")
        self.camera_select_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(5,0))
        # self.camera_select_frame.grid_propagate(False) # Let it shrink to content height

        # Buttons - Left and Right aligned as per sketch
        self.btn_cam1 = ctk.CTkButton(self.camera_select_frame, text="Camera 1", width=120, height=40, font=ctk.CTkFont(size=14, weight="bold"),
                                      corner_radius=5, command=lambda: self.select_camera(0))
        self.btn_cam1.pack(side="left", padx=25, pady=5)
        
        self.btn_cam2 = ctk.CTkButton(self.camera_select_frame, text="Camera 2", width=120, height=40, font=ctk.CTkFont(size=14, weight="bold"),
                                      corner_radius=5, command=lambda: self.select_camera(1))
        self.btn_cam2.pack(side="right", padx=25, pady=5)

        # 2. Main Video Display (Row 1)
        self.video_display = ctk.CTkLabel(self.center_panel, text="Press Start", bg_color="black", corner_radius=5)
        self.video_display.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)


        # --- Right Panel (Attendance & Student) ---
        self.right_panel = ctk.CTkFrame(self, width=280, corner_radius=10)
        self.right_panel.grid(row=1, column=2, sticky="ns", padx=10, pady=10)
        self.right_panel.grid_propagate(False)

        self.btn_new_student = ctk.CTkButton(self.right_panel, text="+ New Student", height=40, command=self.open_student_enrollment)
        self.btn_new_student.pack(pady=20, padx=20, fill="x")

        self.lbl_log = ctk.CTkLabel(self.right_panel, text="Attendance Log", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_log.pack(pady=(10, 5))

        self.log_scroll = ctk.CTkScrollableFrame(self.right_panel, label_text="Recent Entries")
        self.log_scroll.pack(expand=True, fill="both", padx=10, pady=10)

        # Variables
        self.running = False
        self.camera_sources = [] # List of CameraSource objects
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.active_cam_index = 0 # Default to Cam 1
        
        # Set Initial Button State
        self.update_tab_buttons()

    def select_camera(self, index):
        self.active_cam_index = index
        self.update_tab_buttons()
        # Immediate cleanup of display if needed, but next frame update will handle it
        
    def update_tab_buttons(self):
        # Highlight active camera
        if self.active_cam_index == 0:
            self.btn_cam1.configure(fg_color="#1f6aa5") # Theme Blue (Active)
            self.btn_cam2.configure(fg_color="gray")
        else:
            self.btn_cam1.configure(fg_color="gray")
            self.btn_cam2.configure(fg_color="#1f6aa5") # Theme Blue (Active)


    def add_log_entry(self, name, roll_no):
        # Deduplication Logic
        unique_id = roll_no if roll_no != "N/A" else name
        if unique_id in self.marked_students:
            return # Already marked

        self.marked_students.add(unique_id)

        entry_frame = ctk.CTkFrame(self.log_scroll, height=40)
        entry_frame.pack(fill="x", pady=2)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        lbl = ctk.CTkLabel(entry_frame, text=f"{name} ({roll_no}) - {timestamp}", anchor="w", padx=10)
        lbl.pack(side="left", fill="x", expand=True)

    def open_student_enrollment(self):
        # 1. Release Main Camera
        self.stop_system()
        
        # 2. Hide Main Interface
        self.hide_main_ui()

        # 3. Create Enrollment Container
        self.enrollment_container = ctk.CTkFrame(self, fg_color="#F0F0F0")
        self.enrollment_container.grid(row=0, column=0, rowspan=2, columnspan=3, sticky="nsew")

        # 4. Back Button
        btn_back = ctk.CTkButton(self.enrollment_container, text="< Back to Dashboard", 
                                 width=150, height=30, fg_color="gray", command=self.close_enrollment)
        btn_back.pack(anchor="nw", padx=20, pady=10)

        # 5. Inner Frame for AppLayout
        self.inner_enrollment_frame = ctk.CTkFrame(self.enrollment_container, fg_color="white")
        self.inner_enrollment_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # 6. Initialize Enrollment Logic
        try:
            from Main.ui.layout import AppLayout
            # AppLayout expects a root-like object. We pass our inner frame.
            self.enrollment_app = AppLayout(self.inner_enrollment_frame)
            
            # Monkey-patch on_close to call our cleanup
            # (Though AppLayout doesn't have a button to trigger on_close, we trigger it via Back button)
            
        except Exception as e:
            print(f"Failed to load enrollment module: {e}")
            self.close_enrollment()

    def close_enrollment(self):
        if hasattr(self, 'enrollment_app'):
            # This releases the enrollment camera and destroys inner_frame
            try:
                self.enrollment_app.on_close() 
            except Exception as e:
                print(f"Error closing enrollment app: {e}")
        
        if hasattr(self, 'enrollment_container'):
            self.enrollment_container.destroy()
            del self.enrollment_container
        
        self.show_main_ui()
        pass

    def hide_main_ui(self):
        self.header_frame.grid_forget()
        self.left_panel.grid_forget()
        self.center_panel.grid_forget()
        self.right_panel.grid_forget()

    def show_main_ui(self):
        self.header_frame.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.left_panel.grid(row=1, column=0, sticky="ns", padx=10, pady=10)
        self.center_panel.grid(row=1, column=1, sticky="nsew", padx=5, pady=10)
        self.right_panel.grid(row=1, column=2, sticky="ns", padx=10, pady=10)

    def start_system(self):
        if not self.running:
            self.running = True
            self.btn_start.configure(state="disabled", fg_color="gray")
            self.btn_stop.configure(state="normal", fg_color="red")
            
            # Start Cameras
            # Try index 0 and 1. If 1 fails, maybe us 0 twice or just one?
            # User workflow implies 2 cameras.
            self.camera_sources = []
            
            # Camera 1
            cam1 = CameraSource(0, "Cam 1")
            if cam1.start():
                self.camera_sources.append(cam1)
            else:
                print("Camera 1 failed.")

            # Camera 2
            cam2 = CameraSource(1, "Cam 2")
            if cam2.start():
                self.camera_sources.append(cam2)
            else:
                print("Camera 2 failed.")

            self.lbl_cam_status.configure(text=f"Active Inputs: {len(self.camera_sources)}", text_color="green")
            
            # Initialize Futures list explicitly matching max cameras
            # We will use a dictionary or list mapping index -> future
            self.cam_futures = {} # {cam_index: future}
            
            self.update_video_feed()

    def stop_system(self):
        self.running = False
        
        # Stop Cameras
        for cam in self.camera_sources:
            cam.stop()
        self.camera_sources = []
        
        # Cancel futures if possible (though we just ignore results)
        self.cam_futures = {}
        
        self.video_display.configure(image=None, text="Stopped")
        self.btn_start.configure(state="normal", fg_color="green")
        self.btn_stop.configure(state="disabled", fg_color="gray")
        self.lbl_cam_status.configure(text="System Stopped", text_color="gray")

    def process_frame_logic(self, frame, debug_mode):
        """
        Run Detection -> Liveness -> Recognition
        Returns: annotated_frame, stats_dict
        """
        stats = {"real": 0, "fake": 0, "unknown": 0, "faces": 0}
        
        # 1. Detection
        faces = self.detector.detect(frame)
        stats["faces"] = len(faces)
        
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            face_center_id = f"{(x1+x2)//2}_{(y1+y2)//2}"
            
            # 2. Liveness
            label, raw_score, smoothed_score, details = self.liveness_detector.predict(frame, [x1, y1, x2, y2], face_id=face_center_id)
            
            color = (255, 165, 0) # Orange Unknown
            
            if label == "REAL":
                color = (0, 255, 0) # Green
                stats["real"] += 1
                
                # 3. Recognition (Only if Real)
                crop_img = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                
                name = "Unknown"
                if crop_img.size > 0:
                    name, confidence = self.face_recognizer.recognize(crop_img)
                
                if name != "Unknown":
                    # Log Attendance (Return name to main thread for safe logging)
                    stats["log_name"] = name
                    label = name # Show name
                else:
                    color = (0, 200, 200) # Cyan Unknown Real
                    # Save Unknown
                    self.face_recognizer.save_unknown_face(frame, [x1, y1, x2, y2])
                    
            elif label == "FAKE":
                color = (0, 0, 255) # Red
                stats["fake"] += 1
            else:
                stats["unknown"] += 1
                if details.get('quality_reason') != "OK":
                   color = (128, 128, 128)
            
            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            display_text = f"{label} ({smoothed_score*100:.0f}%)"
            cv2.putText(frame, display_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Debug
            if debug_mode:
                cv2.putText(frame, f"Raw: {raw_score:.2f}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                q_reason = details.get('quality_reason', "")
                if q_reason != "OK":
                    cv2.putText(frame, q_reason, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Liveness Mesh Viz
                landmarks = details.get('landmarks', [])
                if landmarks:
                    dim = details.get('square_dim', 0)
                    pad_left = details.get('pad_left', 0)
                    pad_top = details.get('pad_top', 0)
                    
                    # Re-calculate generic crop mapping
                    scale = 2.7
                    w_box = x2 - x1
                    h_box = y2 - y1
                    cx_box = x1 + w_box/2
                    cy_box = y1 + h_box/2
                    new_w = w_box * scale
                    new_h = h_box * scale
                    nx1 = int(cx_box - new_w/2)
                    ny1 = int(cy_box - new_h/2)
                    nx1 = max(0, nx1)
                    ny1 = max(0, ny1)
                    
                    if dim > 0:
                        for lx, ly in landmarks:
                             px_square = lx * dim
                             px_crop = px_square - pad_left
                             final_x = int(px_crop + nx1)
                             
                             py_square = ly * dim
                             py_crop = py_square - pad_top
                             final_y = int(py_crop + ny1)
                             
                             cv2.circle(frame, (final_x, final_y), 1, (0, 255, 0), -1)

        return frame, stats

    def update_video_feed(self):
        if not self.running:
            return

        for i, cam in enumerate(self.camera_sources):
            # If no future running for this cam, submit one
            if i not in self.cam_futures or self.cam_futures[i].done():
                
                # Check 1: Did we just finish one?
                if i in self.cam_futures and self.cam_futures[i].done():
                    try:
                        frame, stats = self.cam_futures[i].result()
                        
                        # Update UI Image ONLY if it's the active camera
                        if i == self.active_cam_index:
                            self.display_frame(frame)
                        
                        # Update Global Stats (Simple accumulation for display)
                        # Note: This updates continuously, might flicker if counters are reset. 
                        # Ideally, keep running total.
                        self.lbl_total_faces.configure(text=f"Faces: {stats['faces']}")
                        self.lbl_fake.configure(text=f"Fake: {stats['fake']}")
                        self.lbl_verified.configure(text=f"Real: {stats['real']}")
                        
                        # Log if name found
                        if "log_name" in stats:
                             parts = stats["log_name"].split('_')
                             if len(parts) >= 2:
                                 self.add_log_entry("_".join(parts[1:]), parts[0])
                             else:
                                 self.add_log_entry(stats["log_name"], "N/A")
                                 
                    except Exception as e:
                        print(f"Frame processing error: {e}")

                # Check 2: Submit new frame
                valid, frame = cam.get_frame()
                if valid:
                     # Submit to executor
                     self.cam_futures[i] = self.executor.submit(self.process_frame_logic, frame, self.debug_mode.get())
        
        self.after(10, self.update_video_feed)
        
    def display_frame(self, frame):
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        
        # Target Label
        target_lbl = self.video_display
        
        # Resize Logic
        display_w = target_lbl.winfo_width()
        display_h = target_lbl.winfo_height()
        
        if display_w > 10 and display_h > 10:
             img_ratio = img.width / img.height
             disp_ratio = display_w / display_h
             
             if disp_ratio > img_ratio:
                 new_w = display_w
                 new_h = int(new_w / img_ratio)
             else:
                 new_h = display_h
                 new_w = int(new_h * img_ratio)
             
             img = img.resize((new_w, new_h), Image.Resampling.NEAREST)
             
             # Center Crop
             left = (new_w - display_w) // 2
             top = (new_h - display_h) // 2
             img = img.crop((left, top, left + display_w, top + display_h))
             
             ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(display_w, display_h))
             target_lbl.configure(image=ctk_img, text="")
             target_lbl.image = ctk_img

if __name__ == "__main__":
    app = SmartAttendanceApp()
    app.mainloop()
