import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os
import time
import winsound  # Use winsound for Windows beep

from ui.widgets import create_camera_widget, create_control_panel
from camera.camera_manager import CameraManager
from camera.capture import save_image
from enrollment_utils.config import POSES, IMAGES_PER_POSE
from enrollment_utils.create_folders import create_student_folders

class AppLayout:
    def __init__(self, root):
        self.root = root
        
        # 1. Detect Cameras
        self.available_cameras = CameraManager.list_available_cameras()
        if not self.available_cameras:
            self.available_cameras = [0] # Fallback
            
        self.camera = CameraManager(source=self.available_cameras[0])
        
        # 2. Build UI
        self.video_label = create_camera_widget(root)
        self.instruction_label, self.start_btn = create_control_panel(
            root, 
            self.available_cameras, 
            self.start_enrollment_click,
            self.change_camera
        )
        
        # 3. State Management
        self.is_enrolling = False
        self.current_face_crop = None
        
        # Workflow State
        self.student_path = None
        self.current_pose_idx = 0
        self.current_img_count = 0
        self.enrollment_queue = [] # Pending actions logic if needed, but we'll use .after recursion

        # Start Video Loop
        self.update_video()

    def update_video(self):
        """Standard live video loop."""
        frame, face_crop = self.camera.get_frame()
        self.current_face_crop = face_crop
        
        if frame is not None:
            # Convert to image for Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        self.root.after(10, self.update_video)

    def change_camera(self, source_idx):
        """Callback for camera dropdown."""
        try:
            idx = int(source_idx)
            self.camera.change_source(idx)
        except ValueError:
            pass

    def start_enrollment_click(self, name, reg_no):
        """Handle Start Button Click."""
        if not name.strip() or not reg_no.strip():
            messagebox.showerror("Validation Error", "Please enter both Name and Registration Number.")
            return

        if self.is_enrolling:
            return

        # Create folders
        try:
            self.student_path = create_student_folders(reg_no, name)
        except Exception as e:
            messagebox.showerror("Error", f"Could not create folders: {e}")
            return

        # Initialize State
        self.is_enrolling = True
        self.start_btn.config(state="disabled")
        self.current_pose_idx = 0
        self.current_img_count = 0
        
        # Start the sequence
        self.process_enrollment_step()

    def process_enrollment_step(self):
        """Main State Machine for Enrollment."""
        if(self.current_pose_idx >= len(POSES)):
            self.finish_enrollment()
            return
            
        current_pose = POSES[self.current_pose_idx]
        
        # Start Countdown before the pose sequence starts
        self.start_countdown(5, current_pose)

    def start_countdown(self, seconds, pose):
        if not self.is_enrolling:
            return
            
        if seconds > 0:
            self.instruction_label.config(
                text=f"Get Ready for {pose.upper()}...\n{seconds}",
                fg="red"
            )
            self.root.after(1000, lambda: self.start_countdown(seconds - 1, pose))
        else:
             # Countdown finished, start capturing
             display_count = self.current_img_count + 1
             self.instruction_label.config(
                text=f"Look {pose.upper()}\nImage {display_count}/{IMAGES_PER_POSE}",
                fg="blue"
             )
             # Wait a small moment before first capture to stabilize
             self.root.after(500, lambda: self.capture_step(pose))
             
    def capture_step(self, pose):
        """Try to capture the current requirement."""
        if not self.is_enrolling:
            return

        # Check if face undetected
        if self.current_face_crop is None:
            self.instruction_label.config(text=f"Look {pose.upper()}\nNO FACE DETECTED!", fg="red")
            # Retry immediately (small delay)
            self.root.after(500, lambda: self.capture_step(pose))
            return
            
        # Capture and Save
        winsound.Beep(1000, 200) # Sound feedback
        
        pose_folder = os.path.join(self.student_path, pose)
        filename_idx = self.current_img_count + 1
        filename = f"{pose}_{filename_idx}.jpg"
        filepath = os.path.join(pose_folder, filename)
        
        success = cv2.imwrite(filepath, self.current_face_crop)
        
        if success:
            print(f"Saved: {filepath}")
            self.current_img_count += 1
            
            # Check if pose complete
            if self.current_img_count >= IMAGES_PER_POSE:
                self.current_pose_idx += 1
                self.current_img_count = 0
                # Next Pose (Loop back to process_enrollment_step which starts countdown)
                self.process_enrollment_step()
            else:
                # Same pose, next image.
                # Just wait 1 second between images of SAME pose, no full countdown
                display_count = self.current_img_count + 1
                self.instruction_label.config(
                    text=f"Look {pose.upper()}\nImage {display_count}/{IMAGES_PER_POSE}",
                    fg="blue"
                )
                self.root.after(1000, lambda: self.capture_step(pose))
        else:
            self.instruction_label.config(text="Error saving. Retrying...", fg="red")
            self.root.after(1000, lambda: self.capture_step(pose))

    def finish_enrollment(self):
        self.is_enrolling = False
        self.start_btn.config(state="normal")
        self.instruction_label.config(text="Enrollment Completed Successfully!", fg="green")
        messagebox.showinfo("Success", "Enrollment Completed Successfully!")
        # Reset instructions
        self.root.after(2000, lambda: self.instruction_label.config(text="Ready"))

    def on_close(self):
        self.is_enrolling = False
        self.camera.release()
        self.root.destroy()

def setup_ui(root):
    app = AppLayout(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
