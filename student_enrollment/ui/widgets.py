import tkinter as tk
from tkinter import ttk

def create_camera_widget(parent):
    """Creates the label to display the camera feed."""
    frame = ttk.LabelFrame(parent, text="Camera Preview")
    frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    video_label = tk.Label(frame, bg="black")
    video_label.pack(fill=tk.BOTH, expand=True)
    return video_label

def create_control_panel(parent, available_cameras, on_start_callback, on_camera_change_callback):
    """Creates the input form and control buttons."""
    frame = ttk.LabelFrame(parent, text="Enrollment Controls")
    frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y, ipadx=10)
    
    # --- Inputs ---
    ttk.Label(frame, text="Student Name:").pack(pady=(10, 5), anchor="w")
    name_entry = ttk.Entry(frame)
    name_entry.pack(pady=5, fill=tk.X)
    
    ttk.Label(frame, text="Registration Number:").pack(pady=5, anchor="w")
    reg_entry = ttk.Entry(frame)
    reg_entry.pack(pady=5, fill=tk.X)
    
    # --- Camera Selection ---
    ttk.Label(frame, text="Select Camera:").pack(pady=(15, 5), anchor="w")
    cam_var = tk.StringVar(value=available_cameras[0] if available_cameras else "0")
    cam_combo = ttk.Combobox(frame, textvariable=cam_var, values=available_cameras, state="readonly")
    cam_combo.pack(pady=5, fill=tk.X)
    cam_combo.bind("<<ComboboxSelected>>", lambda e: on_camera_change_callback(cam_var.get()))
    
    ttk.Separator(frame, orient=tk.HORIZONTAL).pack(pady=20, fill=tk.X)
    
    # --- Action ---
    start_btn = ttk.Button(frame, text="START ENROLLMENT", 
                           command=lambda: on_start_callback(name_entry.get(), reg_entry.get()))
    start_btn.pack(pady=10, fill=tk.X, ipady=5)
    
    # --- Instructions / Status ---
    ttk.Label(frame, text="Instructions:").pack(pady=(20, 5), anchor="w")
    instruction_label = tk.Label(frame, text="Ready", font=("Arial", 14, "bold"), fg="blue", wraplength=200, justify="center")
    instruction_label.pack(pady=10, fill=tk.X)
    
    return instruction_label, start_btn
