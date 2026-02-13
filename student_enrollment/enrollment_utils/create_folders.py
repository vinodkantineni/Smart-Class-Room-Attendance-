"""
Create dataset folders per pose
"""
import os
from enrollment_utils.config import DATASET_PATH, POSES

def create_student_folders(student_id, student_name):
    # Construct folder name: e.g., dataset/22CSE123_Manoj
    folder_name = f"{student_id}_{student_name}"
    student_path = os.path.join(DATASET_PATH, folder_name)

    # Create the main student folder
    if not os.path.exists(student_path):
        os.makedirs(student_path)

    # Create subfolders for each pose
    for pose in POSES:
        pose_path = os.path.join(student_path, pose)
        if not os.path.exists(pose_path):
            os.makedirs(pose_path)
            
    return student_path
