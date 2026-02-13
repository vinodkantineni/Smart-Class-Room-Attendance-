import cv2
import os
import time

def save_image(image, folder_path, prefix="img"):
    if image is None or image.size == 0:
        print("Error: Empty image, cannot save.")
        return False
        
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(folder_path, filename)
    
    try:
        cv2.imwrite(filepath, image)
        print(f"Saved: {filepath}")
        return True
    except Exception as e:
        print(f"Failed to save image: {e}")
        return False
