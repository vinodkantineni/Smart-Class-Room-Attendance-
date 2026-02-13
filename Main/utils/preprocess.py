import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_input_stats(stage_name, array):
    """
    Logs shape, dtype, min, max, and mean values of an input array.
    """
    if array is None:
        logging.warning(f"[{stage_name}] Input is None")
        return

    try:
        min_val = np.min(array)
        max_val = np.max(array)
        mean_val = np.mean(array)
        logging.info(f"[{stage_name}] Shape: {array.shape} | Dtype: {array.dtype} | Min: {min_val:.3f} | Max: {max_val:.3f} | Mean: {mean_val:.3f}")
    except Exception as e:
        logging.error(f"[{stage_name}] Error logging stats: {e}")

def pad_to_square(image, fill_value=0):
    """
    Pads an image to be square.
    Returns: padded_image, (pad_left, pad_top, scale)
    Scale is always 1.0 here as we are only padding, not resizing yet, 
    but we return it for consistent API if we were to incorporate resizing.
    """
    h, w = image.shape[:2]
    if h == w:
        return image, (0, 0, 1.0)
    
    dim = max(h, w)
    pad_img = np.full((dim, dim, 3), fill_value, dtype=image.dtype)
    
    # Center the image? Or Top-Left?
    # For face detection/recognition, centering is usually safer strictly for the square requirement,
    # but top-left is easier to coordinate mapping. 
    # Let's do Center Padding to preserve context balance.
    
    pad_top = (dim - h) // 2
    pad_left = (dim - w) // 2
    
    pad_img[pad_top:pad_top+h, pad_left:pad_left+w] = image
    
    return pad_img, (pad_left, pad_top, 1.0)

def check_image_quality(image, blur_threshold=100.0, dark_threshold=30.0):
    """
    Checks image for blur and darkness.
    Returns: (is_good, reason)
    """
    if image is None or image.size == 0:
        return False, "Empty Image"
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Darkness Check
    brightness = np.mean(gray)
    if brightness < dark_threshold:
        return False, f"Too Dark ({brightness:.1f})"
        
    # Blur Check (Laplacian Variance)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < blur_threshold:
        return False, f"Too Blurry ({variance:.1f})"
        
    return True, "OK"
