"""
Face detection using Haar Cascades (OpenCV default)
"""
import cv2
import os

class FaceDetector:
    def __init__(self, min_neighbors=5, scale_factor=1.1):
        # Load classifiers
        self.frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        
        self.face_cascade = cv2.CascadeClassifier(self.frontal_path)
        self.profile_cascade = cv2.CascadeClassifier(self.profile_path)
        
        self.min_neighbors = min_neighbors
        self.scale_factor = scale_factor
        
        if self.face_cascade.empty():
            print(f"Error: Could not load Frontal Cascade from {self.frontal_path}")
        if self.profile_cascade.empty():
            print(f"Error: Could not load Profile Cascade from {self.profile_path}")

    def detect(self, frame):
        """
        Detects faces in the frame.
        Returns a list of bounding boxes (x, y, w, h).
        """
        if frame is None:
            return []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Try Frontal Face
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            return [tuple(f) for f in faces]
            
        # 2. Try Profile Face (if no frontal found)
        # Profile cascade only detects left profiles? Flip for right?
        # Haar profile cascade usually detects faces looking to the right (from observer POV) or left.
        # We can try detecting on normal and flipped image to catch both sides.
        
        # Detect on normal
        profiles = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        if len(profiles) > 0:
             return [tuple(f) for f in profiles]
             
        # Detect on flipped (for the other profile)
        flipped_gray = cv2.flip(gray, 1)
        profiles_flipped = self.profile_cascade.detectMultiScale(
            flipped_gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        
        if len(profiles_flipped) > 0:
            # We need to map coordinates back to original frame
            h, w = gray.shape
            corrected_profiles = []
            for (x, y, w_box, h_box) in profiles_flipped:
                # x in flipped is w - (x + w_box) in original
                original_x = w - (x + w_box)
                corrected_profiles.append((original_x, y, w_box, h_box))
            return corrected_profiles
            
        return []
