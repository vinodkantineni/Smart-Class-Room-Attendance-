import cv2
from enrollment_utils.detector import FaceDetector

class CameraManager:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.detector = FaceDetector()
        
    @staticmethod
    def list_available_cameras(max_check=2):
        """
        Check for available cameras by trying to open indices 0 to max_check.
        Result is a list of available indices.
        """
        available_cameras = []
        for i in range(max_check):
            stream = cv2.VideoCapture(i)
            if stream.isOpened():
                available_cameras.append(i)
                stream.release()
        return available_cameras

    def change_source(self, source):
        if self.source == source:
            return
        if self.cap.isOpened():
            self.cap.release()
        self.source = source
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        if not self.cap.isOpened():
            return None, None
            
        ret, frame = self.cap.read()
        if not ret:
            return None, None
            
        frame = cv2.flip(frame, 1) # Mirror effect
        faces = self.detector.detect(frame)
        
        # Draw rectangles around faces
        annotated_frame = frame.copy()
        detected_face_crop = None

        if faces:
            # Assume the largest face is the target
            # Sort by area (w*h) descending
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Crop the face with some margin if possible, for now just strict bbox
            # Ensure ROI is valid
            if y < frame.shape[0] and x < frame.shape[1]:
                detected_face_crop = frame[y:y+h, x:x+w]

        return annotated_frame, detected_face_crop

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
