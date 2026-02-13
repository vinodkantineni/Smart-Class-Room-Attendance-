import cv2
import threading
import time

class CameraSource:
    def __init__(self, source_id, name="Camera"):
        self.source_id = source_id
        self.name = name
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return

        try:
            self.cap = cv2.VideoCapture(self.source_id)
            if not self.cap.isOpened():
                print(f"Warning: {self.name} (Source {self.source_id}) failed to open.")
                return False
            
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print(f"Started {self.name} on source {self.source_id}")
            return True
        except Exception as e:
            print(f"Error starting {self.name}: {e}")
            return False

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        print(f"Stopped {self.name}")

    def _update_loop(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
                else:
                    self.frame = None
            
            # Small sleep to prevent CPU hogging if capture is fast, 
            # though usually read() blocks to match fps.
            time.sleep(0.005)

    def get_frame(self):
        with self.lock:
            if self.ret and self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def is_active(self):
        return self.running and self.cap is not None and self.cap.isOpened()
