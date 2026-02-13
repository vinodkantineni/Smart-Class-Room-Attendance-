import cv2
import numpy as np
import os
import mediapipe as mp
import torch
import torch.nn.functional as F
from collections import deque
import logging

# Standardized Utils
from Main.utils.preprocess import pad_to_square, log_input_stats, check_image_quality

# Try to import MiniFASNet loader
try:
    from Main.minifasnet import load_model
    LOGGING_PREFIX = "Main.liveness"
except ImportError:
    print("Warning: Could not import 'Main.minifasnet'. Liveness model will be disabled.")
    load_model = None

class LivenessDetector:
    def __init__(self, model_path, history_length=5):
        base_dir = os.path.dirname(model_path)
        self.model_path = os.path.join(base_dir, "liveness.pth")
        
        self.model = None
        self.device = torch.device('cpu') # Use CPU for safety
        self.history_length = history_length
        self.history = {} # Key: track_id, Value: deque
        self.consecutive_real = {} # Key: track_id, Value: int (count of consecutive real frames)
        self.required_consecutive_frames = 3        
        # 1. Load PyTorch Model (Anti-Spoofing)
        if os.path.exists(self.model_path) and load_model:
            try:
                self.model, _ = load_model(self.model_path)
                if self.model:
                    self.model.to(self.device)
                    self.model.eval()
                    print(f"Success: Liveness Model Loaded: {self.model_path}")
                else:
                    print(f"Warning: Failed to load liveness model structure.")
            except Exception as e:
                print(f"Warning: Error loading liveness model: {e}")
        else:
            print(f"Warning: Liveness model not found at: {self.model_path}")
            print("   Info: Running in Fallback Mode: Blink Detection Only.")

        # 2. Initialize MediaPipe Face Mesh (Active Liveness)
        self.use_mediapipe = False
        try:
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True, 
                    max_num_faces=1,
                    refine_landmarks=True, 
                    min_detection_confidence=0.5
                )
                self.use_mediapipe = True
            else:
                 print("Warning: mp.solutions.face_mesh not found. Active liveness disabled.")
                 self.face_mesh = None
        except Exception as e:
            print(f"Warning: MediaPipe init failed: {e}. Active liveness disabled.")
            self.face_mesh = None
        
        self.blink_threshold = 0.30 # Lower for refined landmarks often

    def get_history_score(self, face_id, score):
        if face_id not in self.history:
            self.history[face_id] = deque(maxlen=self.history_length)
        self.history[face_id].append(score)
        return np.median(self.history[face_id])

    def predict(self, frame, bbox, face_id=None):
        """
        Predict liveness with hybrid logic (Model + Blink) and consecutive validation.
        Returns: label, score, smoothed_score, details_dict
        """
        details = {}
        x1, y1, x2, y2 = bbox
        
        if face_id is None:
            # Simple hash of center coord if no ID provided
            face_id = f"{(x1+x2)//2}_{(y1+y2)//2}"

        # Initialize history tracks if new
        if face_id not in self.consecutive_real:
            self.consecutive_real[face_id] = 0

        # --- Step 1: Geometry & Preprocessing ---
        scale = 2.7 
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        new_w = w * scale
        new_h = h * scale
        nx1 = int(cx - new_w/2)
        ny1 = int(cy - new_h/2)
        nx2 = int(cx + new_w/2)
        ny2 = int(cy + new_h/2)
        
        h_img, w_img = frame.shape[:2]
        nx1, ny1 = max(0, nx1), max(0, ny1)
        nx2, ny2 = min(w_img, nx2), min(h_img, ny2)
        
        crop = frame[ny1:ny2, nx1:nx2]
        
        # Quality Check (Stricter)
        is_good, quality_reason = check_image_quality(crop)
        details['quality_reason'] = quality_reason
        if not is_good:
            # Reset consecutive count on bad quality to be safe
            self.consecutive_real[face_id] = 0 
            return "UNCERTAIN", 0.0, 0.0, details

        # --- Step 2: MiniFASNetV2 Inference (Passive) ---
        model_score = 0.0
        
        if self.model:
            try:
                # Resize to 80x80 for Model
                resized_crop = cv2.resize(crop, (80, 80))
                rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                
                blob = rgb_crop.astype(np.float32)
                blob = np.transpose(blob, (2, 0, 1)) 
                blob = np.expand_dims(blob, axis=0) 
                
                tensor_input = torch.from_numpy(blob).to(self.device)
                
                with torch.no_grad():
                    logits = self.model(tensor_input)
                    prob = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                # prob shape: [spoof, real, other]
                if len(prob) >= 2:
                    model_score = float(prob[1]) # REAL class probability
            except Exception as e:
                print(f"Inference Error: {e}")
                model_score = 0.0
        
        details['raw_score'] = model_score
        
        # --- Step 3: MediaPipe Active Liveness (Blink Detection) ---
        square_crop, (pad_left, pad_top, _) = pad_to_square(crop)
        square_rgb = cv2.cvtColor(square_crop, cv2.COLOR_BGR2RGB)
        
        results = None
        if self.use_mediapipe and self.face_mesh:
            try:
                results = self.face_mesh.process(square_rgb)
            except Exception as e:
                print(f"MP Process Error: {e}")
        
        blink_ratio = 0.0
        user_is_blinking = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Log landmarks for debugging if needed
                details['landmarks'] = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                
                # EAR Calculation
                left_eye = [33, 160, 158, 133, 153, 144]
                right_eye = [362, 385, 387, 263, 373, 380]
                
                def get_ear(eye_indices, landmarks):
                    pts = []
                    for idx in eye_indices:
                        pt = landmarks.landmark[idx]
                        pts.append(np.array([pt.x, pt.y]))
                    
                    v1 = np.linalg.norm(pts[1] - pts[5])
                    v2 = np.linalg.norm(pts[2] - pts[4])
                    h_dist = np.linalg.norm(pts[0] - pts[3])
                    return (v1 + v2) / (2.0 * h_dist + 1e-6)
                
                ear_left = get_ear(left_eye, face_landmarks)
                ear_right = get_ear(right_eye, face_landmarks)
                blink_ratio = (ear_left + ear_right) / 2.0
                break
        
        details['blink_ratio'] = blink_ratio
        
        # Add dimensions to details for UI visualization
        # pad_to_square returns (pad_left, pad_top, pad_bottom)
        details['square_dim'] = square_crop.shape[0] # Height (and width since it's square)
        details['pad_left'] = pad_left
        details['pad_top'] = pad_top
        
        # Check Blink Threshold
        if blink_ratio < self.blink_threshold and blink_ratio > 0.05: # >0.05 to avoid closed-eye/tracking errors
            user_is_blinking = True

        # --- Step 4: Hybrid Decision Logic ---
        smoothed_score = self.get_history_score(face_id, model_score)
        details['smoothed_score'] = smoothed_score
        
        final_label = "UNKNOWN"
        is_candidate_real = False

        if self.model:
            STRONG_REAL_THRESH = 0.70
            WEAK_REAL_THRESH = 0.40 # "Gray Zone" start
            
            if smoothed_score > STRONG_REAL_THRESH:
                # High confidence -> Real
                is_candidate_real = True
                details['reason'] = "High Confidence"
            elif smoothed_score > WEAK_REAL_THRESH:
                # 'Gray Zone' -> Require Blink to confirm
                if user_is_blinking:
                    is_candidate_real = True
                    details['reason'] = "Medium Confidence + Blink Verified"
                else:
                    is_candidate_real = False
                    details['reason'] = "Medium Confidence - Waiting for Blink"
            else:
                # Low confidence -> Fake
                is_candidate_real = False
                details['reason'] = "Low Confidence (Spoof)"
        else:
            # Fallback (No Model)
            if user_is_blinking:
                is_candidate_real = True
                smoothed_score = 1.0
            else:
                is_candidate_real = False
                smoothed_score = 0.0

        # --- Step 5: Consecutive Frame Validation ---
        if is_candidate_real:
            self.consecutive_real[face_id] += 1
        else:
            # Reset if flow breaks (strict)
            self.consecutive_real[face_id] = 0
            
        consecutive_count = self.consecutive_real[face_id]
        details['consecutive_frames'] = consecutive_count

        if consecutive_count >= self.required_consecutive_frames:
            final_label = "REAL"
        elif consecutive_count > 0:
            final_label = "UNCERTAIN" # Transitioning...
        else:
            final_label = "FAKE"

        return final_label, model_score, smoothed_score, details


