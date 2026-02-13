import cv2
import onnxruntime as ort
import numpy as np
import os
import time

# Standardized Utils
from Main.utils.preprocess import pad_to_square, log_input_stats

class SCRFD:
    def __init__(self, model_path, input_size=(640, 640), conf_thres=0.5, nms_thres=0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        
        # Initialize ONNX Runtime
        try:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.session = None
            return

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # SCRFD parameters
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

    def detect(self, img):
        if self.session is None: return []

        height, width = img.shape[:2]
        
        # 1. Standardized Preprocessing
        # Pad to square first to preserve aspect ratio strictly
        padded_img, (pad_left, pad_top, _) = pad_to_square(img)
        pad_h, pad_w = padded_img.shape[:2]
        
        # Resize to input_size (640x640)
        blob_img = cv2.resize(padded_img, self.input_size)
        
        # Standardize blob
        blob = cv2.dnn.blobFromImage(blob_img, 1.0/128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True)
        
        # Log Stats (Once/debug or sample)
        # log_input_stats("SCRFD_Input", blob)
        
        # Interference
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        
        # Decode
        scores_list = []
        bboxes_list = []
        
        # Scale factor from Original Padded -> 640
        # resize_scale = 640 / padded_dimension
        resize_scale = self.input_size[0] / pad_w 
        
        fmaps = {8: {}, 16: {}, 32: {}}
        
        for val in outputs:
            if val.ndim == 3: val = val[0]
            
            rows, cols = val.shape
            stride = 0
            if rows == 12800: stride = 8
            elif rows == 3200: stride = 16
            elif rows == 800: stride = 32
            else: continue
            
            if cols == 1: fmaps[stride]['score'] = val
            elif cols == 4: fmaps[stride]['bbox'] = val
            
        for stride in self._feat_stride_fpn:
            if 'score' not in fmaps[stride] or 'bbox' not in fmaps[stride]: continue
                
            scores = fmaps[stride]['score']
            bbox_preds = fmaps[stride]['bbox']
            
            # Generate anchors
            feat_h = self.input_size[1] // stride
            feat_w = self.input_size[0] // stride
            
            shift_y, shift_x = np.mgrid[:feat_h, :feat_w]
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            
            if self._num_anchors > 1:
                shift_x = np.stack([shift_x]*self._num_anchors, axis=1).reshape(-1)
                shift_y = np.stack([shift_y]*self._num_anchors, axis=1).reshape(-1)

            anchor_centers = np.stack([shift_x, shift_y], axis=-1) * stride
            
            valid_idxs = np.where(scores > self.conf_thres)[0]
            
            if len(valid_idxs) > 0:
                valid_scores = scores[valid_idxs]
                valid_bbox_preds = bbox_preds[valid_idxs] * stride
                valid_centers = anchor_centers[valid_idxs]
                
                x1 = valid_centers[:, 0] - valid_bbox_preds[:, 0]
                y1 = valid_centers[:, 1] - valid_bbox_preds[:, 1]
                x2 = valid_centers[:, 0] + valid_bbox_preds[:, 2]
                y2 = valid_centers[:, 1] + valid_bbox_preds[:, 3]
                
                bboxes = np.stack([x1, y1, x2, y2], axis=-1)
                
                scores_list.append(valid_scores)
                bboxes_list.append(bboxes)
        
        if not scores_list: return []
            
        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0)
        
        # Rescale bboxes: 640 -> Padded Full
        bboxes /= resize_scale
        
        # Remove Padding offset
        bboxes[:, 0] -= pad_left
        bboxes[:, 2] -= pad_left
        bboxes[:, 1] -= pad_top
        bboxes[:, 3] -= pad_top
        
        # NMS
        keep = self.nms(bboxes, scores, self.nms_thres)
        
        final_faces = []
        for i in keep:
            bbox = bboxes[i]
            if bbox.ndim > 1: bbox = bbox.flatten()
            
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            final_faces.append({
                'bbox':  np.array([x1, y1, x2, y2], dtype=int),
                'score': float(scores[i].item()) if hasattr(scores[i], 'item') else float(scores[i])
            })
            
        return final_faces

    def nms(self, boxes, scores, thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        if scores.ndim > 1: scores = scores.flatten()
        
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

def main():
    print("Initializing SCRFD Face Detection...")
    model_path = os.path.join("Main", "models", "scrfd_2.5g_bnkps.onnx")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    detector = SCRFD(model_path)
    cap_webcam = cv2.VideoCapture(0)
    webcam_open = cap_webcam.isOpened()
    
    print("Press 'q' to quit.")
    while True:
        if webcam_open:
            ret, frame = cap_webcam.read()
            if not ret: break
            
            faces = detector.detect(frame)
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.imshow("Detection", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    if cap_webcam.isOpened(): cap_webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
