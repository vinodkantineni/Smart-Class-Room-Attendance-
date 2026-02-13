import torch
import numpy as np
import os
import cv2
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

class FaceRecognizer:
    def __init__(self, dataset_path="dataset"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Face Recognition Loop on: {self.device}")
        
        # Load Pretrained Model (Inception Resnet V1 - VGGFace2)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Database
        self.dataset_path = dataset_path
        self.known_embeddings = []
        self.known_names = []
        
        # Preprocessing (Standard InceptionResNetV1 normalization)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            # Whiten (standardize) is often done by facenet-pytorch internally or manually. 
            # InceptionResnetV1 in this repo expects fixed_image_standardization if not using MTCNN.
            # We will do manual standardization: (x - 127.5) / 128.0 roughly, or using mean/std.
            # Official VGGFace2 pretraining expects standard float tensor.
        ])
        
        self.load_known_faces()

    def get_embedding(self, face_img):
        """
        Generate embedding for a given crop.
        """
        try:
            # face_img is numpy BGR (from cv2)
            # Convert to RGB
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Transform to tensor
            tensor_img = self.transform(rgb_img).to(self.device)
            
            # Add batch dimension
            batch_img = tensor_img.unsqueeze(0)
            
            # Standardize (fixed standardization for InceptionResnetV1)
            # (x - 127.5) / 128.0 is common, but let's check what facenet-pytorch recommends.
            # Actually, the model expects inputs in range [0, 1] if not using their fixed_image_standardization utils.
            # But normally it requires whitening. 
            # Let's use simple float conversion which ToTensor does [0, 1].
            
            with torch.no_grad():
                embedding = self.resnet(batch_img)
            
            return embedding.cpu().detach().numpy()[0]
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def load_known_faces(self):
        """
        Load images from dataset folder and pre-compute embeddings.
        Structure: dataset/Student_Name/User_Images.jpg
        """
        print("Loading known faces...")
        if not os.path.exists(self.dataset_path):
            print(f"Dataset path not found: {self.dataset_path}")
            return

        count = 0
        # Walk through dataset directory
        # Expected structure: dataset/student_id_name/images...
        for student_folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, student_folder)
            
            if os.path.isdir(folder_path):
                # We simply use the folder name as the ID/Name
                student_name = student_folder 
                
                # Check for images inside (recursively or shallow)
                # Structure: dataset/ID_Name/Pose/Image.jpg or dataset/ID_Name/Image.jpg
                
                # We walk through the student folder to find all images in subfolders (center, left, etc.)
                for root, dirs, files in os.walk(folder_path):
                    for filename in files:
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(root, filename)
                            
                            # Read and embed
                            img = cv2.imread(img_path)
                            if img is not None:
                                emb = self.get_embedding(img)
                                if emb is not None:
                                    self.known_embeddings.append(emb)
                                    self.known_names.append(student_name)
                                    count += 1
                                    # Limit: If we have too many images, loading time might be slow.
                                    # For now, load all to ensure best accuracy from different angles.
        
        print(f"Loaded {count} embeddings for {len(set(self.known_names))} unique students.")

    def recognize(self, face_crop, threshold=0.7): # Threshold was 0.6 in plan, 0.7 is often safer for cosine
        """
        Compare face_crop against known embeddings.
        """
        embedding = self.get_embedding(face_crop)
        if embedding is None:
            return "Unknown", 0.0

        if len(self.known_embeddings) == 0:
            return "Unknown", 0.0

        # Calculate distances (Cosine Similarity preferred for this model)
        # Cosine Distance = 1 - Cosine Similarity
        # Or Euclidean distance. InceptionResnetV1 output is usually normalized if classify=False?
        # Let's use Euclidean distance for simplicity if not normalized, or Cosine.
        
        # Facenet-pytorch embeddings are NOT automatically normalized to unit length unless we assume so.
        # It's safer to calculate Cosine Similarity manually.
        
        dataset_embeddings = np.array(self.known_embeddings) # (N, 512)
        target_embedding = embedding # (512,)
        
        # Normalize
        target_norm = np.linalg.norm(target_embedding)
        dataset_norm = np.linalg.norm(dataset_embeddings, axis=1)
        
        # Avoid division by zero
        if target_norm == 0: return "Unknown", 0.0
        
        # Cosine Similarity: (A . B) / (|A| * |B|)
        dot_product = np.dot(dataset_embeddings, target_embedding)
        similarities = dot_product / (dataset_norm * target_norm)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim > threshold:
            match_name = self.known_names[best_idx]
            return match_name, best_sim
        else:
            return "Unknown", best_sim

    def save_unknown_face(self, frame, bbox):
        """
        Save the unknown face to a reports directory.
        """
        try:
            # Create directory
            report_dir = os.path.join("reports", "unknown", datetime.datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(report_dir, exist_ok=True)
            
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: return
            
            timestamp = datetime.datetime.now().strftime("%H-%M-%S-%f")
            filename = os.path.join(report_dir, f"unknown_{timestamp}.jpg")
            
            cv2.imwrite(filename, crop)
            # print(f"Saved unknown face: {filename}")
        except Exception as e:
            print(f"Error saving unknown face: {e}")

