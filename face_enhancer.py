import cv2
import numpy as np
import torch
from ultralytics import YOLO
from gfpgan import GFPGANer
import matplotlib.pyplot as plt
from pathlib import Path

class FaceEnhancer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.face_detector = YOLO('yolov8n.pt')
        self.gfpgan = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
    
    def detect_faces(self, image):
        results = self.face_detector(image)
        faces = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        # Ensure bounding box is within image dimensions
                        h, w, _ = image.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        faces.append((x1, y1, x2, y2, conf))
        return faces
    
    def process_image(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image from path: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Detect faces using our chosen detector (YOLOv8)
        faces = self.detect_faces(image_rgb)
        enhanced_image = image_rgb.copy()
        
        # Store intermediate results for visualization
        original_crops = []
        enhanced_crops = []

        # 2. Loop through detected faces, enhance each one, and paste it back
        for (x1, y1, x2, y2, conf) in faces:
            # Crop the face from the image
            face_crop = image_rgb[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue
            
            original_crops.append(face_crop)

            # 3. Enhance the face crop using GFPGAN
            _, restored_faces, _ = self.gfpgan.enhance(
                face_crop,
                has_aligned=False,
                only_center_face=True,
                paste_back=False
            )
            
            # 4. Paste the enhanced face back into the image
            if restored_faces and len(restored_faces) > 0:
                restored_face = restored_faces[0]
                enhanced_crops.append(restored_face)
                
                # Resize the enhanced face to fit the original bounding box
                original_h, original_w, _ = face_crop.shape
                restored_face_resized = cv2.resize(restored_face, (original_w, original_h))
                
                enhanced_image[y1:y2, x1:x2] = restored_face_resized
            else:
                # If enhancement fails for a crop, add a placeholder
                enhanced_crops.append(np.zeros_like(face_crop))
        
        return {
            'original': image_rgb,
            'enhanced': enhanced_image,
            'faces_detected': len(faces),
            'face_boxes': faces,
            'original_crops': original_crops,
            'enhanced_crops': enhanced_crops
        }
    
    def enhance_face(self, image):
        _, _, restored_img = self.gfpgan.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
        return restored_img
    
    
    def visualize_results(self, results):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(results['original'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(results['enhanced'])
        axes[1].set_title(f'Enhanced Image ({results["faces_detected"]} faces)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()