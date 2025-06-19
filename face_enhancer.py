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
                        faces.append((x1, y1, x2, y2, conf))
        return faces
    
    def enhance_face(self, image):
        _, _, restored_img = self.gfpgan.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
        return restored_img
    
    def process_image(self, image_path):
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        faces = self.detect_faces(image_rgb)
        enhanced_image = self.enhance_face(image_rgb)
        
        return {
            'original': image_rgb,
            'enhanced': enhanced_image,
            'faces_detected': len(faces),
            'face_boxes': faces
        }
    
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