import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class FaceEnhancerFinetuned:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing models on device: {self.device.upper()}")
        
        self.face_detector = YOLO('runs/detect/yolov8n_widerface_finetuned4/weights/best.pt')
        
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True if self.device == 'cuda' else False,
            device=self.device
        )
        print("✅ Face Enhancer Initialized with Real-ESRGAN.")

    def detect_faces(self, image_rgb):
        results = self.face_detector(image_rgb)
        faces = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:
                        h, w, _ = image_rgb.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        faces.append((x1, y1, x2, y2, conf))
        return faces

    def enhance_face(self, image_crop_bgr):
        if image_crop_bgr is None or image_crop_bgr.size == 0:
            return None
        try:
            enhanced_face, _ = self.upsampler.enhance(image_crop_bgr)
            return enhanced_face
        except Exception as e:
            print(f"Error during Real-ESRGAN enhancement: {e}")
            h, w, _ = image_crop_bgr.shape
            return cv2.resize(image_crop_bgr, (w*2, h*2), interpolation=cv2.INTER_LANCZOS4)

    def process_image(self, image_path):
        """
        Processes an image using the robust color-space-aware logic.
        """
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image from path: {image_path}")

        # Convert to RGB once for detection and cropping
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        face_boxes = self.detect_faces(image_rgb)

        original_crops = []
        enhanced_crops = []

        for (x1, y1, x2, y2, conf) in face_boxes:
            # Crop from the same RGB image we detected on
            face_crop_rgb = image_rgb[y1:y2, x1:x2]
            if face_crop_rgb.size == 0:
                continue

            # Store the original crop in BGR for consistent data types
            original_crops.append(cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR))

            # Convert the RGB crop to BGR for the Real-ESRGAN model
            face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
            enhanced_face = self.enhance_face(face_crop_bgr) # This will be BGR

            if enhanced_face is not None:
                enhanced_crops.append(enhanced_face)
            else:
                enhanced_crops.append(np.zeros_like(face_crop_bgr))

        return {
            'original': image_rgb, # Return the main image in RGB for direct display
            'faces_detected': len(face_boxes),
            'face_boxes': face_boxes,
            'original_crops': original_crops, # These are BGR
            'enhanced_crops': enhanced_crops  # These are also BGR
        }

    def visualize_results(self, results):
        """
        Displays the results, converting BGR crops to RGB for display.
        This function is now compatible with the corrected process_image output.
        """
        if not results.get('faces_detected'):
            return

        num_faces = len(results.get('original_crops', []))
        if num_faces > 0:
            fig, axes = plt.subplots(num_faces, 2, figsize=(10, 5 * num_faces))
            if num_faces == 1:
                axes = [axes]
            
            fig.suptitle("Original Crop vs. Real-ESRGAN Super-Resolution", fontsize=16)

            for i in range(num_faces):
                orig_crop_bgr = results['original_crops'][i]
                enh_crop_bgr = results['enhanced_crops'][i]
                
                # Convert BGR crops to RGB *just for displaying* with Matplotlib
                orig_crop_rgb = cv2.cvtColor(orig_crop_bgr, cv2.COLOR_BGR2RGB)
                enh_crop_rgb = cv2.cvtColor(enh_crop_bgr, cv2.COLOR_BGR2RGB)

                axes[i][0].imshow(orig_crop_rgb)
                axes[i][0].set_title(f'Original Face {i+1}\n({orig_crop_rgb.shape[1]}x{orig_crop_rgb.shape[0]})')
                axes[i][0].axis('off')
                
                axes[i][1].imshow(enh_crop_rgb)
                axes[i][1].set_title(f'Enhanced Face {i+1}\n({enh_crop_rgb.shape[1]}x{enh_crop_rgb.shape[0]})')
                axes[i][1].axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()