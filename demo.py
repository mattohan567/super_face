from face_enhancer import FaceEnhancer
from evaluator import ImageEvaluator
import cv2
import numpy as np

def main():
    # Initialize components
    enhancer = FaceEnhancer()
    evaluator = ImageEvaluator()
    
    # Example usage
    image_path = "sample_image.jpg"  # Add your test image here
    
    try:
        # Process image
        results = enhancer.process_image(image_path)
        
        # Evaluate enhancement
        metrics = evaluator.evaluate_enhancement(results['original'], results['enhanced'])
        
        # Display results
        print(f"Faces detected: {results['faces_detected']}")
        print(f"PSNR: {metrics['psnr']:.2f}")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"LPIPS: {metrics['lpips']:.4f}")
        
        # Visualize
        enhancer.visualize_results(results)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please add a sample image to test the system")

if __name__ == "__main__":
    main()