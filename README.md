# Face Super-Resolution with Fine-tuned YOLO and Real-ESRGAN

A comprehensive face super-resolution system combining fine-tuned YOLOv8 face detection with Real-ESRGAN 4x enhancement. This project demonstrates advanced model fine-tuning, problem-solving through model migration, and robust pipeline development.

## Key Features

- **Fine-tuned YOLOv8**: Custom-trained on WiderFace dataset (12,880 images) for improved face detection
- **Real-ESRGAN 4x Enhancement**: High-quality face super-resolution without artifacts
- **Robust Pipeline**: Complete color space management and error handling
- **Problem-Solving**: Successfully migrated from GFPGAN to Real-ESRGAN to eliminate gray padding issues
- **Comprehensive Evaluation**: PSNR, SSIM, and LPIPS metrics for quality assessment

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Explore the Complete Implementation
```bash
jupyter notebook Face_Super_Resolution_Project.ipynb
```

This notebook contains the full pipeline, training process, and analysis.

## Core Implementation Files

### Main Implementation
- **`Face_Super_Resolution_Project.ipynb`** - Complete project with training, analysis, and results
- **`face_enhancer_yoloft.py`** - Main implementation class with fine-tuned YOLO + Real-ESRGAN
- **`evaluator.py`** - Image quality evaluation metrics (PSNR, SSIM, LPIPS)

### Training Artifacts
- `runs/detect/yolov8n_widerface_finetuned4/weights/best.pt` - Fine-tuned YOLO weights
- `wider_face.yaml` - YOLO training configuration
- `yolo_training/` - WiderFace dataset structure (ignored)

### Legacy Files
- `face_enhancer.py` - Original GFPGAN implementation (superseded)
- `demo.py` - Simple demo using old approach (not current implementation)

## 🛠️ Technical Implementation

### Model Architecture
1. **YOLOv8n Fine-tuned**: Specialized face detection
   - Trained on WiderFace dataset (50 epochs)
   - Improved accuracy on diverse face conditions
   - Confidence threshold: 0.5

2. **Real-ESRGAN x4plus**: Super-resolution enhancement
   - RRDBNet architecture (64 features, 23 blocks)
   - 4x upscaling factor
   - No padding artifacts (unlike GFPGAN)

### Pipeline Process
1. **Face Detection**: Fine-tuned YOLO identifies faces with high accuracy
2. **Face Extraction**: Precise cropping with boundary validation
3. **Color Space Management**: Robust RGB/BGR conversion
4. **4x Super-Resolution**: Real-ESRGAN enhancement
5. **Result Visualization**: Side-by-side comparison

## 🔧 Key Improvements Made

### Problem Identification & Solution
- **Issue**: GFPGAN produced gray padding artifacts around enhanced faces
- **Solution**: Migrated to Real-ESRGAN for cleaner results
- **Outcome**: Achieved 4x scaling (better than original 2x goal) without artifacts

### Model Fine-tuning Success
- **Dataset**: WiderFace (10,304 train + 2,576 validation images)
- **Training**: 50 epochs, batch size 8, 640px resolution
- **Result**: Significantly improved face detection across diverse conditions

## Performance Metrics

- **Face Detection**: High accuracy with fine-tuned model
- **Enhancement Quality**: 4x super-resolution with clean results
- **Evaluation**: PSNR >25dB, SSIM >0.8, LPIPS <0.2 targets
- **Processing**: GPU-accelerated for efficient inference

## Course Project Requirements

 **Advanced Implementation**: Fine-tuned models + custom pipeline  
 **Problem-Solving**: GFPGAN → Real-ESRGAN migration  
 **Documentation**: Comprehensive Jupyter notebook analysis  
 **Evaluation**: Multiple quality metrics and comparisons  
 **Reproducibility**: Complete setup and training pipeline  

## Future Enhancements

- **Extended Training**: Additional epochs for YOLO fine-tuning
- **Multi-scale Detection**: Variable input sizes for different face scales  
- **Real-time Optimization**: Video processing capabilities
- **Ensemble Methods**: Combining multiple enhancement approaches

##  Usage Example

```python
from face_enhancer_yoloft import FaceEnhancerFinetuned
from evaluator import ImageEvaluator

# Initialize with fine-tuned models
enhancer = FaceEnhancerFinetuned()
evaluator = ImageEvaluator()

# Process image
results = enhancer.process_image('path/to/image.jpg')

# Visualize results
enhancer.visualize_results(results)

# Evaluate quality
if results['enhanced_crops']:
    metrics = evaluator.evaluate_enhancement(
        results['original_crops'][0], 
        results['enhanced_crops'][0]
    )
    print(f"PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.3f}")
```

## 📋 Project Structure Summary
