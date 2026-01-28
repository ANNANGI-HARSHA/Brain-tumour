# Brain Tumor Detection & Localization - Quick Guide

## What This System Does

This system takes a brain MRI image and:
1. **Detects** if there's a tumor present
2. **Identifies** the tumor type (glioma, meningioma, or pituitary)
3. **Shows WHERE** the tumor is located using a heatmap

## Training the Model

```bash
cd training
python train_classification.py --data_dir C:\Users\harsh\OneDrive\Desktop\BrainTumourAI\Testing --epochs 50 --batch_size 16
```

This will:
- Train on your Testing dataset (4 classes: glioma, meningioma, notumor, pituitary)
- Save best model to `../saved_models/classifier_best.pth`
- Generate training curves
- Take ~30-60 minutes depending on your GPU

## Using the Detection System

Once training is complete, detect and localize tumors:

```bash
cd inference
python detect_and_localize.py --image path/to/brain_scan.jpg --model ../saved_models/classifier_best.pth
```

### Example Output

The system will:

**If TUMOR is detected:**
```
⚠️  TUMOR DETECTED
   Type: GLIOMA
   Confidence: 94.3%

Class Probabilities:
   glioma: 94.3%
   meningioma: 3.2%
   pituitary: 1.8%
   notumor: 0.7%
```

**If NO TUMOR:**
```
✓  NO TUMOR DETECTED
   Confidence: 96.7%

Class Probabilities:
   notumor: 96.7%
   glioma: 2.1%
   meningioma: 0.8%
   pituitary: 0.4%
```

## Visualization

The system generates a 4-panel visualization:

1. **Original Image** - The input brain MRI
2. **Heatmap** - Red regions show where tumor is likely located
3. **Overlay** - Heatmap overlaid on original image
4. **Detection Info** - Classification results and probabilities

### How to Read the Heatmap

- **Red areas** = High probability of tumor presence
- **Yellow areas** = Moderate probability
- **Blue/Dark areas** = Low/No tumor detected

The heatmap uses GradCAM (Gradient-weighted Class Activation Mapping) to show which regions the model focused on when making its prediction.

## Batch Processing

To process multiple images:

```python
from inference.detect_and_localize import TumorDetector

detector = TumorDetector('../saved_models/classifier_best.pth')

images = ['scan1.jpg', 'scan2.jpg', 'scan3.jpg']
for img_path in images:
    result = detector.detect(img_path)
    
    if result['has_tumor']:
        print(f"{img_path}: TUMOR ({result['tumor_type']}) - {result['confidence']*100:.1f}%")
    else:
        print(f"{img_path}: NO TUMOR - {result['confidence']*100:.1f}%")
```

## Options

### Training Options
- `--epochs 50` - Number of training epochs (more = better but slower)
- `--batch_size 16` - Batch size (decrease if out of memory)
- `--learning_rate 0.001` - Initial learning rate
- `--img_size 256` - Image size (higher = more detail but slower)

### Detection Options
- `--output results.png` - Save visualization to specific file
- `--img_size 256` - Must match training size
- `--device cuda` - Force GPU usage (if available)

## Supported Image Formats

- JPG/JPEG
- PNG
- BMP
- TIFF

## System Requirements

- **Minimum**: 8GB RAM, CPU-only (slower training)
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM
- **Python**: 3.8+
- **Dependencies**: PyTorch, OpenCV, Matplotlib

## How It Works

1. **Training Phase:**
   - Model learns to classify images into 4 categories
   - Uses CNN architecture with 4 blocks
   - ~11M parameters optimized for medical imaging

2. **Detection Phase:**
   - Input image is preprocessed and normalized
   - Model predicts class and confidence
   - GradCAM generates localization heatmap

3. **Localization:**
   - GradCAM analyzes which image regions activated the model
   - Creates a heatmap showing important regions
   - Highlights where the tumor is most likely located

## Accuracy Tips

- Use consistent image sizes
- Ensure images are well-lit and clear
- Model performs best on MRI scans similar to training data
- Confidence below 70% may indicate uncertain predictions

## Troubleshooting

**Training too slow?**
- Reduce `--batch_size` to 8 or 4
- Reduce `--img_size` to 128
- Use GPU if available

**Out of memory?**
- Reduce batch size
- Reduce image size
- Close other applications

**Low accuracy?**
- Train for more epochs (100+)
- Ensure dataset is balanced
- Check if images are preprocessed correctly

## Next Steps

After training completes:
1. Check `saved_models/classifier_best.pth` exists
2. View training curves in `saved_models/classifier_training_curves.png`
3. Test detection on a sample image
4. Process your entire test set
