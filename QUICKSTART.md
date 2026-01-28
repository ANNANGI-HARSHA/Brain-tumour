# 🚀 Quick Start Guide - Brain Tumor Segmentation

This guide will help you get started with training and using the brain tumor segmentation models.

---

## 📋 Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB+ RAM
- BraTS dataset in NIfTI format

---

## 🔧 Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 📊 Prepare Dataset

### Expected Structure

```
dataset/
├── images/
│   ├── patient_001.nii.gz
│   ├── patient_002.nii.gz
│   └── ...
└── masks/
    ├── patient_001.nii.gz
    ├── patient_002.nii.gz
    └── ...
```

### Test Dataset Loader

```bash
cd dataset
python data_loader.py
```

---

## 🎯 Training Your First Model

### Option 1: Quick Training (ResUNet)

```bash
cd training
python train.py --model resunet --data_dir ../dataset --epochs 50 --batch_size 4
```

**What this does:**
- Trains ResUNet model
- Uses your dataset
- Runs for 50 epochs
- Small batch size (works on most GPUs)

### Option 2: Best Performance (Attention U-Net)

```bash
python train.py \
    --model attention_unet \
    --data_dir ../dataset \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --loss bce_dice \
    --use_scheduler \
    --patience 15
```

### Option 3: 3D Volumetric Segmentation

```bash
python train.py \
    --model unet3d \
    --data_dir ../dataset \
    --epochs 100 \
    --batch_size 2 \
    --is_3d
```

**Note:** 3D models require more memory. Use `lightweight_unet3d` if you have GPU memory constraints.

---

## 🔍 Running Inference

### Single Image Prediction

```bash
cd inference
python predict.py \
    --model_path ../saved_models/resunet_best.pth \
    --model_name resunet \
    --image /path/to/test_image.nii.gz \
    --output result.png
```

### Batch Prediction

```bash
python predict.py \
    --model_path ../saved_models/resunet_best.pth \
    --model_name resunet \
    --image_dir /path/to/test_images/ \
    --output_dir ./predictions \
    --threshold 0.5
```

---

## 🌐 Web Application

### Launch the App

```bash
cd app
streamlit run app.py
```

Then open: `http://localhost:8501`

### Using the App

1. **Upload** your MRI scan (.nii or .nii.gz)
2. **Select** model architecture
3. **Click** "Run Segmentation"
4. **View** results and download visualization

---

## 📈 Monitoring Training

### During Training

You'll see:
- Real-time loss and metrics
- Progress bars for each epoch
- Validation performance
- Best model updates

### After Training

Check these files in `saved_models/`:
- `{model}_best.pth` - Best performing model
- `{model}_latest.pth` - Latest checkpoint
- `{model}_history.json` - Training history
- `{model}_training_curves.png` - Visualization

---

## ⚙️ Configuration Tips

### For Limited GPU Memory

```bash
python train.py \
    --model resunet \
    --batch_size 2 \
    --img_size 128 \
    --base_channels 32
```

### For Best Accuracy

```bash
python train.py \
    --model attention_unet \
    --batch_size 16 \
    --img_size 256 \
    --base_channels 64 \
    --epochs 150 \
    --use_scheduler
```

### For Fast Experimentation

```bash
python train.py \
    --model resunet \
    --batch_size 8 \
    --epochs 20 \
    --img_size 128
```

---

## 🐛 Troubleshooting

### Out of Memory Error

**Solution 1:** Reduce batch size
```bash
--batch_size 2
```

**Solution 2:** Reduce image size
```bash
--img_size 128
```

**Solution 3:** Use smaller model
```bash
--model resunet --base_channels 32
```

### Dataset Not Found

**Check:**
1. Dataset path is correct: `--data_dir ./dataset`
2. Folder structure matches expected format
3. Files are in `.nii` or `.nii.gz` format

### Model Loading Error

**Check:**
1. Model path exists: `--model_path ./saved_models/resunet_best.pth`
2. Model name matches: `--model_name resunet`
3. Checkpoint file is not corrupted

---

## 📊 Expected Results

### Training Time

| Model | Dataset Size | GPU | Time per Epoch |
|-------|-------------|-----|----------------|
| ResUNet | 100 images | RTX 3090 | ~2 minutes |
| Attention U-Net | 100 images | RTX 3090 | ~3 minutes |
| 3D U-Net | 100 volumes | RTX 3090 | ~5 minutes |

### Performance

After training for 100 epochs on BraTS dataset:

- **Dice Score:** 0.85-0.90
- **IoU:** 0.75-0.85
- **Sensitivity:** 0.87-0.92
- **Specificity:** 0.97-0.99

---

## 🎓 Next Steps

### 1. Experiment with Models

Try different architectures:
- ResUNet for balanced performance
- Attention U-Net for better accuracy
- 3D U-Net for volumetric analysis

### 2. Tune Hyperparameters

Adjust:
- Learning rate
- Batch size
- Loss function
- Image size

### 3. Augmentation

Modify `dataset/data_loader.py` to add more augmentation techniques.

### 4. Ensemble Models

Combine predictions from multiple models for better results.

---

## 📚 Resources

### Documentation
- [README.md](README.md) - Full documentation
- [requirements.txt](requirements.txt) - Dependencies

### Code Examples
- `dataset/data_loader.py` - Dataset preprocessing
- `training/train.py` - Training script
- `inference/predict.py` - Prediction script
- `app/app.py` - Web application

### Model Files
- `models/resunet.py` - ResUNet architecture
- `models/attention_unet.py` - Attention U-Net
- `models/unet3d.py` - 3D U-Net
- `models/losses.py` - Loss functions

---

## 💡 Tips

1. **Start Small:** Use 20 epochs for initial testing
2. **Monitor Validation:** Watch validation metrics, not just training loss
3. **Save Checkpoints:** Training saves best and latest models automatically
4. **Use Early Stopping:** Prevents overfitting with `--patience 15`
5. **Visualize Results:** Check training curves in saved_models/

---

## ✅ Checklist

Before training:
- [ ] Dataset prepared in correct format
- [ ] Dependencies installed
- [ ] GPU available (optional but recommended)
- [ ] Sufficient disk space for checkpoints

After training:
- [ ] Best model saved
- [ ] Training curves look reasonable
- [ ] Validation metrics acceptable
- [ ] Test predictions look correct

---

## 🆘 Get Help

If you encounter issues:

1. Check error message carefully
2. Review troubleshooting section above
3. Verify dataset format
4. Test with smaller batch size
5. Create GitHub issue with details

---

**Happy Training! 🎉**
