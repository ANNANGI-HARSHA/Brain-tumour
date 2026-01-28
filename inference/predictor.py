"""
Inference Module for Brain Tumor Segmentation

Provides utilities for:
- Loading trained models
- Predicting on single images
- Predicting on 3D volumes
- Visualizing results
- Saving predictions

Author: Brain Tumor AI Team
Date: December 2025
"""

import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.resunet import ResUNet
from models.attention_unet import AttentionUNet
from models.unet3d import UNet3D, LightweightUNet3D


class BrainTumorPredictor:
    """
    Brain tumor segmentation predictor.
    
    Handles model loading, preprocessing, prediction, and visualization.
    
    Args:
        model_path (str): Path to trained model checkpoint
        model_name (str): Name of model architecture
        device (str): Device to run inference on
        img_size (Tuple[int, int]): Target image size
    """
    
    def __init__(
        self,
        model_path: str,
        model_name: str = 'resunet',
        device: str = 'cuda',
        img_size: Tuple[int, int] = (256, 256)
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        print(f"✓ Model loaded: {model_name}")
        print(f"✓ Device: {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint."""
        # Create model
        model_dict = {
            'resunet': ResUNet,
            'attention_unet': AttentionUNet,
            'unet3d': UNet3D,
            'lightweight_unet3d': LightweightUNet3D
        }
        
        if self.model_name not in model_dict:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        model = model_dict[self.model_name](
            in_channels=1,
            out_channels=1
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        return model
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess NIfTI image for inference.
        
        Args:
            image_path (str): Path to NIfTI image
            
        Returns:
            Tuple[torch.Tensor, np.ndarray]: (preprocessed_tensor, original_image)
        """
        # Load NIfTI file
        nifti_img = nib.load(image_path)
        image = nifti_img.get_fdata().astype(np.float32)
        
        # Extract middle slice
        slice_idx = image.shape[2] // 2
        image_slice = image[:, :, slice_idx]
        original = image_slice.copy()
        
        # Normalize
        non_zero = image_slice[image_slice > 0]
        if len(non_zero) > 0:
            mean = non_zero.mean()
            std = non_zero.std()
            if std > 0:
                image_slice = (image_slice - mean) / std
            else:
                image_slice = image_slice - mean
        
        image_slice = np.clip(image_slice, -5, 5)
        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
        
        # Resize
        image_slice = cv2.resize(image_slice, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).float()
        
        return tensor, original
    
    def predict(self, image_path: str, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict tumor segmentation for a single image.
        
        Args:
            image_path (str): Path to NIfTI image
            threshold (float): Threshold for binarizing prediction
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (original_image, predicted_mask)
        """
        # Preprocess
        tensor, original = self.preprocess_image(image_path)
        tensor = tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(tensor)
            pred_prob = torch.sigmoid(output)
        
        # Convert to numpy
        pred_mask = pred_prob.cpu().numpy()[0, 0]
        
        # Binarize
        pred_mask = (pred_mask > threshold).astype(np.uint8)
        
        # Resize back to original size
        pred_mask = cv2.resize(pred_mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return original, pred_mask
    
    def visualize_prediction(
        self,
        original: np.ndarray,
        mask: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize segmentation result.
        
        Args:
            original (np.ndarray): Original MRI image
            mask (np.ndarray): Predicted segmentation mask
            save_path (Optional[str]): Path to save visualization
            show (bool): Whether to display the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Normalize original for visualization
        original_vis = (original - original.min()) / (original.max() - original.min() + 1e-8)
        
        # Original image
        axes[0].imshow(original_vis, cmap='gray')
        axes[0].set_title('Original MRI', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(mask, cmap='Reds', alpha=0.8)
        axes[1].set_title('Predicted Tumor Mask', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(original_vis, cmap='gray')
        # Create colored mask overlay
        mask_overlay = np.zeros((*mask.shape, 3))
        mask_overlay[mask > 0] = [1, 0, 0]  # Red for tumor
        axes[2].imshow(mask_overlay, alpha=0.5)
        axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def predict_and_visualize(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        threshold: float = 0.5,
        show: bool = True
    ):
        """
        Complete pipeline: predict and visualize.
        
        Args:
            image_path (str): Path to input image
            save_path (Optional[str]): Path to save visualization
            threshold (float): Threshold for binarization
            show (bool): Whether to display the plot
        """
        print(f"\nProcessing: {image_path}")
        
        # Predict
        original, mask = self.predict(image_path, threshold=threshold)
        
        # Calculate statistics
        tumor_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        print(f"✓ Prediction complete")
        print(f"  Tumor pixels: {tumor_pixels:,}")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Tumor coverage: {tumor_percentage:.2f}%")
        
        # Visualize
        self.visualize_prediction(original, mask, save_path, show)


class BatchPredictor:
    """
    Batch prediction for multiple images.
    
    Args:
        predictor (BrainTumorPredictor): Initialized predictor
        output_dir (str): Directory to save predictions
    """
    
    def __init__(
        self,
        predictor: BrainTumorPredictor,
        output_dir: str = './predictions'
    ):
        self.predictor = predictor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def predict_directory(
        self,
        image_dir: str,
        file_pattern: str = '*.nii.gz',
        threshold: float = 0.5
    ):
        """
        Predict on all images in a directory.
        
        Args:
            image_dir (str): Directory containing images
            file_pattern (str): File pattern to match
            threshold (float): Threshold for binarization
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob(file_pattern))
        
        print(f"\nFound {len(image_files)} images")
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            
            # Generate output path
            output_name = image_path.stem.replace('.nii', '') + '_prediction.png'
            output_path = self.output_dir / output_name
            
            # Predict and visualize
            try:
                self.predictor.predict_and_visualize(
                    str(image_path),
                    save_path=str(output_path),
                    threshold=threshold,
                    show=False
                )
            except Exception as e:
                print(f"✗ Error processing {image_path.name}: {e}")
        
        print("\n" + "="*60)
        print(f"✓ Batch prediction complete!")
        print(f"  Results saved to: {self.output_dir}")


def save_mask_as_nifti(
    mask: np.ndarray,
    reference_path: str,
    output_path: str
):
    """
    Save predicted mask as NIfTI file using reference image's affine.
    
    Args:
        mask (np.ndarray): Predicted mask
        reference_path (str): Path to reference NIfTI image
        output_path (str): Path to save mask
    """
    # Load reference image to get affine
    ref_img = nib.load(reference_path)
    
    # Create NIfTI image
    mask_img = nib.Nifti1Image(mask, ref_img.affine, ref_img.header)
    
    # Save
    nib.save(mask_img, output_path)
    print(f"✓ Mask saved: {output_path}")


if __name__ == "__main__":
    print("Brain Tumor Segmentation - Inference Module")
    print("=" * 60)
    print("\nThis module provides prediction and visualization utilities.")
    print("\nUsage example:")
    print("  predictor = BrainTumorPredictor(")
    print("      model_path='./saved_models/resunet_best.pth',")
    print("      model_name='resunet'")
    print("  )")
    print("  predictor.predict_and_visualize('path/to/image.nii.gz')")
    print()
