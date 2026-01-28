"""
Brain Tumor Detection and Localization

Detects if an image contains a tumor and shows WHERE the tumor is located
using GradCAM (Gradient-weighted Class Activation Mapping).

Usage:
    python detect_and_localize.py --image path/to/brain_scan.jpg --model ../saved_models/classifier_best.pth

Author: Brain Tumor AI Team
Date: December 2025
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class BrainTumorClassifier(nn.Module):
    """CNN for brain tumor classification."""
    
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super(BrainTumorClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Get feature maps before pooling."""
        return self.features(x)


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class):
        """Generate Class Activation Map."""
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate weights
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


class TumorDetector:
    """Detects and localizes brain tumors."""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        self.class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])
        num_classes = len(self.class_names)
        
        # Create model
        self.model = BrainTumorClassifier(num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Create GradCAM
        target_layer = self.model.features[-2]  # Last conv layer before pooling
        self.gradcam = GradCAM(self.model, target_layer)
        
        print(f"✓ Model loaded (Classes: {self.class_names})")
    
    def preprocess_image(self, image_path, img_size=(256, 256)):
        """Load and preprocess image."""
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # Resize
        image = cv2.resize(image, img_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor, original_image
    
    def detect(self, image_path, img_size=(256, 256)):
        """
        Detect tumor and generate localization map.
        
        Returns:
            dict: Detection results with class, confidence, and localization map
        """
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image_path, img_size)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = F.softmax(output, dim=1)[0]
        
        # Get prediction
        confidence, pred_idx = torch.max(probabilities, dim=0)
        predicted_class = self.class_names[pred_idx.item()]
        confidence = confidence.item()
        
        # Generate localization map using GradCAM
        cam = self.gradcam.generate_cam(image_tensor, pred_idx.item())
        
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Get all class probabilities
        class_probs = {
            name: probabilities[i].item() 
            for i, name in enumerate(self.class_names)
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'has_tumor': predicted_class != 'notumor',
            'tumor_type': predicted_class if predicted_class != 'notumor' else None,
            'class_probabilities': class_probs,
            'original_image': original_image,
            'heatmap': cam_resized
        }
    
    def visualize(self, result, save_path=None):
        """Create visualization of detection and localization."""
        fig = plt.figure(figsize=(16, 5))
        
        # Original image
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(result['original_image'])
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Heatmap
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(result['heatmap'], cmap='jet')
        ax2.set_title('Tumor Localization Heatmap', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Overlay
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(result['original_image'])
        ax3.imshow(result['heatmap'], cmap='jet', alpha=0.5)
        ax3.set_title('Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Detection info
        ax4 = plt.subplot(1, 4, 4)
        ax4.axis('off')
        
        # Title
        if result['has_tumor']:
            title = f"⚠️ TUMOR DETECTED"
            title_color = 'red'
        else:
            title = f"✓ NO TUMOR DETECTED"
            title_color = 'green'
        
        ax4.text(0.5, 0.95, title, ha='center', va='top', 
                fontsize=14, fontweight='bold', color=title_color,
                transform=ax4.transAxes)
        
        # Detection details
        y_pos = 0.80
        
        if result['has_tumor']:
            ax4.text(0.1, y_pos, f"Tumor Type:", fontsize=11, fontweight='bold',
                    transform=ax4.transAxes)
            ax4.text(0.1, y_pos - 0.08, f"{result['tumor_type'].upper()}", 
                    fontsize=11, color='red', transform=ax4.transAxes)
            y_pos -= 0.20
        
        ax4.text(0.1, y_pos, f"Confidence:", fontsize=11, fontweight='bold',
                transform=ax4.transAxes)
        ax4.text(0.1, y_pos - 0.08, f"{result['confidence']*100:.1f}%", 
                fontsize=11, transform=ax4.transAxes)
        y_pos -= 0.20
        
        # Class probabilities
        ax4.text(0.1, y_pos, "Class Probabilities:", fontsize=11, fontweight='bold',
                transform=ax4.transAxes)
        y_pos -= 0.10
        
        for class_name, prob in sorted(result['class_probabilities'].items(), 
                                      key=lambda x: x[1], reverse=True):
            color = 'red' if class_name == result['predicted_class'] else 'black'
            ax4.text(0.1, y_pos, f"{class_name}: {prob*100:.1f}%", 
                    fontsize=10, color=color, transform=ax4.transAxes)
            y_pos -= 0.08
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()
        return fig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect and Localize Brain Tumors'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input brain MRI image'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output visualization'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Image size for model input'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("Brain Tumor Detection and Localization")
    print("="*60 + "\n")
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}\n")
    
    # Create detector
    detector = TumorDetector(args.model, device=device)
    
    # Detect
    print(f"Processing image: {args.image}")
    result = detector.detect(args.image, img_size=(args.img_size, args.img_size))
    
    # Print results
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    
    if result['has_tumor']:
        print(f"\n⚠️  TUMOR DETECTED")
        print(f"   Type: {result['tumor_type'].upper()}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
    else:
        print(f"\n✓  NO TUMOR DETECTED")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
    
    print(f"\nClass Probabilities:")
    for class_name, prob in sorted(result['class_probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"   {class_name}: {prob*100:.1f}%")
    
    print("\n" + "="*60 + "\n")
    
    # Visualize
    output_path = args.output
    if output_path is None:
        output_path = Path(args.image).stem + "_detection.png"
    
    detector.visualize(result, save_path=output_path)


if __name__ == "__main__":
    main()
