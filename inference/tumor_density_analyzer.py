"""
Advanced Tumor Detection with Density Analysis using Green's Theorem

Detects tumor location and calculates density flux using Green's theorem.
Green's Theorem: ∮_C (Pdx + Qdy) = ∬_R (∂Q/∂x - ∂P/∂y) dA

This relates the flux around tumor boundary to density distribution inside.

Author: Brain Tumor AI Team
Date: December 2025
"""

import argparse
import sys
from pathlib import Path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import ndimage
from skimage import measure

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class BrainTumorClassifier(nn.Module):
    """CNN for brain tumor classification."""
    
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super(BrainTumorClassifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
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


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class):
        output = self.model(input_tensor)
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = torch.mean(gradients, dim=(1, 2))
        
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


class TumorDensityAnalyzer:
    """Analyzes tumor density using Green's theorem for flux calculation."""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=device)
            self.class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])
            num_classes = len(self.class_names)
            
            self.model = BrainTumorClassifier(num_classes=num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            
            target_layer = self.model.features[-2]
            self.gradcam = GradCAM(self.model, target_layer)
            print(f"✓ Model loaded from {model_path}")
        else:
            # Initialize without trained model
            self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
            self.model = None
            self.gradcam = None
            print("⚠ No model loaded - using image processing only")
    
    def preprocess_image(self, image_path, img_size=(256, 256)):
        """Load and preprocess image."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        image_resized = cv2.resize(image, img_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor, original_image, image_resized
    
    def calculate_density_field(self, heatmap):
        """
        Calculate density field from heatmap.
        Density represents tumor cell concentration at each point.
        """
        # Normalize heatmap to [0, 1]
        density = heatmap / (heatmap.max() + 1e-8)
        
        # Apply Gaussian smoothing for continuous density field
        density_smooth = ndimage.gaussian_filter(density, sigma=2.0)
        
        return density_smooth
    
    def compute_greens_theorem_flux(self, density_field, threshold=0.3):
        """
        Apply Green's Theorem to calculate flux through tumor boundary.
        
        Green's Theorem: ∮_C (P dx + Q dy) = ∬_R (∂Q/∂x - ∂P/∂y) dA
        
        For flux calculation:
        - P(x,y) = density * y  (vector field x-component)
        - Q(x,y) = -density * x (vector field y-component)
        
        Returns:
            dict: Flux measurements and tumor properties
        """
        h, w = density_field.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Define vector field: F = (P, Q) = (ρ*y, -ρ*x)
        # where ρ is the density field
        P = density_field * y_coords
        Q = -density_field * x_coords
        
        # Calculate partial derivatives for Green's theorem
        # ∂Q/∂x
        dQ_dx = np.gradient(Q, axis=1)
        # ∂P/∂y
        dP_dy = np.gradient(P, axis=0)
        
        # Green's theorem integrand: ∂Q/∂x - ∂P/∂y
        curl = dQ_dx - dP_dy
        
        # Create tumor region mask (high density areas)
        tumor_mask = density_field > threshold
        
        # Calculate total flux over tumor region (double integral)
        total_flux = np.sum(curl * tumor_mask)
        
        # Calculate average flux density
        tumor_area = np.sum(tumor_mask)
        avg_flux_density = total_flux / (tumor_area + 1e-8)
        
        # Find tumor boundary using contours
        tumor_mask_uint8 = (tumor_mask * 255).astype(np.uint8)
        contours = measure.find_contours(tumor_mask_uint8, 128)
        
        # Calculate boundary properties
        if len(contours) > 0:
            largest_contour = max(contours, key=len)
            perimeter = len(largest_contour)
            
            # Line integral around boundary (approximation)
            boundary_flux = 0.0
            for i in range(len(largest_contour)):
                y, x = largest_contour[i]
                y_int, x_int = int(y), int(x)
                if 0 <= y_int < h and 0 <= x_int < w:
                    # Approximate line integral: P dx + Q dy
                    if i < len(largest_contour) - 1:
                        dy = largest_contour[i+1][0] - y
                        dx = largest_contour[i+1][1] - x
                        boundary_flux += P[y_int, x_int] * dx + Q[y_int, x_int] * dy
        else:
            perimeter = 0
            boundary_flux = 0.0
        
        # Calculate density statistics
        tumor_density_values = density_field[tumor_mask]
        
        return {
            'total_flux': float(total_flux),
            'avg_flux_density': float(avg_flux_density),
            'boundary_flux': float(boundary_flux),
            'tumor_area_pixels': int(tumor_area),
            'tumor_perimeter': int(perimeter),
            'max_density': float(tumor_density_values.max()) if len(tumor_density_values) > 0 else 0.0,
            'mean_density': float(tumor_density_values.mean()) if len(tumor_density_values) > 0 else 0.0,
            'density_std': float(tumor_density_values.std()) if len(tumor_density_values) > 0 else 0.0,
            'curl_field': curl,
            'tumor_mask': tumor_mask,
            'contours': contours
        }
    
    def analyze(self, image_path, img_size=(256, 256), threshold=0.3):
        """Complete tumor analysis with density calculation."""
        image_tensor, original_image, image_resized = self.preprocess_image(image_path, img_size)
        
        if self.model is not None:
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)[0]
            
            confidence, pred_idx = torch.max(probabilities, dim=0)
            predicted_class = self.class_names[pred_idx.item()]
            confidence = confidence.item()
            
            # Generate heatmap
            cam = self.gradcam.generate_cam(image_tensor, pred_idx.item())
            heatmap = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
            
            class_probs = {
                name: probabilities[i].item() 
                for i, name in enumerate(self.class_names)
            }
        else:
            # Fallback: use image intensity analysis
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Create pseudo-heatmap from intensity
            heatmap = enhanced.astype(np.float32) / 255.0
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            predicted_class = "unknown"
            confidence = 0.0
            class_probs = {}
        
        # Calculate density field
        density_field = self.calculate_density_field(heatmap)
        
        # Apply Green's theorem for flux calculation
        flux_analysis = self.compute_greens_theorem_flux(density_field, threshold)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'has_tumor': predicted_class != 'notumor',
            'tumor_type': predicted_class if predicted_class != 'notumor' else None,
            'class_probabilities': class_probs,
            'original_image': original_image,
            'heatmap': heatmap,
            'density_field': density_field,
            'flux_analysis': flux_analysis
        }
    
    def visualize_complete(self, result, save_path=None):
        """Create comprehensive visualization with density analysis."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Detection visualizations
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(result['original_image'])
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(result['heatmap'], cmap='hot')
        ax2.set_title('Activation Heatmap', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(result['original_image'])
        ax3.imshow(result['heatmap'], cmap='hot', alpha=0.5)
        ax3.set_title('Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(result['density_field'], cmap='viridis')
        ax4.set_title('Tumor Density Field (ρ)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, label='Density')
        
        # Row 2: Green's theorem analysis
        flux = result['flux_analysis']
        
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(flux['curl_field'], cmap='RdBu_r')
        ax5.set_title('Curl Field (∂Q/∂x - ∂P/∂y)', fontsize=12, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, label='Curl')
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(result['original_image'])
        tumor_mask_display = np.ma.masked_where(~flux['tumor_mask'], flux['tumor_mask'])
        ax6.imshow(tumor_mask_display, cmap='Reds', alpha=0.6)
        ax6.set_title('Tumor Region Mask', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(result['original_image'])
        # Draw contours
        for contour in flux['contours']:
            ax7.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
        ax7.set_title('Tumor Boundary (∂R)', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # Vector field visualization
        ax8 = fig.add_subplot(gs[1, 3])
        h, w = result['density_field'].shape
        y_coords, x_coords = np.mgrid[0:h:20, 0:w:20]
        density_sample = result['density_field'][::20, ::20]
        P_sample = density_sample * y_coords
        Q_sample = -density_sample * x_coords
        ax8.quiver(x_coords, y_coords, Q_sample, -P_sample, scale=h*5)
        ax8.set_title('Vector Field F = (ρy, -ρx)', fontsize=12, fontweight='bold')
        ax8.set_aspect('equal')
        ax8.invert_yaxis()
        
        # Row 3: Statistics and info
        ax9 = fig.add_subplot(gs[2, :2])
        ax9.axis('off')
        
        # Detection info
        if result['has_tumor']:
            title = f"⚠️ TUMOR DETECTED: {result['tumor_type'].upper()}"
            title_color = 'red'
        else:
            title = f"✓ NO TUMOR DETECTED"
            title_color = 'green'
        
        info_text = f"{title}\n"
        info_text += f"Confidence: {result['confidence']*100:.1f}%\n\n"
        
        info_text += "=" * 40 + "\n"
        info_text += "GREEN'S THEOREM FLUX ANALYSIS\n"
        info_text += "=" * 40 + "\n\n"
        
        info_text += f"Total Flux (∬_R curl·dA): {flux['total_flux']:.4f}\n"
        info_text += f"Boundary Flux (∮_C F·dr): {flux['boundary_flux']:.4f}\n"
        info_text += f"Avg Flux Density: {flux['avg_flux_density']:.6f}\n\n"
        
        info_text += "TUMOR METRICS:\n"
        info_text += f"  Area: {flux['tumor_area_pixels']} pixels\n"
        info_text += f"  Perimeter: {flux['tumor_perimeter']} pixels\n"
        info_text += f"  Max Density: {flux['max_density']:.4f}\n"
        info_text += f"  Mean Density: {flux['mean_density']:.4f}\n"
        info_text += f"  Density Std: {flux['density_std']:.4f}\n"
        
        ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Class probabilities
        ax10 = fig.add_subplot(gs[2, 2:])
        ax10.axis('off')
        
        if result['class_probabilities']:
            ax10.text(0.1, 0.95, "CLASS PROBABILITIES:", transform=ax10.transAxes,
                     fontsize=11, fontweight='bold', verticalalignment='top')
            
            y_pos = 0.80
            for class_name, prob in sorted(result['class_probabilities'].items(), 
                                          key=lambda x: x[1], reverse=True):
                color = 'red' if class_name == result['predicted_class'] else 'black'
                ax10.text(0.1, y_pos, f"{class_name}: {prob*100:.1f}%", 
                         transform=ax10.transAxes, fontsize=10, color=color)
                y_pos -= 0.12
        
        plt.suptitle('Tumor Detection & Density Analysis using Green\'s Theorem', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.close()
        return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze tumor density using Green\'s Theorem'
    )
    
    parser.add_argument('--image', type=str, required=True, help='Path to brain MRI image')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model (optional)')
    parser.add_argument('--output', type=str, default=None, help='Output path')
    parser.add_argument('--threshold', type=float, default=0.3, help='Density threshold')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("TUMOR DENSITY ANALYZER - GREEN'S THEOREM FLUX CALCULATION")
    print("="*70 + "\n")
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Device: {device}\n")
    
    # Create analyzer
    analyzer = TumorDensityAnalyzer(args.model, device=device)
    
    # Analyze image
    print(f"Analyzing: {args.image}\n")
    result = analyzer.analyze(args.image, img_size=(args.img_size, args.img_size), 
                             threshold=args.threshold)
    
    # Print results
    print("="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    if result['has_tumor']:
        print(f"\n⚠️  TUMOR DETECTED: {result['tumor_type'].upper()}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
    else:
        print(f"\n✓  NO TUMOR DETECTED")
    
    flux = result['flux_analysis']
    print(f"\nGREEN'S THEOREM FLUX ANALYSIS:")
    print(f"  Total Flux: {flux['total_flux']:.4f}")
    print(f"  Boundary Flux: {flux['boundary_flux']:.4f}")
    print(f"  Avg Flux Density: {flux['avg_flux_density']:.6f}")
    print(f"\nTUMOR METRICS:")
    print(f"  Area: {flux['tumor_area_pixels']} pixels")
    print(f"  Perimeter: {flux['tumor_perimeter']} pixels")
    print(f"  Max Density: {flux['max_density']:.4f}")
    print(f"  Mean Density: {flux['mean_density']:.4f}")
    print(f"  Density Std Dev: {flux['density_std']:.4f}")
    print("\n" + "="*70 + "\n")
    
    # Visualize
    output_path = args.output or Path(args.image).stem + "_density_analysis.png"
    analyzer.visualize_complete(result, save_path=output_path)


if __name__ == "__main__":
    main()
