"""
Professional Web Interface for Brain Tumor Density Analysis

A comprehensive Streamlit-based web application for tumor detection and 
density analysis using Green's Theorem.

Author: Brain Tumor AI Team
Date: December 2025
"""

import streamlit as st
import sys
from pathlib import Path
import io
import base64

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy import ndimage
from skimage import measure
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


# Model Architecture
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
        else:
            self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
            self.model = None
            self.gradcam = None
    
    def preprocess_image(self, image, img_size=(256, 256)):
        """Preprocess PIL image."""
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        original_image = image.copy()
        image_resized = cv2.resize(image, img_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor, original_image, image_resized
    
    def calculate_density_field(self, heatmap):
        """Calculate density field from heatmap."""
        density = heatmap / (heatmap.max() + 1e-8)
        density_smooth = ndimage.gaussian_filter(density, sigma=2.0)
        return density_smooth
    
    def compute_greens_theorem_flux(self, density_field, threshold=0.3):
        """Apply Green's Theorem to calculate flux through tumor boundary."""
        h, w = density_field.shape
        
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        P = density_field * y_coords
        Q = -density_field * x_coords
        
        dQ_dx = np.gradient(Q, axis=1)
        dP_dy = np.gradient(P, axis=0)
        
        curl = dQ_dx - dP_dy
        
        tumor_mask = density_field > threshold
        
        total_flux = np.sum(curl * tumor_mask)
        tumor_area = np.sum(tumor_mask)
        avg_flux_density = total_flux / (tumor_area + 1e-8)
        
        tumor_mask_uint8 = (tumor_mask * 255).astype(np.uint8)
        contours = measure.find_contours(tumor_mask_uint8, 128)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=len)
            perimeter = len(largest_contour)
            
            boundary_flux = 0.0
            for i in range(len(largest_contour)):
                y, x = largest_contour[i]
                y_int, x_int = int(y), int(x)
                if 0 <= y_int < h and 0 <= x_int < w:
                    if i < len(largest_contour) - 1:
                        dy = largest_contour[i+1][0] - y
                        dx = largest_contour[i+1][1] - x
                        boundary_flux += P[y_int, x_int] * dx + Q[y_int, x_int] * dy
        else:
            perimeter = 0
            boundary_flux = 0.0
        
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
    
    def analyze(self, image, img_size=(256, 256), threshold=0.3):
        """Complete tumor analysis with density calculation."""
        image_tensor, original_image, image_resized = self.preprocess_image(image, img_size)
        
        if self.model is not None:
            image_tensor = image_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)[0]
            
            confidence, pred_idx = torch.max(probabilities, dim=0)
            predicted_class = self.class_names[pred_idx.item()]
            confidence = confidence.item()
            
            cam = self.gradcam.generate_cam(image_tensor, pred_idx.item())
            heatmap = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
            
            class_probs = {
                name: probabilities[i].item() 
                for i, name in enumerate(self.class_names)
            }
        else:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            heatmap = enhanced.astype(np.float32) / 255.0
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            predicted_class = "Analysis Mode"
            confidence = 0.0
            class_probs = {}
        
        density_field = self.calculate_density_field(heatmap)
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


def create_visualization_panels(result):
    """Create individual visualization panels."""
    panels = {}
    
    # Panel 1: Original Image
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(result['original_image'])
    ax1.set_title('Original MRI Image', fontsize=14, fontweight='bold', pad=15)
    ax1.axis('off')
    plt.tight_layout()
    panels['original'] = fig1
    
    # Panel 2: Heatmap
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    im2 = ax2.imshow(result['heatmap'], cmap='hot')
    ax2.set_title('Tumor Activation Heatmap', fontsize=14, fontweight='bold', pad=15)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    plt.tight_layout()
    panels['heatmap'] = fig2
    
    # Panel 3: Overlay
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    ax3.imshow(result['original_image'])
    ax3.imshow(result['heatmap'], cmap='hot', alpha=0.5)
    ax3.set_title('Overlay Visualization', fontsize=14, fontweight='bold', pad=15)
    ax3.axis('off')
    plt.tight_layout()
    panels['overlay'] = fig3
    
    # Panel 4: Density Field
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    im4 = ax4.imshow(result['density_field'], cmap='viridis')
    ax4.set_title('Tumor Density Field (ρ)', fontsize=14, fontweight='bold', pad=15)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, label='Density')
    plt.tight_layout()
    panels['density'] = fig4
    
    flux = result['flux_analysis']
    
    # Panel 5: Curl Field
    fig5, ax5 = plt.subplots(figsize=(6, 6))
    im5 = ax5.imshow(flux['curl_field'], cmap='RdBu_r')
    ax5.set_title('Curl Field (∂Q/∂x - ∂P/∂y)', fontsize=14, fontweight='bold', pad=15)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, label='Curl')
    plt.tight_layout()
    panels['curl'] = fig5
    
    # Panel 6: Tumor Mask
    fig6, ax6 = plt.subplots(figsize=(6, 6))
    ax6.imshow(result['original_image'])
    tumor_mask_display = np.ma.masked_where(~flux['tumor_mask'], flux['tumor_mask'])
    ax6.imshow(tumor_mask_display, cmap='Reds', alpha=0.6)
    ax6.set_title('Tumor Region Mask', fontsize=14, fontweight='bold', pad=15)
    ax6.axis('off')
    plt.tight_layout()
    panels['mask'] = fig6
    
    # Panel 7: Contours
    fig7, ax7 = plt.subplots(figsize=(6, 6))
    ax7.imshow(result['original_image'])
    for contour in flux['contours']:
        ax7.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
    ax7.set_title('Tumor Boundary (∂R)', fontsize=14, fontweight='bold', pad=15)
    ax7.axis('off')
    plt.tight_layout()
    panels['boundary'] = fig7
    
    # Panel 8: Vector Field
    fig8, ax8 = plt.subplots(figsize=(6, 6))
    h, w = result['density_field'].shape
    y_coords, x_coords = np.mgrid[0:h:20, 0:w:20]
    density_sample = result['density_field'][::20, ::20]
    P_sample = density_sample * y_coords
    Q_sample = -density_sample * x_coords
    ax8.quiver(x_coords, y_coords, Q_sample, -P_sample, scale=h*5, color='blue')
    ax8.set_title('Vector Field F = (ρy, -ρx)', fontsize=14, fontweight='bold', pad=15)
    ax8.set_aspect('equal')
    ax8.invert_yaxis()
    ax8.set_facecolor('#f0f0f0')
    plt.tight_layout()
    panels['vector'] = fig8
    
    return panels


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 for display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# Streamlit Configuration
st.set_page_config(
    page_title="Brain Tumor Density Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .metric-title {
        font-size: 1rem;
        color: #666;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        color: #1f77b4;
        font-weight: bold;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 Brain Tumor Density Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Medical Imaging Analysis using Green\'s Theorem for Density Flux Calculation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/brain.png", width=100)
    st.markdown("## 🔬 Analysis Settings")
    
    st.markdown("---")
    
    density_threshold = st.slider(
        "Density Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Threshold for identifying tumor regions (higher = more selective)"
    )
    
    img_size = st.select_slider(
        "Processing Resolution",
        options=[128, 256, 512],
        value=256,
        help="Higher resolution = more detail but slower processing"
    )
    
    st.markdown("---")
    st.markdown("### 📚 About Green's Theorem")
    st.markdown("""
    **Green's Theorem** relates the circulation around a boundary to the flux over the enclosed region:
    
    ∮_C (P dx + Q dy) = ∬_R (∂Q/∂x - ∂P/∂y) dA
    
    This powerful mathematical tool allows us to:
    - Calculate density flux through tumor regions
    - Quantify tumor boundary properties
    - Measure density accumulation/depletion
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Tumor Types")
    st.markdown("""
    - **Glioma**: Most common malignant brain tumor
    - **Meningioma**: Usually benign, arises from meninges
    - **Pituitary**: Affects pituitary gland function
    - **No Tumor**: Healthy brain tissue
    """)

# Main Content
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "📊 Results Dashboard", "📖 User Guide", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Brain MRI Image")
        
        uploaded_file = st.file_uploader(
            "Choose an MRI image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a brain MRI scan image for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Image', use_column_width=True)
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("✅ **Image uploaded successfully!**")
            st.markdown(f"- **File name**: {uploaded_file.name}")
            st.markdown(f"- **Image size**: {image.size[0]} x {image.size[1]} pixels")
            st.markdown(f"- **Format**: {image.format}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🚀 Start Analysis")
        
        if uploaded_file is not None:
            if st.button("🔬 Analyze Tumor Density", key="analyze_btn"):
                with st.spinner("🔄 Processing image and calculating density flux..."):
                    # Initialize analyzer
                    model_path = Path(__file__).parent.parent / "saved_models" / "classifier_best.pth"
                    analyzer = TumorDensityAnalyzer(model_path if model_path.exists() else None)
                    
                    # Analyze
                    result = analyzer.analyze(
                        image,
                        img_size=(img_size, img_size),
                        threshold=density_threshold
                    )
                    
                    # Store in session state
                    st.session_state['result'] = result
                    st.session_state['panels'] = create_visualization_panels(result)
                
                st.success("✅ Analysis complete! View results in the Results Dashboard tab.")
                st.balloons()
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("ℹ️ **Please upload an image to begin analysis**")
            st.markdown("Supported formats: JPG, PNG, BMP, TIFF")
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    if 'result' in st.session_state:
        result = st.session_state['result']
        flux = result['flux_analysis']
        
        # Detection Summary
        st.markdown("## 🎯 Detection Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Detection Status</div>', unsafe_allow_html=True)
            if result['has_tumor']:
                st.markdown('<div class="metric-value" style="color: #ff5252;">⚠️ TUMOR</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-value" style="color: #4caf50;">✓ CLEAR</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Tumor Type</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="font-size: 1.5rem;">{result["predicted_class"].upper()}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Confidence</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{result["confidence"]*100:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-title">Tumor Area</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="font-size: 1.3rem;">{flux["tumor_area_pixels"]:,} px</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("## 📊 Visual Analysis")
        
        panels = st.session_state['panels']
        
        # Row 1: Detection
        st.markdown("### 🔍 Detection Visualizations")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.pyplot(panels['original'])
            st.caption("Original MRI scan showing brain tissue structure")
        
        with col2:
            st.pyplot(panels['heatmap'])
            st.caption("Activation heatmap highlighting areas of interest")
        
        with col3:
            st.pyplot(panels['overlay'])
            st.caption("Combined view of original image with heatmap overlay")
        
        with col4:
            st.pyplot(panels['density'])
            st.caption("Tumor density field ρ(x,y) showing concentration")
        
        st.markdown("---")
        
        # Row 2: Green's Theorem Analysis
        st.markdown("### 🧮 Green's Theorem Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.pyplot(panels['curl'])
            st.caption("Curl field (∂Q/∂x - ∂P/∂y) showing rotation")
        
        with col2:
            st.pyplot(panels['mask'])
            st.caption("Binary tumor region mask highlighting affected area")
        
        with col3:
            st.pyplot(panels['boundary'])
            st.caption("Tumor boundary contours (∂R) for flux calculation")
        
        with col4:
            st.pyplot(panels['vector'])
            st.caption("Vector field F=(ρy,-ρx) showing density flow")
        
        st.markdown("---")
        
        # Metrics
        st.markdown("## 📈 Quantitative Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌀 Green's Theorem Flux Analysis")
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Total Flux (∬_R curl·dA)**  \n`{flux['total_flux']:.4f}`")
            st.markdown("*Represents the net density accumulation over the entire tumor region*")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Boundary Flux (∮_C F·dr)**  \n`{flux['boundary_flux']:.4f}`")
            st.markdown("*Measures the circulation of density around the tumor boundary*")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Average Flux Density**  \n`{flux['avg_flux_density']:.6f}`")
            st.markdown("*Mean flux per pixel, indicating density concentration rate*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📏 Tumor Morphology")
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Tumor Area**  \n`{flux['tumor_area_pixels']:,} pixels`")
            st.markdown("*Total number of pixels classified as tumor tissue*")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f"**Tumor Perimeter**  \n`{flux['tumor_perimeter']:,} pixels`")
            st.markdown("*Length of the tumor boundary contour*")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            circularity = (4 * np.pi * flux['tumor_area_pixels']) / (flux['tumor_perimeter']**2 + 1e-8)
            st.markdown(f"**Circularity Index**  \n`{circularity:.4f}`")
            st.markdown("*Measure of shape regularity (1.0 = perfect circle)*")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Density Statistics
        st.markdown("## 🎚️ Density Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Maximum Density", f"{flux['max_density']:.4f}", help="Highest concentration point in tumor")
        
        with col2:
            st.metric("Mean Density", f"{flux['mean_density']:.4f}", help="Average tumor density")
        
        with col3:
            st.metric("Density Std Dev", f"{flux['density_std']:.4f}", help="Variation in density distribution")
        
        # Class Probabilities
        if result['class_probabilities']:
            st.markdown("---")
            st.markdown("## 🎯 Classification Probabilities")
            
            prob_data = sorted(result['class_probabilities'].items(), key=lambda x: x[1], reverse=True)
            
            for class_name, prob in prob_data:
                st.progress(prob, text=f"{class_name.upper()}: {prob*100:.2f}%")
        
    else:
        st.info("📤 Upload and analyze an image in the 'Upload & Analyze' tab to view results here.")

with tab3:
    st.markdown("## 📖 User Guide")
    
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. **Upload Image**: Go to the 'Upload & Analyze' tab and upload a brain MRI image
    2. **Adjust Settings**: Use the sidebar to configure density threshold and resolution
    3. **Run Analysis**: Click the 'Analyze Tumor Density' button
    4. **View Results**: Switch to the 'Results Dashboard' tab to see comprehensive analysis
    """)
    
    st.markdown("### 🎛️ Settings Explained")
    
    st.markdown("#### Density Threshold")
    st.markdown("""
    - **Low (0.1-0.2)**: More sensitive, captures subtle density changes
    - **Medium (0.3-0.4)**: Balanced detection (recommended)
    - **High (0.5-0.9)**: Only detects dense tumor cores
    """)
    
    st.markdown("#### Processing Resolution")
    st.markdown("""
    - **128px**: Fast processing, lower detail
    - **256px**: Balanced speed and quality (recommended)
    - **512px**: High detail, slower processing
    """)
    
    st.markdown("### 📊 Understanding Results")
    
    st.markdown("#### Detection Status")
    st.markdown("""
    - **⚠️ TUMOR**: Abnormal tissue detected
    - **✓ CLEAR**: No significant abnormalities found
    """)
    
    st.markdown("#### Flux Interpretation")
    st.markdown("""
    - **Negative Flux**: Density accumulation (tumor consuming resources)
    - **Positive Flux**: Density depletion (tumor expanding)
    - **High Absolute Value**: Active tumor region
    - **Low Value**: Stable or dormant region
    """)
    
    st.markdown("#### Density Levels")
    st.markdown("""
    - **High (>0.7)**: Dense tumor core
    - **Medium (0.3-0.7)**: Active tumor region
    - **Low (<0.3)**: Tumor periphery or healthy tissue
    """)

with tab4:
    st.markdown("## ℹ️ About This Application")
    
    st.markdown("""
    ### 🧠 Brain Tumor Density Analyzer
    
    This professional medical imaging analysis tool combines advanced deep learning with 
    mathematical physics to provide comprehensive tumor detection and density analysis.
    
    ### 🔬 Key Features
    
    - **AI-Powered Detection**: Deep learning model for accurate tumor classification
    - **Green's Theorem Analysis**: Mathematical flux calculation for density quantification
    - **Multi-Modal Visualization**: 8 different views of tumor characteristics
    - **Real-Time Processing**: Instant analysis with adjustable parameters
    - **Professional Interface**: Intuitive design for clinical and research use
    
    ### 📐 Mathematical Foundation
    
    The application applies **Green's Theorem** from vector calculus:
    
    ```
    ∮_C (P dx + Q dy) = ∬_R (∂Q/∂x - ∂P/∂y) dA
    ```
    
    This relates the line integral around the tumor boundary to the double integral 
    over the tumor region, providing rigorous quantification of density flux.
    
    ### 🎯 Tumor Classifications
    
    - **Glioma**: Most common primary brain tumor, varying grades of malignancy
    - **Meningioma**: Usually benign tumor of the meninges
    - **Pituitary Adenoma**: Tumor of the pituitary gland
    - **No Tumor**: Healthy brain tissue
    
    ### ⚠️ Medical Disclaimer
    
    This tool is designed for research and educational purposes. It should not be used 
    as the sole basis for medical diagnosis or treatment decisions. Always consult 
    qualified healthcare professionals for medical advice.
    
    ### 👥 Development Team
    
    **Brain Tumor AI Team**  
    Advanced Medical Imaging Research
    
    ### 📅 Version Information
    
    - **Version**: 1.0.0
    - **Release Date**: December 2025
    - **Framework**: Streamlit + PyTorch
    - **Analysis Method**: Green's Theorem Flux Calculation
    """)
    
    st.markdown("---")
    st.markdown("### 🙏 Acknowledgments")
    st.markdown("""
    This application leverages cutting-edge technologies in:
    - Deep Learning (PyTorch)
    - Medical Image Processing (OpenCV, scikit-image)
    - Mathematical Analysis (NumPy, SciPy)
    - Visualization (Matplotlib)
    - Web Interface (Streamlit)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>Brain Tumor Density Analyzer</strong> | Version 1.0.0 | December 2025</p>
    <p>🧠 Advanced Medical Imaging Analysis • 🔬 Research Tool • ⚕️ Educational Purpose</p>
</div>
""", unsafe_allow_html=True)
