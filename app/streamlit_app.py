"""
Professional Brain Tumor Density Analyzer - Production Ready
Optimized for Google Cloud deployment
"""

import streamlit as st
import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

# [Previous model and analyzer classes remain the same - copying from tumor_analyzer_app.py]
class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super(BrainTumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class GradCAM:
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
        density = heatmap / (heatmap.max() + 1e-8)
        density_smooth = ndimage.gaussian_filter(density, sigma=2.0)
        return density_smooth
    
    def compute_greens_theorem_flux(self, density_field, threshold=0.3):
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
            class_probs = {name: probabilities[i].item() for i, name in enumerate(self.class_names)}
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
    panels = {}
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(result['original_image'])
    ax1.set_title('Original MRI Scan', fontsize=18, fontweight='bold', pad=20)
    ax1.axis('off')
    plt.tight_layout()
    panels['original'] = fig1
    
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    im2 = ax2.imshow(result['heatmap'], cmap='hot')
    ax2.set_title('Activation Heatmap', fontsize=18, fontweight='bold', pad=20)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    plt.tight_layout()
    panels['heatmap'] = fig2
    
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.imshow(result['original_image'])
    ax3.imshow(result['heatmap'], cmap='hot', alpha=0.5)
    ax3.set_title('Tumor Localization', fontsize=18, fontweight='bold', pad=20)
    ax3.axis('off')
    plt.tight_layout()
    panels['overlay'] = fig3
    
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    im4 = ax4.imshow(result['density_field'], cmap='viridis')
    ax4.set_title('Density Field ρ(x,y)', fontsize=18, fontweight='bold', pad=20)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, label='Density')
    plt.tight_layout()
    panels['density'] = fig4
    
    flux = result['flux_analysis']
    
    fig5, ax5 = plt.subplots(figsize=(8, 8))
    im5 = ax5.imshow(flux['curl_field'], cmap='RdBu_r')
    ax5.set_title('Curl Field (∂Q/∂x - ∂P/∂y)', fontsize=18, fontweight='bold', pad=20)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, label='Curl')
    plt.tight_layout()
    panels['curl'] = fig5
    
    fig6, ax6 = plt.subplots(figsize=(8, 8))
    ax6.imshow(result['original_image'])
    tumor_mask_display = np.ma.masked_where(~flux['tumor_mask'], flux['tumor_mask'])
    ax6.imshow(tumor_mask_display, cmap='Reds', alpha=0.7)
    ax6.set_title('Tumor Region Mask', fontsize=18, fontweight='bold', pad=20)
    ax6.axis('off')
    plt.tight_layout()
    panels['mask'] = fig6
    
    fig7, ax7 = plt.subplots(figsize=(8, 8))
    ax7.imshow(result['original_image'])
    for contour in flux['contours']:
        ax7.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=3)
    ax7.set_title('Boundary Contours ∂R', fontsize=18, fontweight='bold', pad=20)
    ax7.axis('off')
    plt.tight_layout()
    panels['boundary'] = fig7
    
    fig8, ax8 = plt.subplots(figsize=(8, 8))
    h, w = result['density_field'].shape
    y_coords, x_coords = np.mgrid[0:h:20, 0:w:20]
    density_sample = result['density_field'][::20, ::20]
    P_sample = density_sample * y_coords
    Q_sample = -density_sample * x_coords
    ax8.quiver(x_coords, y_coords, Q_sample, -P_sample, scale=h*5, color='#2196F3', width=0.003)
    ax8.set_title('Vector Field F=(ρy,-ρx)', fontsize=18, fontweight='bold', pad=20)
    ax8.set_aspect('equal')
    ax8.invert_yaxis()
    ax8.set_facecolor('#f5f5f5')
    plt.tight_layout()
    panels['vector'] = fig8
    
    return panels

st.set_page_config(page_title="Brain Tumor Analyzer", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stApp {background: linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%); font-family: 'Segoe UI', sans-serif;}
@keyframes fadeInDown {from {opacity: 0; transform: translateY(-30px);} to {opacity: 1; transform: translateY(0);}}
@keyframes fadeInUp {from {opacity: 0; transform: translateY(30px);} to {opacity: 1; transform: translateY(0);}}
@keyframes scaleIn {from {opacity: 0; transform: scale(0.9);} to {opacity: 1; transform: scale(1);}}
@keyframes pulse {0%, 100% {transform: scale(1);} 50% {transform: scale(1.05);}}
@keyframes slideInRight {from {opacity: 0; transform: translateX(50px);} to {opacity: 1; transform: translateX(0);}}
@keyframes gradientShift {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
.main-container {background: #f4f8fb; border-radius: 30px; padding: 3rem; margin: 2rem; box-shadow: 0 20px 60px rgba(0,0,0,0.12); animation: scaleIn 0.6s ease-out;}
.hero-header {text-align: center; padding: 3rem 0; animation: fadeInDown 0.8s ease-out;}
.hero-title {font-size: 4rem; font-weight: 900; background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%); background-size: 200% 200%; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: gradientShift 3s ease infinite; margin-bottom: 1rem;}
.hero-subtitle {font-size: 1.5rem; color: #222; font-weight: 300; animation: fadeInUp 1s ease-out;}
.upload-section {background: linear-gradient(135deg, #e0e7ef 0%, #f8fafc 100%); border-radius: 20px; padding: 3rem; margin: 2rem 0; border: 3px dashed #2c5364; text-align: center; animation: fadeInUp 1.2s ease-out; transition: all 0.3s ease;}
.upload-section:hover {transform: translateY(-5px); box-shadow: 0 10px 30px rgba(44, 83, 100, 0.15);}
.upload-icon {font-size: 5rem; animation: pulse 2s ease-in-out infinite; color: #2c5364;}
.metric-card {background: #f8fafc; border-radius: 20px; padding: 2rem; box-shadow: 0 10px 30px rgba(44,83,100,0.08); border-left: 6px solid #2c5364; margin: 1.5rem 0; animation: slideInRight 0.8s ease-out; transition: all 0.3s ease;}
.metric-card:hover {transform: translateY(-10px) scale(1.02); box-shadow: 0 15px 40px rgba(44, 83, 100, 0.18);}
.metric-value {font-size: 3rem; font-weight: 900; background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0.5rem 0;}
.metric-label {font-size: 1.1rem; color: #2c5364; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;}
.status-badge {display: inline-block; padding: 1rem 2rem; border-radius: 50px; font-size: 1.5rem; font-weight: 700; margin: 1rem 0; animation: scaleIn 0.5s ease-out;}
.status-tumor {background: linear-gradient(135deg, #e63946 0%, #ffb703 100%); color: white; animation: pulse 2s ease-in-out infinite;}
.status-clear {background: linear-gradient(135deg, #43aa8b 0%, #577590 100%); color: white;}
.viz-container {background: #f4f8fb; border-radius: 20px; padding: 2rem; margin: 1.5rem 0; box-shadow: 0 10px 30px rgba(44,83,100,0.08); animation: fadeInUp 0.8s ease-out; transition: all 0.3s ease;}
.viz-container:hover {transform: scale(1.02); box-shadow: 0 15px 40px rgba(44,83,100,0.13);}
.section-header {font-size: 2.5rem; font-weight: 800; color: #2c5364; margin: 2rem 0 1rem 0; animation: fadeInDown 0.6s ease-out; border-bottom: 4px solid #2c5364; padding-bottom: 1rem;}
.info-card {background: linear-gradient(135deg, #b6e0fe 0%, #e0e7ef 100%); border-radius: 15px; padding: 2rem; margin: 1rem 0; border-left: 5px solid #0f2027; animation: slideInRight 0.8s ease-out; color: #0f2027;}
.stButton>button {background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%); color: white; font-size: 1.3rem; font-weight: 700; padding: 1rem 3rem; border: none; border-radius: 50px; width: 100%; transition: all 0.3s ease;}
.stButton>button:hover {transform: translateY(-5px); box-shadow: 0 15px 40px rgba(44, 83, 100, 0.18);}
.report-container {background: #f4f8fb; border-radius: 25px; padding: 3rem; margin: 2rem 0; box-shadow: 0 15px 50px rgba(44,83,100,0.13); animation: fadeInUp 0.8s ease-out;}
.report-header {text-align: center; padding: 2rem; background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%); border-radius: 20px; color: white; margin-bottom: 2rem; animation: fadeInDown 0.6s ease-out;}
.report-section {margin: 2rem 0; padding: 2rem; border-radius: 15px; background: #e0e7ef; animation: slideInRight 0.8s ease-out;}
.flux-metric {background: linear-gradient(135deg, #b6e0fe 0%, #e0e7ef 100%); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; border-left: 5px solid #0f2027; animation: fadeInUp 0.8s ease-out; color: #0f2027;}
.metric-row {display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0;}
.metric-name {font-weight: 600; font-size: 1.1rem; color: #0f2027;}
.metric-number {font-size: 1.3rem; font-weight: 700; color: #2c5364;}
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'panels' not in st.session_state:
    st.session_state['panels'] = None

if st.session_state['page'] == 'home':
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="hero-header"><div class="hero-title">🧠 Brain Tumor Analyzer</div><div class="hero-subtitle">Advanced AI-Powered Medical Imaging Analysis with Green\'s Theorem</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-section"><div class="upload-icon">📤</div><h2 style="color: #667eea; margin: 1rem 0;">Upload Brain MRI Scan</h2><p style="font-size: 1.2rem; color: #666;">Drag and drop or click to browse</p></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose MRI Image", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<div class="viz-container">', unsafe_allow_html=True)
            st.image(image, use_column_width=True, caption="Uploaded MRI Scan")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown('<div class="info-card"><strong>Density Threshold</strong></div>', unsafe_allow_html=True)
                density_threshold = st.slider("", 0.1, 0.9, 0.3, 0.05, label_visibility="collapsed")
            with col_b:
                st.markdown('<div class="info-card"><strong>Resolution</strong></div>', unsafe_allow_html=True)
                img_size = st.select_slider("", [128, 256, 512], 256, label_visibility="collapsed")
            with col_c:
                st.markdown(f'<div class="info-card"><strong>File Info</strong><br>Size: {image.size[0]}x{image.size[1]}</div>', unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            if st.button("🔬 ANALYZE TUMOR DENSITY", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Loading AI model...")
                progress_bar.progress(20)
                time.sleep(0.3)
                
                model_path = Path(__file__).parent.parent / "saved_models" / "classifier_best.pth"
                analyzer = TumorDensityAnalyzer(model_path if model_path.exists() else None)
                
                status_text.text("🧠 Processing brain scan...")
                progress_bar.progress(40)
                time.sleep(0.3)
                
                result = analyzer.analyze(image, img_size=(img_size, img_size), threshold=density_threshold)
                
                status_text.text("🌀 Calculating Green's Theorem flux...")
                progress_bar.progress(60)
                time.sleep(0.3)
                
                status_text.text("📊 Generating visualizations...")
                progress_bar.progress(80)
                panels = create_visualization_panels(result)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                time.sleep(0.5)
                
                st.session_state['result'] = result
                st.session_state['panels'] = panels
                st.session_state['page'] = 'report'
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state['page'] == 'report':
    result = st.session_state['result']
    panels = st.session_state['panels']
    flux = result['flux_analysis']
    
    if st.button("← Back to Home"):
        st.session_state['page'] = 'home'
        st.rerun()
    
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown('<div class="report-header"><h1 style="margin: 0; font-size: 3rem;">🧠 Medical Analysis Report</h1><p style="margin-top: 1rem; font-size: 1.3rem; opacity: 0.9;">Brain Tumor Density Analysis with Green\'s Theorem</p></div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">🎯 Detection Results</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">Status</div>', unsafe_allow_html=True)
        if result['has_tumor']:
            st.markdown('<div class="status-badge status-tumor">⚠️ TUMOR DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-clear">✓ NO TUMOR</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Classification</div><div class="metric-value" style="font-size: 2rem;">{result["predicted_class"].upper()}</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{result["confidence"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Tumor Area</div><div class="metric-value" style="font-size: 1.8rem;">{flux["tumor_area_pixels"]:,}px</div></div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">📊 Visual Analysis</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.pyplot(panels['original'])
        st.markdown('<p style="text-align: center; color: #666; font-weight: 600;">Original MRI</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.pyplot(panels['heatmap'])
        st.markdown('<p style="text-align: center; color: #666; font-weight: 600;">Activation Heatmap</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.pyplot(panels['overlay'])
        st.markdown('<p style="text-align: center; color: #666; font-weight: 600;">Tumor Localization</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.pyplot(panels['density'])
        st.markdown('<p style="text-align: center; color: #666; font-weight: 600;">Density Field ρ(x,y)</p></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">🌀 Green\'s Theorem Flux Analysis</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.pyplot(panels['curl'])
        st.markdown('<p style="text-align: center; color: #666; font-weight: 600;">Curl Field</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.pyplot(panels['mask'])
        st.markdown('<p style="text-align: center; color: #666; font-weight: 600;">Tumor Mask</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.pyplot(panels['boundary'])
        st.markdown('<p style="text-align: center; color: #666; font-weight: 600;">Boundary ∂R</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="viz-container">', unsafe_allow_html=True)
        st.pyplot(panels['vector'])
        st.markdown('<p style="text-align: center; color: #666; font-weight: 600;">Vector Field</p></div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">📈 Quantitative Metrics</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="report-section"><h3>🌀 Flux Calculations</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="flux-metric"><div class="metric-row"><span class="metric-name">Total Flux (∬_R curl·dA)</span><span class="metric-number">{flux["total_flux"]:.4f}</span></div><p style="color: #666; margin-top: 0.5rem; font-size: 0.9rem;">Net density accumulation over tumor region</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="flux-metric"><div class="metric-row"><span class="metric-name">Boundary Flux (∮_C F·dr)</span><span class="metric-number">{flux["boundary_flux"]:.4f}</span></div><p style="color: #666; margin-top: 0.5rem; font-size: 0.9rem;">Circulation around tumor boundary</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="flux-metric"><div class="metric-row"><span class="metric-name">Average Flux Density</span><span class="metric-number">{flux["avg_flux_density"]:.6f}</span></div><p style="color: #666; margin-top: 0.5rem; font-size: 0.9rem;">Mean flux per pixel</p></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="report-section"><h3>📏 Morphological Properties</h3>', unsafe_allow_html=True)
        st.markdown(f'<div class="flux-metric"><div class="metric-row"><span class="metric-name">Tumor Area</span><span class="metric-number">{flux["tumor_area_pixels"]:,} px</span></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="flux-metric"><div class="metric-row"><span class="metric-name">Perimeter</span><span class="metric-number">{flux["tumor_perimeter"]:,} px</span></div></div>', unsafe_allow_html=True)
        circularity = (4 * np.pi * flux['tumor_area_pixels']) / (flux['tumor_perimeter']**2 + 1e-8)
        st.markdown(f'<div class="flux-metric"><div class="metric-row"><span class="metric-name">Circularity Index</span><span class="metric-number">{circularity:.4f}</span></div><p style="color: #666; margin-top: 0.5rem; font-size: 0.9rem;">Shape regularity (1.0 = perfect circle)</p></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">🎚️ Density Distribution</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Maximum Density</div><div class="metric-value">{flux["max_density"]:.4f}</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Density</div><div class="metric-value">{flux["mean_density"]:.4f}</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Std Deviation</div><div class="metric-value">{flux["density_std"]:.4f}</div></div>', unsafe_allow_html=True)
    
    if result['class_probabilities']:
        st.markdown('<h2 class="section-header">🎯 Classification Confidence</h2>', unsafe_allow_html=True)
        for class_name, prob in sorted(result['class_probabilities'].items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{class_name.upper()}**")
            st.progress(prob)
            st.markdown(f"<p style='text-align: right; margin-top: -15px; font-weight: 700; color: #667eea;'>{prob*100:.2f}%</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Analyze Another Image", use_container_width=True):
            st.session_state['page'] = 'home'
            st.session_state['result'] = None
            st.session_state['panels'] = None
            st.rerun()

st.markdown("<div style='text-align: center; padding: 3rem 0; color: white;'><p style='font-size: 1.1rem; font-weight: 600;'>Brain Tumor Analyzer | Medical AI Platform</p><p style='font-size: 0.9rem; opacity: 0.8;'>Version 1.0.0 | December 2025</p></div>", unsafe_allow_html=True)
