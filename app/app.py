"""
Brain Tumor Segmentation Web Application

Interactive Streamlit web app for brain tumor detection and segmentation.

Features:
- Upload MRI images (NIfTI format)
- Real-time tumor detection
- Interactive visualization
- Download predictions
- Model selection

Usage:
    streamlit run app.py

Author: Brain Tumor AI Team
Date: December 2025
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import io

import streamlit as st
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.predictor import BrainTumorPredictor


# Page configuration
st.set_page_config(
    page_title="Brain Tumor Segmentation",
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
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path: str, model_name: str, device: str):
    """Load model with caching."""
    try:
        predictor = BrainTumorPredictor(
            model_path=model_path,
            model_name=model_name,
            device=device,
            img_size=(256, 256)
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def process_uploaded_file(uploaded_file) -> tuple:
    """
    Process uploaded NIfTI file.
    
    Returns:
        tuple: (image_array, file_path)
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    # Load NIfTI file
    nifti_img = nib.load(tmp_path)
    image = nifti_img.get_fdata().astype(np.float32)
    
    return image, tmp_path


def create_visualization(original: np.ndarray, mask: np.ndarray) -> Image.Image:
    """
    Create visualization of original image, mask, and overlay.
    
    Returns:
        PIL.Image: Visualization image
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normalize original for visualization
    original_vis = (original - original.min()) / (original.max() - original.min() + 1e-8)
    
    # Original image
    axes[0].imshow(original_vis, cmap='gray')
    axes[0].set_title('Original MRI', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(mask, cmap='Reds', alpha=0.8)
    axes[1].set_title('Tumor Segmentation', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_vis, cmap='gray')
    mask_overlay = np.zeros((*mask.shape, 3))
    mask_overlay[mask > 0] = [1, 0, 0]  # Red for tumor
    axes[2].imshow(mask_overlay, alpha=0.5)
    axes[2].set_title('Overlay', fontsize=16, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<div class="main-header">🧠 Brain Tumor Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-powered MRI analysis for tumor detection and segmentation</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    model_name = st.sidebar.selectbox(
        "Select Model Architecture",
        ["resunet", "attention_unet", "unet3d", "lightweight_unet3d"],
        help="Choose the deep learning model for segmentation"
    )
    
    # Model path
    model_path = st.sidebar.text_input(
        "Model Checkpoint Path",
        value=f"./saved_models/{model_name}_best.pth",
        help="Path to the trained model checkpoint file"
    )
    
    # Device selection
    device = st.sidebar.radio(
        "Computation Device",
        ["cuda", "cpu"],
        index=0 if torch.cuda.is_available() else 1,
        help="GPU (CUDA) or CPU"
    )
    
    if device == "cuda" and not torch.cuda.is_available():
        st.sidebar.warning("⚠️ CUDA not available. Using CPU instead.")
        device = "cpu"
    
    # Prediction threshold
    threshold = st.sidebar.slider(
        "Prediction Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for binarizing tumor predictions"
    )
    
    st.sidebar.markdown("---")
    
    # Information
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info(
        """
        This application uses deep learning to detect and segment brain tumors in MRI scans.
        
        **Supported formats:** NIfTI (.nii, .nii.gz)
        
        **Models available:**
        - ResUNet
        - Attention U-Net
        - 3D U-Net
        - Lightweight 3D U-Net
        """
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload MRI Image")
        
        uploaded_file = st.file_uploader(
            "Choose a NIfTI file",
            type=['nii', 'gz'],
            help="Upload a brain MRI scan in NIfTI format"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Display file info
            st.markdown("**File Information:**")
            file_size = uploaded_file.size / (1024 * 1024)  # MB
            st.write(f"- Size: {file_size:.2f} MB")
            st.write(f"- Type: {uploaded_file.type}")
    
    with col2:
        st.subheader("🔍 Prediction")
        
        if uploaded_file is not None:
            # Load model button
            if st.button("🚀 Run Segmentation", use_container_width=True):
                
                # Load model
                with st.spinner("Loading model..."):
                    if not Path(model_path).exists():
                        st.error(f"❌ Model not found: {model_path}")
                        st.info("Please train a model first or provide a valid model path.")
                        return
                    
                    predictor = load_model(model_path, model_name, device)
                    
                    if predictor is None:
                        return
                
                # Process image
                with st.spinner("Processing image..."):
                    try:
                        # Load image
                        image_array, tmp_path = process_uploaded_file(uploaded_file)
                        
                        # Get middle slice
                        slice_idx = image_array.shape[2] // 2
                        st.info(f"Analyzing slice {slice_idx}/{image_array.shape[2]}")
                        
                        # Predict
                        original, mask = predictor.predict(tmp_path, threshold=threshold)
                        
                        # Calculate statistics
                        tumor_pixels = np.sum(mask > 0)
                        total_pixels = mask.size
                        tumor_percentage = (tumor_pixels / total_pixels) * 100
                        
                        # Has tumor?
                        has_tumor = tumor_percentage > 0.1  # At least 0.1% coverage
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {e}")
                        return
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Tumor detection status
                if has_tumor:
                    st.error("⚠️ **Tumor Detected**")
                else:
                    st.success("✅ **No Tumor Detected**")
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric(
                        label="Tumor Coverage",
                        value=f"{tumor_percentage:.2f}%"
                    )
                
                with col_m2:
                    st.metric(
                        label="Tumor Pixels",
                        value=f"{tumor_pixels:,}"
                    )
                
                with col_m3:
                    st.metric(
                        label="Total Pixels",
                        value=f"{total_pixels:,}"
                    )
                
                # Visualization
                st.markdown("---")
                st.subheader("🖼️ Visualization")
                
                with st.spinner("Creating visualization..."):
                    viz_img = create_visualization(original, mask)
                    st.image(viz_img, use_column_width=True)
                
                # Download button
                st.markdown("---")
                buf = io.BytesIO()
                viz_img.save(buf, format='PNG')
                buf.seek(0)
                
                st.download_button(
                    label="📥 Download Visualization",
                    data=buf,
                    file_name=f"segmentation_{uploaded_file.name.split('.')[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                st.success("✅ Analysis complete!")
        
        else:
            st.info("👆 Please upload an MRI image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p><strong>Brain Tumor AI</strong> | Advanced Medical Imaging Analysis</p>
            <p style='font-size: 0.9rem;'>⚕️ For research and educational purposes only</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
