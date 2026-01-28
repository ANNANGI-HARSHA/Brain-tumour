"""
Quick Run Script - Tumor Density Analysis

Analyzes a random tumor image and calculates density flux using Green's Theorem.

Usage:
    python run_analysis.py

Author: Brain Tumor AI Team
"""

import os
import random
from pathlib import Path
import subprocess
import sys

def find_random_image(data_dir):
    """Find a random image from the dataset."""
    data_path = Path(data_dir)
    
    # Get all tumor class directories (exclude notumor for testing)
    tumor_classes = ['glioma', 'meningioma', 'pituitary']
    
    # Pick random class
    tumor_class = random.choice(tumor_classes)
    class_path = data_path / tumor_class
    
    if not class_path.exists():
        print(f"Error: Directory not found: {class_path}")
        return None
    
    # Get all images in that class
    images = list(class_path.glob('*.jpg'))
    
    if not images:
        print(f"Error: No images found in {class_path}")
        return None
    
    # Pick random image
    random_image = random.choice(images)
    
    return random_image, tumor_class


def main():
    print("\n" + "="*70)
    print("TUMOR DENSITY ANALYSIS - RANDOM IMAGE SELECTION")
    print("="*70 + "\n")
    
    # Configuration
    data_dir = Path(__file__).parent.parent / "Testing"
    inference_dir = Path(__file__).parent
    python_exe = r"C:/Users/harsh/OneDrive/Desktop/BrainTumourAI/.venv/Scripts/python.exe"
    analyzer_script = inference_dir / "tumor_density_analyzer.py"
    
    # Find random image
    print("🎲 Selecting random tumor image...")
    result = find_random_image(data_dir)
    
    if result is None:
        print("\n❌ Failed to find image. Please check the Testing directory.")
        return
    
    image_path, tumor_class = result
    
    print(f"✓ Selected: {image_path.name}")
    print(f"   Tumor Type: {tumor_class.upper()}")
    print(f"   Full Path: {image_path}")
    print()
    
    # Run analyzer
    print("🔬 Running density analysis with Green's Theorem...")
    print("="*70 + "\n")
    
    cmd = [
        python_exe,
        str(analyzer_script),
        "--image", str(image_path),
        "--threshold", "0.3"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, cwd=str(inference_dir))
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        
        # Find output file
        output_file = inference_dir / f"{image_path.stem}_density_analysis.png"
        if output_file.exists():
            print(f"\n📊 Visualization saved: {output_file}")
            print("\n🔍 Analysis includes:")
            print("   • Tumor location detection")
            print("   • Density field calculation (ρ)")
            print("   • Green's Theorem flux analysis (∮_C F·dr = ∬_R curl·dA)")
            print("   • Tumor boundary contours")
            print("   • Vector field visualization")
            print("   • Comprehensive density metrics")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Analysis failed with error code {e.returncode}")
        return
    
    print("\n" + "="*70)
    print("To analyze another image, run this script again!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
