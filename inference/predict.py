"""
Predict Brain Tumor Segmentation

Command-line interface for running inference on MRI images.

Usage:
    python predict.py --model_path ./saved_models/resunet_best.pth --image path/to/image.nii.gz

Author: Brain Tumor AI Team
Date: December 2025
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from inference.predictor import BrainTumorPredictor, BatchPredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Predict brain tumor segmentation from MRI images'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='resunet',
        choices=['resunet', 'attention_unet', 'unet3d', 'lightweight_unet3d'],
        help='Model architecture name'
    )
    
    # Input arguments
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to single MRI image (.nii or .nii.gz)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Directory containing multiple MRI images for batch prediction'
    )
    parser.add_argument(
        '--file_pattern',
        type=str,
        default='*.nii.gz',
        help='File pattern for batch prediction (default: *.nii.gz)'
    )
    
    # Prediction arguments
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binarizing prediction (0.0-1.0)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help='Image size for prediction'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save visualization (for single image)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./predictions',
        help='Directory to save predictions (for batch)'
    )
    parser.add_argument(
        '--no_show',
        action='store_true',
        help='Do not display visualization'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("Brain Tumor Segmentation - Inference")
    print("="*60)
    
    # Validate input
    if args.image is None and args.image_dir is None:
        print("\nError: Must specify either --image or --image_dir")
        return
    
    if args.image is not None and args.image_dir is not None:
        print("\nError: Cannot specify both --image and --image_dir")
        return
    
    # Create predictor
    print(f"\nLoading model...")
    print(f"  Model path: {args.model_path}")
    print(f"  Model name: {args.model_name}")
    print(f"  Device: {args.device}")
    
    try:
        predictor = BrainTumorPredictor(
            model_path=args.model_path,
            model_name=args.model_name,
            device=args.device,
            img_size=(args.img_size, args.img_size)
        )
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return
    
    print()
    
    # Single image prediction
    if args.image is not None:
        print("Mode: Single image prediction")
        print(f"Input: {args.image}")
        
        if not Path(args.image).exists():
            print(f"\n✗ Error: Image not found: {args.image}")
            return
        
        try:
            predictor.predict_and_visualize(
                image_path=args.image,
                save_path=args.output,
                threshold=args.threshold,
                show=not args.no_show
            )
            
            if args.output:
                print(f"\n✓ Success! Visualization saved to: {args.output}")
            else:
                print(f"\n✓ Success!")
                
        except Exception as e:
            print(f"\n✗ Error during prediction: {e}")
            import traceback
            traceback.print_exc()
    
    # Batch prediction
    else:
        print("Mode: Batch prediction")
        print(f"Input directory: {args.image_dir}")
        print(f"File pattern: {args.file_pattern}")
        print(f"Output directory: {args.output_dir}")
        
        if not Path(args.image_dir).exists():
            print(f"\n✗ Error: Directory not found: {args.image_dir}")
            return
        
        try:
            batch_predictor = BatchPredictor(
                predictor=predictor,
                output_dir=args.output_dir
            )
            
            batch_predictor.predict_directory(
                image_dir=args.image_dir,
                file_pattern=args.file_pattern,
                threshold=args.threshold
            )
            
        except Exception as e:
            print(f"\n✗ Error during batch prediction: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
