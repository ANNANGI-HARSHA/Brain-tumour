"""
Model Accuracy Evaluation Script

Calculate comprehensive accuracy metrics for the brain tumor classification model.

Author: Brain Tumor AI Team
Date: January 2026
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

# Classification model architecture
class BrainTumorClassifier(nn.Module):
    """Simple CNN classifier for brain tumor types."""
    
    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BrainTumorDataset(Dataset):
    """Dataset for brain tumor images."""
    
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths
        self.samples = []
        for class_name in self.classes:
            class_path = self.root_dir / class_name
            if class_path.exists():
                for img_path in class_path.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('L')  # Grayscale
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def evaluate_model(
    model_path: str,
    test_dir: str,
    device: str = 'cuda',
    img_size: int = 224,
    batch_size: int = 32
) -> Dict:
    """
    Evaluate model and calculate all accuracy metrics.
    
    Args:
        model_path: Path to trained model checkpoint
        test_dir: Directory containing test images
        device: Device to run evaluation on
        img_size: Image size for model input
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing all metrics and predictions
    """
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Brain Tumor Classification - Accuracy Evaluation")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Test directory: {test_dir}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = BrainTumorClassifier(num_classes=4)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Model loaded successfully")
    else:
        print(f"⚠ Model not found at {model_path}")
        print("⚠ Using untrained model for demonstration")
    
    model = model.to(device)
    model.eval()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load test dataset
    print(f"\nLoading test data...")
    test_dataset = BrainTumorDataset(test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Evaluation
    print(f"\nEvaluating model on {len(test_dataset)} images...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print(f"{'='*60}")
    
    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n★ Overall Accuracy: {accuracy*100:.2f}%")
    
    # Per-class metrics
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    f1 = f1_score(all_labels, all_preds, average=None)
    
    # Weighted averages
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\nWeighted Metrics:")
    print(f"  Precision: {precision_weighted*100:.2f}%")
    print(f"  Recall:    {recall_weighted*100:.2f}%")
    print(f"  F1-Score:  {f1_weighted*100:.2f}%")
    
    # Per-class accuracy
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    print(f"\nPer-Class Metrics:")
    print(f"{'-'*60}")
    print(f"{'Class':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*60}")
    
    for i, class_name in enumerate(classes):
        class_accuracy = np.mean(all_preds[all_labels == i] == i)
        print(f"{class_name:<15} {class_accuracy*100:>10.2f}%  {precision[i]*100:>10.2f}%  {recall[i]*100:>10.2f}%  {f1[i]*100:>10.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n{'='*60}")
    print("Confusion Matrix:")
    print(f"{'='*60}")
    print(cm)
    
    # Detailed classification report
    print(f"\n{'='*60}")
    print("Detailed Classification Report:")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Create results dictionary
    results = {
        'overall_accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class_accuracy': {classes[i]: np.mean(all_preds[all_labels == i] == i) 
                               for i in range(len(classes))},
        'per_class_precision': {classes[i]: precision[i] for i in range(len(classes))},
        'per_class_recall': {classes[i]: recall[i] for i in range(len(classes))},
        'per_class_f1': {classes[i]: f1[i] for i in range(len(classes))},
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, classes)
    
    # Plot accuracy metrics
    plot_accuracy_metrics(results, classes)
    
    return results


def plot_confusion_matrix(cm: np.ndarray, classes: List[str]):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    save_path = 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to: {save_path}")
    plt.close()


def plot_accuracy_metrics(results: Dict, classes: List[str]):
    """Plot accuracy metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Per-class accuracy
    ax = axes[0, 0]
    accuracies = [results['per_class_accuracy'][cls] * 100 for cls in classes]
    bars = ax.bar(classes, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Precision comparison
    ax = axes[0, 1]
    precisions = [results['per_class_precision'][cls] * 100 for cls in classes]
    bars = ax.bar(classes, precisions, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Precision (%)', fontsize=12)
    ax.set_title('Per-Class Precision', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Recall comparison
    ax = axes[1, 0]
    recalls = [results['per_class_recall'][cls] * 100 for cls in classes]
    bars = ax.bar(classes, recalls, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Per-Class Recall', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # F1-Score comparison
    ax = axes[1, 1]
    f1_scores = [results['per_class_f1'][cls] * 100 for cls in classes]
    bars = ax.bar(classes, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('F1-Score (%)', fontsize=12)
    ax.set_title('Per-Class F1-Score', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    save_path = 'accuracy_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Accuracy metrics plot saved to: {save_path}")
    plt.close()
    
    # Overall metrics summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [
        results['overall_accuracy'] * 100,
        results['precision_weighted'] * 100,
        results['recall_weighted'] * 100,
        results['f1_weighted'] * 100
    ]
    
    bars = ax.bar(metrics_names, metrics_values, 
                  color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Overall Model Performance', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = 'overall_performance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Overall performance plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "saved_models/classifier_best.pth"
    TEST_DIR = "Testing"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    
    # Check if test directory exists
    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory not found: {TEST_DIR}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = evaluate_model(
            model_path=MODEL_PATH,
            test_dir=TEST_DIR,
            device=DEVICE,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
        
        print(f"\n{'='*60}")
        print("✓ Evaluation Complete!")
        print(f"{'='*60}")
        print(f"\n★ Final Accuracy: {results['overall_accuracy']*100:.2f}%")
        print(f"\nResults saved to:")
        print("  - confusion_matrix.png")
        print("  - accuracy_metrics.png")
        print("  - overall_performance.png")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
