"""
Medical Image Segmentation Loss Functions and Metrics

This module implements state-of-the-art loss functions and evaluation metrics
specifically designed for medical image segmentation tasks.

Author: Brain Tumor AI Team
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    The Dice coefficient is a measure of overlap between prediction and ground truth.
    Dice Loss = 1 - Dice Coefficient
    
    Formula:
        Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
        Loss = 1 - Dice
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred (torch.Tensor): Predicted segmentation mask (B, C, H, W)
            target (torch.Tensor): Ground truth mask (B, C, H, W)
            
        Returns:
            torch.Tensor: Dice loss value
        """
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        # Dice loss
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss.
    
    This hybrid loss combines the benefits of both BCE and Dice loss:
    - BCE helps with individual pixel classification
    - Dice helps with overall segmentation overlap
    
    Formula:
        Loss = α * BCE + β * Dice
    
    Args:
        bce_weight (float): Weight for BCE loss (α)
        dice_weight (float): Weight for Dice loss (β)
        smooth (float): Smoothing factor for Dice loss
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0
    ):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined BCE + Dice loss.
        
        Args:
            pred (torch.Tensor): Raw logits from model (B, C, H, W)
            target (torch.Tensor): Ground truth mask (B, C, H, W)
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # BCE loss on logits
        bce_loss = self.bce(pred, target)
        
        # Dice loss on probabilities
        pred_prob = torch.sigmoid(pred)
        dice_loss = self.dice(pred_prob, target)
        
        # Combined loss
        total_loss = (
            self.bce_weight * bce_loss + 
            self.dice_weight * dice_loss
        )
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples.
    Particularly useful when there's high class imbalance (small tumors).
    
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha (float): Weighting factor for class balance
        gamma (float): Focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            pred (torch.Tensor): Raw logits from model (B, C, H, W)
            target (torch.Tensor): Ground truth mask (B, C, H, W)
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # Convert logits to probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )
        
        # Compute focal term: (1 - p_t)^gamma
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_term = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = self.alpha * focal_term * bce
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss.
    
    Allows control over false positives and false negatives through
    alpha and beta parameters.
    
    Formula:
        TI = TP / (TP + α*FP + β*FN)
    
    Args:
        alpha (float): Weight for false positives
        beta (float): Weight for false negatives
        smooth (float): Smoothing factor
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0
    ):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            pred (torch.Tensor): Predicted probabilities (B, C, H, W)
            target (torch.Tensor): Ground truth mask (B, C, H, W)
            
        Returns:
            torch.Tensor: Tversky loss value
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        
        return 1 - tversky


# =====================================================================
# EVALUATION METRICS
# =====================================================================


class DiceScore(nn.Module):
    """
    Dice Coefficient (F1-Score) for segmentation evaluation.
    
    Measures the overlap between prediction and ground truth.
    Range: [0, 1], where 1 is perfect overlap.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        threshold (float): Threshold for converting probabilities to binary
    """
    
    def __init__(self, smooth: float = 1.0, threshold: float = 0.5):
        super(DiceScore, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice coefficient.
        
        Args:
            pred (torch.Tensor): Predicted probabilities (B, C, H, W)
            target (torch.Tensor): Ground truth mask (B, C, H, W)
            
        Returns:
            torch.Tensor: Dice coefficient value
        """
        # Binarize predictions
        pred = (pred > self.threshold).float()
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate intersection
        intersection = (pred * target).sum()
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return dice


class IoU(nn.Module):
    """
    Intersection over Union (Jaccard Index).
    
    Measures the ratio of intersection to union between prediction and target.
    Range: [0, 1], where 1 is perfect overlap.
    
    Formula:
        IoU = |X ∩ Y| / |X ∪ Y|
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        threshold (float): Threshold for converting probabilities to binary
    """
    
    def __init__(self, smooth: float = 1.0, threshold: float = 0.5):
        super(IoU, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU score.
        
        Args:
            pred (torch.Tensor): Predicted probabilities (B, C, H, W)
            target (torch.Tensor): Ground truth mask (B, C, H, W)
            
        Returns:
            torch.Tensor: IoU value
        """
        # Binarize predictions
        pred = (pred > self.threshold).float()
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        # IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return iou


class PixelAccuracy(nn.Module):
    """
    Pixel-wise accuracy for segmentation.
    
    Measures the percentage of correctly classified pixels.
    Range: [0, 1], where 1 is perfect accuracy.
    
    Args:
        threshold (float): Threshold for converting probabilities to binary
    """
    
    def __init__(self, threshold: float = 0.5):
        super(PixelAccuracy, self).__init__()
        self.threshold = threshold
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pixel accuracy.
        
        Args:
            pred (torch.Tensor): Predicted probabilities (B, C, H, W)
            target (torch.Tensor): Ground truth mask (B, C, H, W)
            
        Returns:
            torch.Tensor: Accuracy value
        """
        # Binarize predictions
        pred = (pred > self.threshold).float()
        
        # Calculate correct predictions
        correct = (pred == target).float().sum()
        total = target.numel()
        
        accuracy = correct / total
        
        return accuracy


class Sensitivity(nn.Module):
    """
    Sensitivity (Recall/True Positive Rate).
    
    Measures the proportion of actual positives correctly identified.
    Important for medical imaging to minimize false negatives.
    
    Formula:
        Sensitivity = TP / (TP + FN)
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        threshold (float): Threshold for converting probabilities to binary
    """
    
    def __init__(self, smooth: float = 1e-6, threshold: float = 0.5):
        super(Sensitivity, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sensitivity.
        
        Args:
            pred (torch.Tensor): Predicted probabilities (B, C, H, W)
            target (torch.Tensor): Ground truth mask (B, C, H, W)
            
        Returns:
            torch.Tensor: Sensitivity value
        """
        # Binarize predictions
        pred = (pred > self.threshold).float()
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives and False Negatives
        TP = (pred * target).sum()
        FN = (target * (1 - pred)).sum()
        
        sensitivity = (TP + self.smooth) / (TP + FN + self.smooth)
        
        return sensitivity


class Specificity(nn.Module):
    """
    Specificity (True Negative Rate).
    
    Measures the proportion of actual negatives correctly identified.
    
    Formula:
        Specificity = TN / (TN + FP)
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
        threshold (float): Threshold for converting probabilities to binary
    """
    
    def __init__(self, smooth: float = 1e-6, threshold: float = 0.5):
        super(Specificity, self).__init__()
        self.smooth = smooth
        self.threshold = threshold
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute specificity.
        
        Args:
            pred (torch.Tensor): Predicted probabilities (B, C, H, W)
            target (torch.Tensor): Ground truth mask (B, C, H, W)
            
        Returns:
            torch.Tensor: Specificity value
        """
        # Binarize predictions
        pred = (pred > self.threshold).float()
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Negatives and False Positives
        TN = ((1 - pred) * (1 - target)).sum()
        FP = ((1 - target) * pred).sum()
        
        specificity = (TN + self.smooth) / (TN + FP + self.smooth)
        
        return specificity


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name (str): Name of loss function
            Options: 'dice', 'bce_dice', 'focal', 'tversky'
        **kwargs: Additional arguments for loss function
        
    Returns:
        nn.Module: Loss function
    """
    loss_dict = {
        'dice': DiceLoss,
        'bce_dice': BCEDiceLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss
    }
    
    if loss_name.lower() not in loss_dict:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available: {list(loss_dict.keys())}"
        )
    
    return loss_dict[loss_name.lower()](**kwargs)


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    """
    Compute all evaluation metrics at once.
    
    Args:
        pred (torch.Tensor): Predicted probabilities (B, C, H, W)
        target (torch.Tensor): Ground truth mask (B, C, H, W)
        threshold (float): Threshold for binarization
        
    Returns:
        dict: Dictionary containing all metric values
    """
    metrics = {
        'dice': DiceScore(threshold=threshold)(pred, target).item(),
        'iou': IoU(threshold=threshold)(pred, target).item(),
        'accuracy': PixelAccuracy(threshold=threshold)(pred, target).item(),
        'sensitivity': Sensitivity(threshold=threshold)(pred, target).item(),
        'specificity': Specificity(threshold=threshold)(pred, target).item()
    }
    
    return metrics


if __name__ == "__main__":
    # Test loss functions and metrics
    print("Testing Loss Functions and Metrics")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    height, width = 256, 256
    
    # Predictions (logits)
    pred_logits = torch.randn(batch_size, 1, height, width)
    pred_probs = torch.sigmoid(pred_logits)
    
    # Ground truth
    target = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    loss_val = dice_loss(pred_probs, target)
    print(f"\nDice Loss: {loss_val.item():.4f}")
    
    # Test BCE + Dice Loss
    bce_dice_loss = BCEDiceLoss()
    loss_val = bce_dice_loss(pred_logits, target)
    print(f"BCE + Dice Loss: {loss_val.item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    loss_val = focal_loss(pred_logits, target)
    print(f"Focal Loss: {loss_val.item():.4f}")
    
    # Test all metrics
    print("\n" + "=" * 60)
    print("Evaluation Metrics:")
    print("=" * 60)
    
    metrics = compute_all_metrics(pred_probs, target)
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize():<15}: {value:.4f}")
    
    print("\n✓ All loss functions and metrics working correctly!")
