"""
Image Classification Dataset Loader for Brain Tumor Detection

Handles JPG/PNG images organized in class folders for tumor classification.

Author: Brain Tumor AI Team
Date: December 2025
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BrainTumorClassificationDataset(Dataset):
    """
    PyTorch Dataset for brain tumor classification from JPG images.
    
    Supports multi-class classification:
    - glioma
    - meningioma
    - notumor
    - pituitary
    
    Args:
        image_paths (List[str]): List of paths to images
        labels (List[int]): List of class labels
        img_size (Tuple[int, int]): Target image size (height, width)
        transform (Optional[A.Compose]): Albumentations transform pipeline
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        img_size: Tuple[int, int] = (256, 256),
        transform: Optional[A.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.transform = transform
        
        assert len(image_paths) == len(labels), \
            "Number of images and labels must match"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a single image.
        
        Returns:
            image (torch.Tensor): Preprocessed image
            label (torch.Tensor): Class label
        """
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


class BrainTumorClassificationLoader:
    """
    High-level data loader manager for brain tumor classification dataset.
    
    Supports folder-based organization:
    data_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            image2.jpg
    
    Args:
        data_dir (str): Root directory containing class folders
        batch_size (int): Batch size for training
        img_size (Tuple[int, int]): Target image size
        val_split (float): Validation split ratio
        test_split (float): Test split ratio
        num_workers (int): Number of workers for data loading
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        img_size: Tuple[int, int] = (256, 256),
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 4
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        
        # Collect image paths and labels
        self.class_names = []
        self.class_to_idx = {}
        self.image_paths, self.labels = self._collect_data()
        
        # Split data
        self.train_imgs, self.val_imgs, self.test_imgs = None, None, None
        self.train_labels, self.val_labels, self.test_labels = None, None, None
        self._split_data()
    
    def _collect_data(self) -> Tuple[List[str], List[int]]:
        """
        Collect all image paths and labels from class folders.
        
        Returns:
            Tuple[List[str], List[int]]: Image paths and labels
        """
        image_paths = []
        labels = []
        
        # Get class folders
        class_folders = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        self.class_names = [folder.name for folder in class_folders]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"\nFound {len(self.class_names)} classes: {self.class_names}")
        
        # Collect images from each class
        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            
            # Get all image files
            image_files = (
                list(class_folder.glob('*.jpg')) +
                list(class_folder.glob('*.jpeg')) +
                list(class_folder.glob('*.png'))
            )
            
            print(f"  {class_folder.name}: {len(image_files)} images")
            
            for img_path in image_files:
                image_paths.append(str(img_path))
                labels.append(class_idx)
        
        print(f"\nTotal images: {len(image_paths)}")
        
        return image_paths, labels
    
    def _split_data(self):
        """Split data into train, validation, and test sets."""
        # First split: separate test set
        train_val_imgs, test_imgs, train_val_labels, test_labels = train_test_split(
            self.image_paths,
            self.labels,
            test_size=self.test_split,
            random_state=42,
            stratify=self.labels  # Maintain class distribution
        )
        
        # Second split: separate validation set from training
        val_ratio = self.val_split / (1 - self.test_split)
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            train_val_imgs,
            train_val_labels,
            test_size=val_ratio,
            random_state=42,
            stratify=train_val_labels
        )
        
        self.train_imgs = train_imgs
        self.val_imgs = val_imgs
        self.test_imgs = test_imgs
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        
        print(f"\nData split:")
        print(f"  Training: {len(self.train_imgs)} samples")
        print(f"  Validation: {len(self.val_imgs)} samples")
        print(f"  Test: {len(self.test_imgs)} samples")
    
    def get_train_transforms(self) -> A.Compose:
        """Get augmentation pipeline for training data."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            ToTensorV2()
        ])
    
    def get_val_transforms(self) -> A.Compose:
        """Get transform pipeline for validation/test data."""
        return A.Compose([
            ToTensorV2()
        ])
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader with augmentation."""
        train_dataset = BrainTumorClassificationDataset(
            image_paths=self.train_imgs,
            labels=self.train_labels,
            img_size=self.img_size,
            transform=self.get_train_transforms()
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader without augmentation."""
        val_dataset = BrainTumorClassificationDataset(
            image_paths=self.val_imgs,
            labels=self.val_labels,
            img_size=self.img_size,
            transform=self.get_val_transforms()
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader without augmentation."""
        test_dataset = BrainTumorClassificationDataset(
            image_paths=self.test_imgs,
            labels=self.test_labels,
            img_size=self.img_size,
            transform=self.get_val_transforms()
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.class_names)


if __name__ == "__main__":
    # Test the data loader
    print("Brain Tumor Classification Dataset Loader")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = BrainTumorClassificationLoader(
        data_dir='C:/Users/harsh/OneDrive/Desktop/BrainTumourAI/Testing',
        batch_size=8,
        img_size=(256, 256),
        num_workers=2
    )
    
    # Get data loaders
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    # Test loading a batch
    print("\nLoading a sample batch...")
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Labels: {labels.tolist()}")
        print(f"Class names: {data_loader.class_names}")
        break
    
    print("\n✓ Data loader working correctly!")
