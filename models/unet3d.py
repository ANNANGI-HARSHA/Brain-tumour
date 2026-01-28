"""
3D U-Net Architecture

3D U-Net for volumetric medical image segmentation.
Processes entire 3D MRI volumes to leverage spatial context in all three dimensions.

Reference:
    Çiçek et al. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    MICCAI 2016

Author: Brain Tumor AI Team
Date: December 2025
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """
    3D Convolutional block: Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolving kernel
        padding (int): Padding added to all three sides
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        super(ConvBlock3D, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D conv block."""
        return self.conv(x)


class EncoderBlock3D(nn.Module):
    """
    3D Encoder block: Conv block + Max pooling
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock3D, self).__init__()
        
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through encoder block.
        
        Returns:
            tuple: (skip_connection, pooled_output)
        """
        skip = self.conv(x)
        x = self.pool(skip)
        return skip, x


class DecoderBlock3D(nn.Module):
    """
    3D Decoder block: Upsampling + Concatenation + Conv block
    
    Args:
        in_channels (int): Number of input channels
        skip_channels (int): Number of channels in skip connection
        out_channels (int): Number of output channels
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):
        super(DecoderBlock3D, self).__init__()
        
        self.upconv = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )
        
        self.conv = ConvBlock3D(
            in_channels // 2 + skip_channels,
            out_channels
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through decoder block.
        
        Args:
            x (torch.Tensor): Input from previous layer
            skip (torch.Tensor): Skip connection from encoder
            
        Returns:
            torch.Tensor: Decoded features
        """
        x = self.upconv(x)
        
        # Handle potential size mismatch
        if x.size() != skip.size():
            # Pad x to match skip dimensions
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            
            x = nn.functional.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2
            ])
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric brain tumor segmentation.
    
    Architecture:
        - Encoder: 4 levels with 3D convolutions and max pooling
        - Bottleneck: 3D convolutions at lowest resolution
        - Decoder: 4 levels with 3D transposed convolutions and skip connections
        - Output: 1x1x1 convolution for final segmentation
    
    Key advantages:
        - Processes full 3D context
        - Better spatial coherence in segmentation
        - Captures inter-slice dependencies
    
    Args:
        in_channels (int): Number of input channels (1 for single MRI modality)
        out_channels (int): Number of output channels (1 for binary segmentation)
        base_channels (int): Number of channels in first layer
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32  # Reduced for memory efficiency
    ):
        super(UNet3D, self).__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoder1 = EncoderBlock3D(base_channels, base_channels * 2)
        self.encoder2 = EncoderBlock3D(base_channels * 2, base_channels * 4)
        self.encoder3 = EncoderBlock3D(base_channels * 4, base_channels * 8)
        self.encoder4 = EncoderBlock3D(base_channels * 8, base_channels * 16)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(base_channels * 16, base_channels * 16)
        
        # Decoder
        self.decoder4 = DecoderBlock3D(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8
        )
        self.decoder3 = DecoderBlock3D(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4
        )
        self.decoder2 = DecoderBlock3D(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2
        )
        self.decoder1 = DecoderBlock3D(
            base_channels * 2,
            base_channels,
            base_channels
        )
        
        # Output convolution
        self.output = nn.Conv3d(
            base_channels,
            out_channels,
            kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D U-Net.
        
        Args:
            x (torch.Tensor): Input 3D volume (B, C, D, H, W)
                where D is depth, H is height, W is width
            
        Returns:
            torch.Tensor: Segmentation logits (B, 1, D, H, W)
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder path
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        
        # Output
        x = self.output(x)
        
        return x


class LightweightUNet3D(nn.Module):
    """
    Lightweight 3D U-Net for memory-constrained environments.
    
    Uses fewer channels and fewer encoding levels to reduce memory consumption
    while maintaining reasonable segmentation performance.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        base_channels (int): Number of channels in first layer (default: 16)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16
    ):
        super(LightweightUNet3D, self).__init__()
        
        # Encoder (3 levels instead of 4)
        self.encoder1 = EncoderBlock3D(in_channels, base_channels)
        self.encoder2 = EncoderBlock3D(base_channels, base_channels * 2)
        self.encoder3 = EncoderBlock3D(base_channels * 2, base_channels * 4)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 4)
        
        # Decoder
        self.decoder3 = DecoderBlock3D(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2
        )
        self.decoder2 = DecoderBlock3D(
            base_channels * 2,
            base_channels,
            base_channels
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(
                base_channels,
                base_channels,
                kernel_size=2,
                stride=2
            ),
            ConvBlock3D(base_channels, base_channels)
        )
        
        # Output
        self.output = nn.Conv3d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through lightweight 3D U-Net."""
        # Encoder
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x)
        
        # Output
        x = self.output(x)
        
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test 3D U-Net architecture
    print("3D U-Net Architecture Test")
    print("=" * 60)
    
    # Create standard model
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32)
    
    print(f"\nModel: 3D U-Net (Standard)")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass with small volume
    batch_size = 1
    depth, height, width = 64, 128, 128
    
    dummy_input = torch.randn(batch_size, 1, depth, height, width)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Input memory: {dummy_input.element_size() * dummy_input.nelement() / (1024**2):.2f} MB")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test lightweight model
    print("\n" + "=" * 60)
    print("Lightweight 3D U-Net")
    print("=" * 60)
    
    lightweight_model = LightweightUNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=16
    )
    
    print(f"\nModel: 3D U-Net (Lightweight)")
    print(f"Parameters: {count_parameters(lightweight_model):,}")
    
    with torch.no_grad():
        output = lightweight_model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    # Compare model sizes
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"Standard 3D U-Net:    {count_parameters(model):,} parameters")
    print(f"Lightweight 3D U-Net: {count_parameters(lightweight_model):,} parameters")
    print(f"Reduction: {(1 - count_parameters(lightweight_model) / count_parameters(model)) * 100:.1f}%")
    
    print("\n✓ 3D U-Net architecture working correctly!")
