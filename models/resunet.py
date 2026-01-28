"""
ResUNet - Residual U-Net Architecture

U-Net with residual blocks for improved gradient flow and feature learning.
Combines the benefits of U-Net's skip connections with ResNet's residual learning.

Author: Brain Tumor AI Team
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual Block with skip connection.
    
    Architecture:
        Input --> Conv --> BN --> ReLU --> Conv --> BN --> (+) --> ReLU
          |_______________________________________________|
                           (skip connection)
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for convolution
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super(ResidualBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity or projection)
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        identity = self.skip_connection(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Add skip connection
        out = self.relu(out)
        
        return out


class EncoderBlock(nn.Module):
    """
    Encoder block for ResUNet.
    
    Consists of two residual blocks followed by max pooling.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(EncoderBlock, self).__init__()
        
        self.res_block1 = ResidualBlock(in_channels, out_channels)
        self.res_block2 = ResidualBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through encoder block.
        
        Returns:
            tuple: (skip_connection, pooled_output)
        """
        x = self.res_block1(x)
        x = self.res_block2(x)
        skip = x
        x = self.pool(x)
        return skip, x


class DecoderBlock(nn.Module):
    """
    Decoder block for ResUNet.
    
    Consists of upsampling, concatenation with skip connection,
    and two residual blocks.
    
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
        super(DecoderBlock, self).__init__()
        
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )
        
        self.res_block1 = ResidualBlock(
            in_channels // 2 + skip_channels,
            out_channels
        )
        self.res_block2 = ResidualBlock(out_channels, out_channels)
    
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
        
        # Handle size mismatch between x and skip
        if x.size() != skip.size():
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            x = F.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2
            ])
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        return x


class ResUNet(nn.Module):
    """
    Residual U-Net for medical image segmentation.
    
    Architecture:
        - Encoder: 4 levels with residual blocks and max pooling
        - Bottleneck: Residual blocks at lowest resolution
        - Decoder: 4 levels with upsampling and skip connections
        - Output: 1x1 convolution for final segmentation
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale MRI)
        out_channels (int): Number of output channels (1 for binary segmentation)
        base_channels (int): Number of channels in first layer
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64
    ):
        super(ResUNet, self).__init__()
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.encoder1 = EncoderBlock(base_channels, base_channels * 2)
        self.encoder2 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.encoder3 = EncoderBlock(base_channels * 4, base_channels * 8)
        self.encoder4 = EncoderBlock(base_channels * 8, base_channels * 16)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 16, base_channels * 16),
            ResidualBlock(base_channels * 16, base_channels * 16)
        )
        
        # Decoder
        self.decoder4 = DecoderBlock(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8
        )
        self.decoder3 = DecoderBlock(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4
        )
        self.decoder2 = DecoderBlock(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2
        )
        self.decoder1 = DecoderBlock(
            base_channels * 2,
            base_channels,
            base_channels
        )
        
        # Output convolution
        self.output = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResUNet.
        
        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            torch.Tensor: Segmentation logits (B, 1, H, W)
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


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test ResUNet architecture
    print("ResUNet Architecture Test")
    print("=" * 60)
    
    # Create model
    model = ResUNet(in_channels=1, out_channels=1, base_channels=64)
    
    # Print model summary
    print(f"\nModel: ResUNet")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    height, width = 256, 256
    
    dummy_input = torch.randn(batch_size, 1, height, width)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test with different input sizes
    print("\n" + "=" * 60)
    print("Testing with different input sizes:")
    print("=" * 60)
    
    for size in [128, 256, 512]:
        test_input = torch.randn(1, 1, size, size)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"Input: {test_input.shape} --> Output: {test_output.shape}")
    
    print("\n✓ ResUNet architecture working correctly!")
