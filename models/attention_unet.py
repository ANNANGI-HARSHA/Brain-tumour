"""
Attention U-Net Architecture

U-Net with attention gates to focus on relevant regions and suppress
irrelevant features. Attention mechanisms help the model focus on tumor regions.

Reference:
    Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    arXiv:1804.03999

Author: Brain Tumor AI Team
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate mechanism for skip connections.
    
    The attention gate learns to focus on target structures of varying shapes
    and sizes by suppressing irrelevant regions in feature maps.
    
    Formula:
        α = σ(ψ(σ(W_x * x + W_g * g + b)) + b_ψ)
        output = α ⊙ x
    
    Args:
        F_g (int): Number of channels in gating signal
        F_l (int): Number of channels in skip connection
        F_int (int): Number of intermediate channels
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(
        self,
        g: torch.Tensor,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through attention gate.
        
        Args:
            g (torch.Tensor): Gating signal from decoder (B, F_g, H, W)
            x (torch.Tensor): Skip connection from encoder (B, F_l, H', W')
            
        Returns:
            torch.Tensor: Attention-weighted features (B, F_l, H', W')
        """
        # Transform gating signal
        g1 = self.W_g(g)
        
        # Transform skip connection
        x1 = self.W_x(x)
        
        # Upsample gating signal if needed
        if g1.size() != x1.size():
            g1 = F.interpolate(
                g1,
                size=x1.size()[2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Combine and apply attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention weights
        return x * psi


class ConvBlock(nn.Module):
    """
    Standard convolutional block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv block."""
        return self.conv(x)


class UpConv(nn.Module):
    """
    Upsampling convolution block.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConv, self).__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through upconv block."""
        return self.up(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for medical image segmentation.
    
    Key features:
    - Attention gates in skip connections
    - Learns to focus on tumor regions
    - Suppresses irrelevant background features
    
    Architecture:
        Encoder (4 levels) -> Bottleneck -> Decoder (4 levels with attention gates)
    
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
        super(AttentionUNet, self).__init__()
        
        # Encoder (Downsampling path)
        self.conv1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.conv5 = ConvBlock(base_channels * 8, base_channels * 16)
        
        # Decoder (Upsampling path with attention gates)
        self.up4 = UpConv(base_channels * 16, base_channels * 8)
        self.att4 = AttentionGate(
            F_g=base_channels * 8,
            F_l=base_channels * 8,
            F_int=base_channels * 4
        )
        self.upconv4 = ConvBlock(base_channels * 16, base_channels * 8)
        
        self.up3 = UpConv(base_channels * 8, base_channels * 4)
        self.att3 = AttentionGate(
            F_g=base_channels * 4,
            F_l=base_channels * 4,
            F_int=base_channels * 2
        )
        self.upconv3 = ConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = UpConv(base_channels * 4, base_channels * 2)
        self.att2 = AttentionGate(
            F_g=base_channels * 2,
            F_l=base_channels * 2,
            F_int=base_channels
        )
        self.upconv2 = ConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = UpConv(base_channels * 2, base_channels)
        self.att1 = AttentionGate(
            F_g=base_channels,
            F_l=base_channels,
            F_int=base_channels // 2
        )
        self.upconv1 = ConvBlock(base_channels * 2, base_channels)
        
        # Output
        self.output = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Attention U-Net.
        
        Args:
            x (torch.Tensor): Input image (B, C, H, W)
            
        Returns:
            torch.Tensor: Segmentation logits (B, 1, H, W)
        """
        # Encoder path
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        
        x2 = self.conv2(x2)
        x3 = self.pool2(x2)
        
        x3 = self.conv3(x3)
        x4 = self.pool3(x3)
        
        x4 = self.conv4(x4)
        x5 = self.pool4(x4)
        
        # Bottleneck
        x5 = self.conv5(x5)
        
        # Decoder path with attention gates
        # Level 4
        d4 = self.up4(x5)
        x4_att = self.att4(g=d4, x=x4)
        d4 = torch.cat([x4_att, d4], dim=1)
        d4 = self.upconv4(d4)
        
        # Level 3
        d3 = self.up3(d4)
        x3_att = self.att3(g=d3, x=x3)
        d3 = torch.cat([x3_att, d3], dim=1)
        d3 = self.upconv3(d3)
        
        # Level 2
        d2 = self.up2(d3)
        x2_att = self.att2(g=d2, x=x2)
        d2 = torch.cat([x2_att, d2], dim=1)
        d2 = self.upconv2(d2)
        
        # Level 1
        d1 = self.up1(d2)
        x1_att = self.att1(g=d1, x=x1)
        d1 = torch.cat([x1_att, d1], dim=1)
        d1 = self.upconv1(d1)
        
        # Output
        out = self.output(d1)
        
        return out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test Attention U-Net architecture
    print("Attention U-Net Architecture Test")
    print("=" * 60)
    
    # Create model
    model = AttentionUNet(in_channels=1, out_channels=1, base_channels=64)
    
    # Print model summary
    print(f"\nModel: Attention U-Net")
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
    
    print("\n✓ Attention U-Net architecture working correctly!")
