## Create a build path of U-Net Model

import torch
from torch import nn


# Convolution -> BatchNorm -> ReLU
# Convolution Block
class ConvBlock(nn.Module):
    """
    Convolution block layers
    2x : Convolution -> Batch Norm -> ReLU

    NumberConvLayer == 2 or 4
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3,
                 numberConvLayers: int = 2):
        super().__init__()
        if numberConvLayers == 2:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        elif numberConvLayers == 4:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # 2x Done
                nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                # 4x Done
            )

    def forward(self, x):
        return self.conv_block(x)


# Downscalling with MaxPool and then dobleconv
# Maxpool -> DouvleConv
class Down(nn.Module):
    """
    Downscalling with MaxPool and then doubleconv
    MaxPool(layer) -> DoubleConv
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3,
                 numberConvLayers: int = 2):
        super().__init__()
        self.maxpool_conv_block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, padding, stride, kernel, numberConvLayers)
        )

    def forward(self, x):
        return self.maxpool_conv_block(x)


# Upscalling previous block and then double conv
# UpSample -> DoubleConv
class Up(nn.Module):
    """
    Upscalling previous block and then double conv
    UpSample() -> DoubleConv(...)

    `mode` - upsampling algoritms cab be: 'nearest', 'bilinear', 'linear'
    `align_corner` - if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels.
    This only has effect when mode is 'linear', 'bilinear'. Default: False
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 mode: str = "bilinear",  # Upsampling algoritms
                 align_corner: bool = False,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3,
                 numberConvLayers: int = 2):
        super().__init__()
        self.up_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corner),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel,
                      padding=padding,
                      stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_block = ConvBlock(in_channels, out_channels, padding, stride, kernel, numberConvLayers)

    def forward(self, x1, x2):
        """
        Concatinate previous downscalling layers with equals shape and at time block layer
        x1 - previous layer
        x2 - layer for concat
        """
        x1 = self.up_layer(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)


# Conv2d -> output (logits)
class OutConv(nn.Module):
    """
    Output layer
    Conv2d -> output (logits)
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel: int = 1):
        super().__init__()
        self.output_conv = nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel)

    def forward(self, x):
        return self.output_conv(x)
