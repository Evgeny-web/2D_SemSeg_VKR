from UnetParts import *

import torch
from torch import nn


## Create class U-Net Model

class UNetModel(nn.Module):
    """
    `up_sample_mode` - upsampling algoritms can be: 'nearest', 'bilinear', 'linear'

    `up_sample_align_corner` - if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels.
    This only has effect when mode is 'linear', 'bilinear'. Default: False

    `NumberConvLayer` == 2 or 4
    """

    def __init__(self, input_channels: int,
                 output_classes: int,
                 up_sample_mode: str = "bilinear",
                 up_sample_align_corner: bool = False,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3,
                 number_conv_layers: int = 2):
        super().__init__()
        self.input_channels = input_channels
        self.output_classes = output_classes
        self.up_sample_mode = up_sample_mode
        self.up_sample_align_corner = up_sample_align_corner
        self.padding = padding
        self.stride = stride
        self.kernel = kernel
        self.number_conv_layers = number_conv_layers

        # Down Layers (Encoder)
        # First layer with input image  (3 color channels)
        self.input_layer = ConvBlock(in_channels=input_channels, out_channels=64, padding=padding, stride=stride,
                                     kernel=kernel, numberConvLayers=number_conv_layers)

        # Blocks layers downscalling image
        self.down_block_1 = Down(in_channels=64, out_channels=128, padding=padding, stride=stride, kernel=kernel,
                                 numberConvLayers=number_conv_layers)
        self.down_block_2 = Down(in_channels=128, out_channels=256, padding=padding, stride=stride, kernel=kernel,
                                 numberConvLayers=number_conv_layers)
        self.down_block_3 = Down(in_channels=256, out_channels=512, padding=padding, stride=stride, kernel=kernel,
                                 numberConvLayers=number_conv_layers)
        self.down_block_4 = Down(in_channels=512, out_channels=1024, padding=padding, stride=stride, kernel=kernel,
                                 numberConvLayers=number_conv_layers)

        # Up layer (Decoder)
        # Blocks layers upscalling image
        self.up_block_1 = Up(in_channels=1024, out_channels=512, mode=up_sample_mode,
                             align_corner=up_sample_align_corner, padding=padding, stride=stride, kernel=kernel,
                             numberConvLayers=number_conv_layers)
        self.up_block_2 = Up(in_channels=512, out_channels=256, mode=up_sample_mode,
                             align_corner=up_sample_align_corner, padding=padding, stride=stride, kernel=kernel,
                             numberConvLayers=number_conv_layers)
        self.up_block_3 = Up(in_channels=256, out_channels=128, mode=up_sample_mode,
                             align_corner=up_sample_align_corner, padding=padding, stride=stride, kernel=kernel,
                             numberConvLayers=number_conv_layers)
        self.up_block_4 = Up(in_channels=128, out_channels=64, mode=up_sample_mode, align_corner=up_sample_align_corner,
                             padding=padding, stride=stride, kernel=kernel, numberConvLayers=number_conv_layers)

        # Last Conv2d layer with output final image (logits)
        self.out_layer = OutConv(in_channels=64, out_channels=output_classes)

    def forward(self, x):
        # Downscalling forward
        x1 = self.input_layer(x)

        x2 = self.down_block_1(x1)
        x3 = self.down_block_2(x2)
        x4 = self.down_block_3(x3)
        x5 = self.down_block_4(x4)

        # Upscalling forward
        # print(f"x5.shape: {x5.shape}")
        # print(f"x4.shape: {x4.shape}")
        x = self.up_block_1(x5, x4)
        # print(f"x.Shape: {x.shape}")
        x = self.up_block_2(x, x3)
        # print(f"x.Shape: {x.shape}")
        x = self.up_block_3(x, x2)
        # print(f"x.Shape: {x.shape}")
        x = self.up_block_4(x, x1)

        logits = self.out_layer(x)
        return logits

