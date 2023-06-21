## Create a build parts of Attention U-Net Model

import torch
import numpy as np
from torch import nn
from einops import rearrange

from torchvision.ops import StochasticDepth

from typing import List, Iterable
from models.created_classes.LinSegFormer.LinAttVision import LinformerSelfAttention


class LinAttentionUnetModel(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 seq_len: int,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3,
                 numberConvLayers: int = 2,
                 mode: str = "bilinear",  # Upsampling algoritms
                 num_heads: int = 8,
                 align_corner: bool = False):
        super().__init__()

        # Encoder
        self.encoder_layer_1 = Encoder_block(in_channels=in_channels, out_channels=64, padding=padding,
                                             stride=stride, kernel=kernel, numberConvLayers=numberConvLayers)
        self.encoder_layer_2 = Encoder_block(in_channels=64, out_channels=128, padding=padding,
                                             stride=stride, kernel=kernel, numberConvLayers=numberConvLayers)
        self.encoder_layer_3 = Encoder_block(in_channels=128, out_channels=256, padding=padding,
                                             stride=stride, kernel=kernel, numberConvLayers=numberConvLayers)

        self.encoder_conv_block = ConvBlock(in_channels=256, out_channels=512, padding=padding,
                                            stride=stride, kernel=kernel, numberConvLayers=numberConvLayers)

        # Decoder
        self.decode_layer_1 = Decoder_block(in_channels=512, out_channels=256, seq_len=seq_len, padding=padding,
                                            stride=stride, kernel=kernel, numberConvLayers=numberConvLayers,
                                            mode=mode, align_corner=align_corner, num_heads=num_heads, patch_size=2, overlap_size=0)
        self.decode_layer_2 = Decoder_block(in_channels=256, out_channels=128, seq_len=seq_len, padding=padding,
                                            stride=stride, kernel=kernel, numberConvLayers=numberConvLayers,
                                            mode=mode, align_corner=align_corner, num_heads=num_heads, patch_size=4, overlap_size=0)
        self.decode_layer_3 = Decoder_block(in_channels=128, out_channels=64, seq_len=seq_len, padding=padding,
                                            stride=stride, kernel=kernel, numberConvLayers=numberConvLayers,
                                            mode=mode, align_corner=align_corner, num_heads=num_heads, patch_size=8, overlap_size=0)

        # Output
        self.output = OutConv(in_channels=64, out_channels=out_channels)

    def forward(self, x):
        p1 = self.encoder_layer_1(x)
        p2 = self.encoder_layer_2(p1)
        p3 = self.encoder_layer_3(p2)

        b1 = self.encoder_conv_block(p3)

        d1 = self.decode_layer_1(b1)
        d2 = self.decode_layer_2(d1)
        d3 = self.decode_layer_3(d2)

        output = self.output(d3)
        return output


def project_vk_linformer(v, k, E):
    # project k,v
    v = torch.einsum('b h j d , j k -> b h k d', v, E)
    k = torch.einsum('b h j d , j k -> b h k d', k, E)
    return v, k


def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = super().forward(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x


class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, overlap_size: int):
        super().__init__()
        self.overlapLayer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                padding=overlap_size,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )

    def forward(self, x):
        pathces_img = self.overlapLayer(x)
        return pathces_img


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


# Encoder block: conv_block and pooling
class Encoder_block(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3,
                 numberConvLayers: int = 2):
        super().__init__()

        self.conv = ConvBlock(in_channels=in_channels, out_channels=out_channels, padding=padding,
                              stride=stride, kernel=kernel, numberConvLayers=numberConvLayers)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        """
        param x: input tensor
        return: "s" tensor after conv block and "p" tensor after maxpool layer
        """
        s = self.conv(x)
        s = self.pool(s)
        return s


class Attention_Gate(nn.Module):
    """
    Mechanism attention for UNet
    :param input: size channels input data
    :param reduction_ration: size kernel in conv2d layer
    :param reduction_ration: step for conv2d layer (stride)

    output: tensor == input.shape
    """

    def __init__(self, input: int,
                 output: int,
                 seq_len: int,
                 patch_size: int = 4,
                 overlap_size: int = 1,
                 num_heads: int = 8,
                 mode: str = "bilinear",  # Upsampling algoritms
                 align_corner: bool = False,
                 ):
        super().__init__()

        self.conv_layer = OverlapPatchMerging(in_channels=input,
                                              out_channels=output,
                                              patch_size=patch_size,
                                              overlap_size=overlap_size)

        self.att_gate = LinformerSelfAttention(output, seq_len=seq_len, num_heads=num_heads)

        self.up_layer = nn.Upsample(scale_factor=patch_size, mode=mode, align_corners=align_corner)

    def forward(self, x):
        x = self.conv_layer(x)
        # print(f'x shape: {x.shape}')
        # print(f'x shape: {x.shape}')

        _, _, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        out = self.att_gate(x)
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        out = self.up_layer(out)

        return out


# t1 = torch.randn(1,256, 64, 128)
# att = Attention_Gate(256, 256)
# out = att(t1)


class Decoder_block(nn.Module):
    """
    Upscalling previous layer, put in attention gate. Concat pooling layer and attention gate out.

    UpSample() -> attention gate() -> concat() -> Conv_block()

    :param mode: - upsampling algoritms cab be: 'nearest', 'bilinear', 'linear'
    :param align_corner: - if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels.
    This only has effect when mode is 'linear', 'bilinear'. Default: False
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 seq_len: int,
                 mode: str = "bilinear",  # Upsampling algoritms
                 align_corner: bool = False,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3,
                 numberConvLayers: int = 2,
                 num_heads: int = 8,
                 patch_size: int = 4,
                 overlap_size: int = 1):
        super().__init__()

        self.up_layer = nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corner)
        self.ag = Attention_Gate(input=in_channels, output=out_channels, seq_len=seq_len, patch_size=patch_size, overlap_size=overlap_size,
                 mode=mode, align_corner=align_corner, num_heads=num_heads)
        self.c1 = ConvBlock(in_channels=out_channels, out_channels=out_channels, padding=padding,
                            stride=stride, kernel=kernel, numberConvLayers=numberConvLayers)

    def forward(self, x):
        """
        Concatinate previous upsample layers with out attention gate tensor and put in ConvBlock
        x - previous layer
        s - skip connection
        """
        x = self.up_layer(x)
        s = self.ag(x)
        z = self.c1(s)
        return z


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


# Linatt_model = LinAttentionUnetModel(3, 20, 2048)
# tensor = torch.randn(1, 3, 256, 512)
# out = Linatt_model(tensor)
# print(f'output shape: {out.shape}')
