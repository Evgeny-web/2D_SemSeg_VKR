## Create a build parts of Attention U-Net Model

import torch
from torch import nn

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
        p = self.pool(s)
        return s, p


class Attention_Gate(nn.Module):
    """
    Mechanism attention for UNet
    :param input: is List tensors from previous layer and skip connection
    :param output: output channels for this layer(conv2d)
    :param kernel: kernel_size for conv2d
    :param stride: stride kernel
    :param padding: padding mask

    output: tensor for concat
    """

    def __init__(self, input: list,
                 output,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(input[0], output, kernel_size=kernel, padding=padding, stride=stride),
            nn.BatchNorm2d(output)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(input[1], output, kernel_size=kernel, padding=padding, stride=stride),
            nn.BatchNorm2d(output)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(output, output, kernel_size=kernel, padding=padding, stride=stride),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        """
        :param g: output tensor previous conv block
        :param s: scip connection from equal level encoder
        :return: tensor from attention gate with allocate features
        """
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)

        return out * s


class Decoder_block(nn.Module):
    """
    Upscalling previous layer, put in attention gate. Concat pooling layer and attention gate out.

    UpSample() -> attention gate() -> concat() -> Conv_block()

    :param mode: - upsampling algoritms cab be: 'nearest', 'bilinear', 'linear'
    :param align_corner: - if True, the corner pixels of the input and output tensors are aligned, and thus preserving the values at those pixels.
    This only has effect when mode is 'linear', 'bilinear'. Default: False
    """

    def __init__(self, in_channels: list,
                 out_channels: int,
                 mode: str = "bilinear",  # Upsampling algoritms
                 align_corner: bool = False,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3,
                 numberConvLayers: int = 2):
        super().__init__()

        self.up_layer = nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corner)
        self.ag = Attention_Gate(input=in_channels, output=out_channels, kernel=kernel, padding=padding, stride=stride)
        self.c1 = ConvBlock(in_channels=in_channels[0] + out_channels, out_channels=out_channels, padding=padding,
                            stride=stride, kernel=kernel, numberConvLayers=numberConvLayers)

    def forward(self, x, s):
        """
        Concatinate previous upsample layers with out attention gate tensor and put in ConvBlock
        x - previous layer
        s - skip connection
        """
        x = self.up_layer(x)
        s = self.ag(x, s)
        z = torch.cat([x, s], axis=1)
        z = self.c1(z)
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


class AttentionUnetModel(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 padding: int = 1,
                 stride: int = 1,
                 kernel: int = 3,
                 numberConvLayers: int = 2,
                 mode: str = "bilinear",  # Upsampling algoritms
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
        self.decode_layer_1 = Decoder_block(in_channels=[512, 256], out_channels=256, padding=padding,
                                            stride=stride, kernel=kernel, numberConvLayers=numberConvLayers,
                                            mode=mode, align_corner=align_corner)
        self.decode_layer_2 = Decoder_block(in_channels=[256, 128], out_channels=128, padding=padding,
                                            stride=stride, kernel=kernel, numberConvLayers=numberConvLayers,
                                            mode=mode, align_corner=align_corner)
        self.decode_layer_3 = Decoder_block(in_channels=[128, 64], out_channels=64, padding=padding,
                                            stride=stride, kernel=kernel, numberConvLayers=numberConvLayers,
                                            mode=mode, align_corner=align_corner)

        # Output
        self.output = OutConv(in_channels=64, out_channels=out_channels)

    def forward(self, x):
        s1, p1 = self.encoder_layer_1(x)
        s2, p2 = self.encoder_layer_2(p1)
        s3, p3 = self.encoder_layer_3(p2)

        b1 = self.encoder_conv_block(p3)

        d1 = self.decode_layer_1(b1, s3)
        d2 = self.decode_layer_2(d1, s2)
        d3 = self.decode_layer_3(d2, s1)

        output = self.output(d3)
        return output
