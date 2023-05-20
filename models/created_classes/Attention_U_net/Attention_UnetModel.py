from models.created_classes.Attention_U_net.Attention_UnetParts import *

from torch import nn

## Create class Attention U-Net Model
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