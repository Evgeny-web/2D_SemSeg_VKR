import torch
from torch import nn
from einops import rearrange

from torchvision.ops import StochasticDepth

from typing import List, Iterable
from models.created_classes.SegForm_with_LongFormAtt.longformer2d import *



class LongSegFormer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            widths: List[int],
            depths: List[int],
            all_num_heads: List[int],
            patch_sizes: List[int],
            overlap_sizes: List[int],
            reduction_ratios: List[int],
            mlp_expansions: List[int],
            decoder_channels: int,
            scale_factors: List[int],
            num_classes: int,
            drop_prob: float = 0.0,
    ):
        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return segmentation


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
                stride=overlap_size,
                padding=patch_size // 2,
                bias=False
            ),
            LayerNorm2d(out_channels)
        )

    def forward(self, x):
        pathces_img = self.overlapLayer(x)
        return pathces_img


# print(f"LongFOrmer _________")
# tensor = torch.randn(1,3,512,512)
# overlap = OverlapPatchMerging(3, 64, 7, 4)
# output, nx, ny = overlap(tensor)
# print(output.shape)
# print(f'nx: {nx}, ny: {ny}')
# print('______________')


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.att = Long2DSCSelfAttention(
            channels, num_heads=num_heads, nglo=0
        )

    def forward(self, x):
        _, _, h, w = x.shape
        # print(f'x shape: {x.shape}')
        nx, ny = x.shape[-2:]

        x = rearrange(x, "b c h w -> b (h w) c")
        # print(f'x after rearrange: {x.shape}')
        out = self.att(x, nx, ny)
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out


# ratio = 4
# channels = output.shape[1]
# att = EfficientMultiHeadAttention(channels, ratio)
# output = att(output, nx, ny)
# print(f'Output after att: {output.shape}')

class MixMLP(nn.Sequential):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__(
            # dense layer
            nn.Conv2d(channels, channels, kernel_size=1),
            # depth wise conv
            nn.Conv2d(
                channels,
                channels * expansion,
                kernel_size=3,
                groups=channels,
                padding=1,
            ),
            nn.GELU(),
            # dense layer
            nn.Conv2d(channels * expansion, channels, kernel_size=1),
        )


class ResidualAdd(nn.Module):
    """Just an util layer"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        x = x + out
        return x


class SegFormerEncoderBlock(nn.Sequential):
    def __init__(
            self,
            channels: int,
            reduction_ratio: int = 1,
            num_heads: int = 8,
            mlp_expansion: int = 4,
            drop_path_prob: float = .0
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    EfficientMultiHeadAttention(channels, reduction_ratio, num_heads),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    LayerNorm2d(channels),
                    MixMLP(channels, expansion=mlp_expansion),
                    StochasticDepth(p=drop_path_prob, mode="batch")
                )
            ),
        )


class SegFormerEncoderStage(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            patch_size: int,
            overlap_size: int,
            drop_probs: List[int],
            depth: int = 2,
            reduction_ratio: int = 1,
            num_heads: int = 8,
            mlp_expansion: int = 4,
    ):
        super().__init__()
        self.overlap_patch_merge = OverlapPatchMerging(
            in_channels, out_channels, patch_size, overlap_size,
        )
        self.blocks = nn.Sequential(
            *[
                SegFormerEncoderBlock(
                    out_channels, reduction_ratio, num_heads, mlp_expansion, drop_probs[i]
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm2d(out_channels)


def chunks(data: Iterable, sizes: List[int]):
    """
    Given an iterable, returns slices using sizes as indices

    Example:
        {
        data = [1,2,3,4,5]
        sizes = [2,3]
        list(chunks(data, sizes)) # out [[1,2], [3,4,5]]
        }
    """
    curr = 0
    for size in sizes:
        chunk = data[curr: curr + size]
        curr += size
        yield chunk


class SegFormerEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            widths: List[int],
            depths: List[int],
            all_num_heads: List[int],
            patch_sizes: List[int],
            overlap_sizes: List[int],
            reduction_ratios: List[int],
            mlp_expansions: List[int],
            drop_prob: float = .0
    ):
        super().__init__()
        # create drop paths probabilities (one for each stage's block)
        drop_probs = [x.item() for x in torch.linspace(0, drop_prob, sum(depths))]
        self.stages = nn.ModuleList(
            [
                SegFormerEncoderStage(*args)
                for args in zip(
                [in_channels, *widths],
                widths,
                patch_sizes,
                overlap_sizes,
                chunks(drop_probs, sizes=depths),
                depths,
                reduction_ratios,
                all_num_heads,
                mlp_expansions
            )
            ]
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class SegFormerDecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )


class SegFormerDecoder(nn.Module):
    def __init__(self, out_channels: int, widths: List[int], scale_factors: List[int]):
        super().__init__()
        self.stages = nn.ModuleList(
            [
                SegFormerDecoderBlock(in_channels, out_channels, scale_factor)
                for in_channels, scale_factor in zip(widths, scale_factors)
            ]
        )

    def forward(self, features):
        new_features = []
        for feature, stage in zip(features, self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features


class SegFormerSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, num_features: int = 4):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(),  # why relu? Who knows
            nn.BatchNorm2d(channels)  # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x

# r = 4
# channels = 8
# x = torch.randn((1, channels, 64, 64))
# _, _, h, w = x.shape
# # we want a vector of shape 1, 8, 32, 32
# x = rearrange(x, "b c h w -> b (h w) c") # shape = [1, 4096, 8]
# x = rearrange(x, "b (hw r) c -> b hw (c r)", r=r) # shape = [1, 1024, 32]
# reducer = nn.Linear(channels*r, channels)
# x = reducer(x) # shape = [1, 1024, 8]
# half_r = r // 2
# x = rearrange(x, "b (h w) c -> b c h w", h=h//half_r) # shape = [1, 8, 32, 32]
# print(x.shape)
#
# x = torch.randn((1, channels, 64, 64))
# block = EfficientMultiHeadAttention(channels, reduction_ratio=r)
# print(block(x).shape)


# segformer = LongSegFormer(
#     in_channels=3,
#     widths=[64, 128, 256, 512],
#     depths=[3, 4, 6, 3],
#     all_num_heads=[1, 2, 4, 8],
#     patch_sizes=[7, 3, 3, 3],
#     overlap_sizes=[4, 2, 2, 2],
#     reduction_ratios=[8, 4, 2, 1],
#     mlp_expansions=[4, 4, 4, 4],
#     decoder_channels=256,
#     scale_factors=[8, 4, 2, 1],
#     num_classes=20,
# )
#
# segmentation = segformer(torch.randn((4, 3, 256, 512)))
# print(segmentation.shape[2])
# print(segmentation.size())
