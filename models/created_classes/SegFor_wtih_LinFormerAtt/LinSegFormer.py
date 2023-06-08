import torch
from torch import nn
from einops import rearrange

from torchvision.ops import StochasticDepth

from typing import List, Iterable
from models.created_classes.SegFor_wtih_LinFormerAtt.mhsa import compute_mhsa


def project_vk_linformer(v, k, E):
    # project k,v
    v = torch.einsum('b h j d , j k -> b h k d', v, E)
    k = torch.einsum('b h j d , j k -> b h k d', k, E)
    return v, k


class LinSegFormer(nn.Module):
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


class LinformerAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, shared_projection=True, proj_shape=None, trainable_proj=True):
        """
        Based on the Linformer paper
        Link: https://arxiv.org/pdf/2006.04768.pdf

        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head.

            shared_projection: if the projection matrix will be shared among layers
            (it will have to be passed in the forward that way)
            trainable_proj: if the projection matrix E matrix is not shared,
            you can enable this option to make it trainable (non trainable in the paper)
            proj_shape: 2-tuple (tokens,k), where k is the projection dimension of the linformer
            """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5
        self.shared_projection = shared_projection

        if not shared_projection:
            self.E = torch.nn.Parameter(torch.randn(proj_shape), requires_grad=trainable_proj)
            self.k = proj_shape[1]

    def forward(self, x, proj_mat=None):
        assert x.dim() == 3
        E = proj_mat if (self.shared_projection and proj_mat is not None) else self.E
        assert x.shape[1] == E.shape[0], f'{x.shape[1]} Token in the input sequence while' \
                                         f' {E.shape[0]} were provided in the E proj matrix'

        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        if E.device != k.device:
            E = E.to(k.device)

        v, k = project_vk_linformer(v, k, E)

        out = compute_mhsa(q, k, v, scale_factor=self.scale_factor)
        # re-compose: merge heads with dim_head

        out = rearrange(out, "b h i d -> b i (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)


class EfficientMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 1, num_heads: int = 8):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio
            ),
            LayerNorm2d(channels),
        )
        self.att = LinformerAttention(
            channels, heads=num_heads
        )

    def forward(self, x):
        # print(f'shapw x before: {x.shape}')
        # print(f'x dim() : {x.dim()}')
        _, _, h, w = x.shape
        reduced_x = self.reducer(x)
        # print(f'reduced_x after reducer: {reduced_x.shape}')
        # attention needs tensor of shape (batch, sequence_length, channels)
        reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c")
        x = rearrange(x, "b c h w -> b (h w) c")
        # print(f"shape reduces_x: {reduced_x.shape}")
        # print(f"shape x: {x.shape}")
        # proj_shape
        proj_mat = torch.nn.Parameter(torch.randn(x.size()[1], x.size()[1]//4), requires_grad=True)
        # print(f'proj_mat.shape: {proj_mat.shape}')
        out = self.att(x, proj_mat)
        # reshape it back to (batch, channels, height, width)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out

# print(f"LinFOrmer _________")
# tensor = torch.randn(1,3,512,512)
# overlap = OverlapPatchMerging(3, 64, 7, 4)
# output = overlap(tensor)
# print(f'output shape after patch merging: {output.shape}')
# print('______________')
#
# ratio = 4
# channels = output.shape[1]
# att = EfficientMultiHeadAttention(channels, ratio)
# output = att(output)
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
            nn.ReLU(), # why relu? Who knows
            nn.BatchNorm2d(channels) # why batchnorm and not layer norm? Idk
        )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x


# segformer = LinSegFormer(
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
# print(segmentation.shape[2]) # torch.Size([1, 100, 56, 56])
# print(segmentation.size())

# r=4
# channels = 8
#
# x = torch.randn((1, channels, 64, 64))
# block = EfficientMultiHeadAttention(channels, reduction_ratio=r)
# print(block(x).shape)