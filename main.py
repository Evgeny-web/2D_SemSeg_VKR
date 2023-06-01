from torchvision import datasets
import torch
from torch import nn

if __name__ == '__main__':
    # NLP Example
    batch, sentence_length, embedding_dim = 20, 5, 10
    embedding = torch.randn(batch, sentence_length, embedding_dim)
    layer_norm = nn.LayerNorm(embedding_dim)
    # Activate module
    print(layer_norm(embedding).shape)
    # Image Example
    N, C, H, W = 20, 5, 10, 10
    input = torch.randn(N, C, H, W)
    # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
    # as shown in the image below
    layer_norm = nn.LayerNorm([C, H, W])
    output = layer_norm(input)
    print(output.shape)
