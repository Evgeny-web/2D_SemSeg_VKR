from torchvision import datasets
import torch
from torch import nn
import timm
from einops import rearrange

# Import needs libraries
from models.created_classes.LinAtt_Unet.LinAttention_Unet import LinAttentionUnetModel
from utils.cityscapes_dataloader import *
from utils.optimizers_loss_functions import *
from utils.checkpoints import checkpoint
from utils.train_loops import *
from utils.vizualise_results import vizualise_result_semseg

import os
from PIL import Image

import torch

if __name__ == '__main__':
    vizualise_result_semseg('LongAttUnetModel', 'val', 'cuda', 300)
