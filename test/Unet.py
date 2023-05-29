# Import needs libraries
from models.created_classes.Attention_U_net.Attention_UnetModel import *
from models.created_classes.U_net.UNetModel import *
from utils.cityscapes_dataloader import *
from utils.optimizers_loss_functions import *
from utils.train_loops import *

import os
from PIL import Image

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchmetrics

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import requests
from tqdm.auto import tqdm

# Setup agnostic-code cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get train dataloader cityscapes

train_dataloader, val_dataloader, test_dataloader, train_step_viz_loss, \
    val_step_viz_loss, test_step_viz_loss = get_dataloader_cityscapes(2)

list_name_loss = ['loss.txt', 'IoU_loss.txt', 'Pixel_acc.txt', 'F1_loss.txt']

# Create instance Unet Model
Unet_model = UNetModel(input_channels=3,
                        output_classes=n_classes,
                        up_sample_mode='bilinear',
                        up_sample_align_corner=False,
                        padding=1,
                        stride=1,
                        kernel=3,
                        number_conv_layers=2,
                        )

# Load saved state dict()
Unet_model.load_state_dict(torch.load(f='models/checkpoints/Unet/300-300.pth'))
Unet_model.to(device)

result_dict = test_step(model=Unet_model, dataloader=test_dataloader, loss_fn=nn.CrossEntropyLoss(),
                        device=device)

