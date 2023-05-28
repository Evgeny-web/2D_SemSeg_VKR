# Import needs libraries
from models.created_classes.Attention_U_net.Attention_UnetModel import *
from models.created_classes.U_net.UNetModel import *
from utils.cityscapes_dataloader import *

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

# Setup agnostic-code cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get train dataloader cityscapes

train_dataloader, val_dataloader, test_dataloader, train_step_viz_loss, val_step_viz_loss, test_step_viz_loss = get_dataloader_cityscapes(1)

from tqdm.auto import tqdm

list_name_loss = ['loss.txt', 'IoU_loss.txt', 'Pixel_acc.txt', 'F1_loss.txt']

# torch.manual_seed(42)
# Train UNet
epochs = 50
unet_train_list_loss = []
unet_val_list_loss = []
unet_list_epochs = []

# Training loop
for epoch in tqdm(range(epochs)):
    epoch += 1
    print(f"------------------\nEpoch: {epoch} from {epochs}\n------------------")
    unet_list_epochs.append(epoch)

    res_train_loss = train_step(model=UnetModel,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=unet_optimizer,
                                epoch=epoch)

    unet_train_list_loss.append(res_train_loss)

    res_val_loss = val_step(model=UnetModel,
                            dataloader=val_dataloader,
                            loss_fn=loss_fn,
                            epoch=epoch)

    unet_val_list_loss.append(res_val_loss)

    if epoch % 10 == 0:
        name = f"models/checkpoints/Unet/Unet-{epoch}-from-{epochs}.pth"
        checkpoint(UnetModel, name)

    for name_loss in list_name_loss:
        index = list_name_loss.index(name_loss)
        with open(name_loss, "a") as file:
            file.write(f"{epoch} - {res_train_loss[index]} - {res_val_loss[index]}")