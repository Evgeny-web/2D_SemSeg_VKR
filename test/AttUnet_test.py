# Import needs libraries
from models.created_classes.Attention_U_net.Attention_UnetModel import AttentionUnetModel
from utils.cityscapes_dataloader import *
from utils.optimizers_loss_functions import *
from utils.checkpoints import checkpoint
from utils.train_loops import *

import os
from PIL import Image

import torch
from tqdm.auto import tqdm

# Setup agnostic-code cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device is {device}!')

# Get train dataloader cityscapes

train_dataloader, val_dataloader, test_dataloader, train_step_viz_loss, val_step_viz_loss, test_step_viz_loss = get_dataloader_cityscapes(
    2)

loss_fn = get_cross_entropy_loss()

segmodel = AttentionUnetModel(in_channels=3,
                              out_channels=20,
                              )
segmodel.load_state_dict(torch.load(f=f'../models/checkpoints/AttUnet/300-300.pth'))
segmodel.to(device)

res_dict = test_step_v2(model=segmodel,
                        dataloader=val_dataloader,
                        loss_fn=loss_fn,
                        device=device)

print(f'Model Name: {res_dict["model_name"]}\n'
      f'IoU loss: {round(res_dict["model_IoU_loss"], 4)}\n'
      f'Pixel accuracy: {round(res_dict["model_Pixel_loss"], 4)}\n'
      f'F1 loss: {round(res_dict["model_F1_loss"], 4)}\n')
