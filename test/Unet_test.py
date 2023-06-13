# Import needs libraries
from models.created_classes.U_net.UNetModel import UNetModel
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

# Get train dataloader cityscapes

train_dataloader, val_dataloader, test_dataloader, train_step_viz_loss, val_step_viz_loss, test_step_viz_loss = get_dataloader_cityscapes(
    2)

loss_fn = get_cross_entropy_loss()

segmodel = UNetModel(input_channels=3,
                     output_classes=20,
                     )
segmodel.load_state_dict(torch.load(f'../models/checkpoints/Unet/300-600.pth'))
segmodel.to(device)

res_dict = test_step_v2(model=segmodel,
                        dataloader=val_dataloader,
                        loss_fn=loss_fn,
                        device=device)

print(f'Model Name: {res_dict["model_name"]}'
      f'IoU loss: {res_dict["model_IoU_loss"]}'
      f'Pixel accuracy: {res_dict["model_Pixel_loss"]}'
      f'F1 loss: {res_dict["model_F1_loss"]}')
