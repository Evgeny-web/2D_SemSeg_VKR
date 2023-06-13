# Import needs libraries
from models.created_classes.SegForm_with_LongFormAtt.LongSegFormer import LongSegFormer
from utils.cityscapes_dataloader import *
from utils.optimizers_loss_functions import *
from utils.checkpoints import checkpoint
from utils.train_loops import *

import os
from PIL import Image

import torch

# Setup agnostic-code cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device is {device}!')

# Get train dataloader cityscapes

train_dataloader, val_dataloader, test_dataloader, train_step_viz_loss, val_step_viz_loss, test_step_viz_loss = get_dataloader_cityscapes(2)

loss_fn = get_cross_entropy_loss()

segmodel = LongSegFormer(in_channels=3,
                         widths=[64, 128, 256, 512],
                         depths=[3, 4, 6, 3],
                         all_num_heads=[1, 2, 4, 8],
                         patch_sizes=[7, 3, 3, 3],
                         overlap_sizes=[4, 2, 2, 2],
                         reduction_ratios=[8, 4, 2, 1],
                         mlp_expansions=[4, 4, 4, 4],
                         decoder_channels=256,
                         scale_factors=[8, 4, 2, 1],
                         num_classes=20)
segmodel.load_state_dict(torch.load(f'../models/checkpoints/LongSegFormer/300-600.pth'))
segmodel.to(device)

res_dict = test_step_v2(model=segmodel,
                        dataloader=val_dataloader,
                        loss_fn=loss_fn,
                        device=device)

print(f'Model Name: {res_dict["model_name"]}'
      f'IoU loss: {res_dict["model_IoU_loss"]}'
      f'Pixel accuracy: {res_dict["model_Pixel_loss"]}'
      f'F1 loss: {res_dict["model_F1_loss"]}')
