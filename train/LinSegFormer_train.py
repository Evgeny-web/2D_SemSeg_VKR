# Import needs libraries
from models.created_classes.LinSegFormer.LinSegFormer import LinSegFormer
from models.created_classes.LinSegFormer.NewLinSegFormer import NewLinSegFormer
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

train_dataloader, val_dataloader, test_dataloader, train_step_viz_loss, val_step_viz_loss, test_step_viz_loss = get_dataloader_cityscapes(4)

list_name_loss = ['loss.txt', 'IoU_loss.txt', 'Pixel_acc.txt', 'F1_loss.txt']
path_loss_metrics = '../models/loss_metrics/LinSegFormer'

segmodel = NewLinSegFormer(in_channels=3,
    widths=[64, 128, 256, 512],
    seq_len=[8192, 2048, 512, 128], # Only for NewLinSegFormer, another ignore this sentence
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=20).to(device)

optimizer = get_segformer_optimizer(segmodel)
loss_fn = get_cross_entropy_loss()

epochs = 600

# Training loop
for epoch in range(epochs):
    epoch += 1
    print(f"------------------\nEpoch: {epoch} from {epochs}\n------------------")
    start_time=time()

    res_train_loss = train_step_v2(model=segmodel,
                                   dataloader=train_dataloader,
                                   loss_fn=loss_fn,
                                   optimizer=optimizer,
                                   epoch=epoch,
                                   device=device,
                                   train_step_vizualise_loss=train_step_viz_loss,
                                   segformer=True)

    res_val_loss = val_step_v2(model=segmodel,
                               dataloader=val_dataloader,
                               loss_fn=loss_fn,
                               epoch=epoch,
                               device=device,
                               val_step_vizualise_loss=val_step_viz_loss,
                               segformer=True)

    end_time = time()
    total_time = end_time - start_time
    print(f'Время затраченное на одну эпоху: {total_time}')

    if epoch % 50 == 0:
        name = f"../models/checkpoints/LinSegFormer/{epoch}-from-{epochs}.pth"
        checkpoint(segmodel, name)

    for name_loss in list_name_loss:
        index = list_name_loss.index(name_loss)
        file_path_metrics = f'{path_loss_metrics}/{name_loss}'
        if epoch == 1:
            with open(file_path_metrics, "w") as file:
                file.write(f"{epoch} - {res_train_loss[index]} - {res_val_loss[index]}\n")
        else:
            with open(file_path_metrics, "a") as file:
                file.write(f"{epoch} - {res_train_loss[index]} - {res_val_loss[index]}\n")