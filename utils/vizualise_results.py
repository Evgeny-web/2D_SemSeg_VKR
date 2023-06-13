from time import time

import torch

from models.created_classes.U_net.UNetModel import UNetModel
from models.created_classes.Attention_U_net.Attention_UnetModel import AttentionUnetModel
from models.created_classes.SegFormer.SegFormer import SegFormer
from models.created_classes.SegFor_wtih_LinFormerAtt.NewLinSegFormer import NewLinSegFormer
from models.created_classes.SegForm_with_LongFormAtt.LongSegFormer import LongSegFormer
from models.created_classes.LinAtt_Unet.LinAttention_Unet import LinAttentionUnetModel
from models.created_classes.LongAttUnet.LongAttentionUnet import LongAttentionUnetModel

from utils.cityscapes_dataloader import *


def get_dataloader(name_dataloader: str = 'val'):
    train_dataloader, val_dataloader, test_dataloader, train_step_viz_loss, val_step_viz_loss, test_step_viz_loss = \
        get_dataloader_cityscapes(4)

    assert name_dataloader == 'train' or name_dataloader == 'val' or name_dataloader == 'test', \
        f'{name_dataloader} не соответствует ни одному из вариантов  -> train | val | test'

    if name_dataloader == 'train':
        return train_dataloader
    elif name_dataloader == 'val':
        return val_dataloader
    elif name_dataloader == 'test':
        return test_dataloader


def vizualise_result_semseg(name_model: str, dataloader: str, device: str, num_epoch: int):
    data = get_dataloader(dataloader)
    list_names = ['UNetModel', 'AttUnetModel', 'SegFormer', 'LinSegFormer', 'LongSegFormer', 'LinAttUnetModel',
                  'LongAttUnetModel']
    count_equals = 0
    for name in list_names:
        if name == name_model:
            count_equals += 1
    assert count_equals == 1, f'{name_model} неверно, оно не принадлежит не к одному из перечисленных в списке: {list_names}'

    if name_model == 'UNetModel':
        model = UNetModel(input_channels=3,
                          output_classes=n_classes,
                          up_sample_mode='bilinear',
                          up_sample_align_corner=False,
                          padding=1,
                          stride=1,
                          kernel=3,
                          number_conv_layers=2,
                          )

        model.load_state_dict(torch.load(f'../models/checkpoints/Unet/{num_epoch}-300.pth'))
        model.to(device)

    elif name_model == 'AttUnetModel':
        model = AttentionUnetModel(in_channels=3,
                                   out_channels=n_classes,
                                   mode='bilinear',
                                   align_corner=False,
                                   padding=1,
                                   stride=1,
                                   kernel=3,
                                   numberConvLayers=2,
                                   )

        model.load_state_dict(torch.load(f'../models/checkpoints/AttUnet/{num_epoch}-300.pth'))
        model.to(device)

    elif name_model == "SegFormer":
        model = SegFormer(in_channels=3,
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
        model.load_state_dict(torch.load(f'../models/checkpoints/SegFormer/SegFormer-{num_epoch}-600.pth'))
        model.to(device)

    elif name_model == "LinSegFormer":
        model = NewLinSegFormer(in_channels=3,
                                widths=[64, 128, 256, 512],
                                seq_len=[8192, 2048, 512, 128],
                                depths=[3, 4, 6, 3],
                                all_num_heads=[1, 2, 4, 8],
                                patch_sizes=[7, 3, 3, 3],
                                overlap_sizes=[4, 2, 2, 2],
                                reduction_ratios=[8, 4, 2, 1],
                                mlp_expansions=[4, 4, 4, 4],
                                decoder_channels=256,
                                scale_factors=[8, 4, 2, 1],
                                num_classes=20)
        model.load_state_dict(torch.load(f'../models/checkpoints/LinSegFormer/LinSegFormer-{num_epoch}-600.pth'))
        model.to(device)

    elif name_model == "LongSegFormer":
        model = LongSegFormer(in_channels=3,
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
        model.load_state_dict(torch.load(f'../models/checkpoints/LongSegFormer/LongSegFormer-{num_epoch}-600.pth'))
        model.to(device)

    elif name_model == "LinAttUnetModel":
        model = LinAttentionUnetModel(in_channels=3,
                                      out_channels=20,
                                      seq_len=2048  # Only for lin Attention Unet model
                                      )
        model.load_state_dict(torch.load(f'../models/checkpoints/LinAttUnet/{num_epoch}-600.pth'))
        model.to(device)

    elif name_model == 'LongAttUnetModel':
        model = LongAttentionUnetModel(in_channels=3,
                                       out_channels=20)
        model.load_state_dict(torch.load(f'../models/checkpoints/LongAttUnet/{num_epoch}-600.pth'))
        model.to(device)

    model.eval()

    with torch.inference_mode():
        for img, seg in data:
            img = img.to(device)
            model.to(device)

            out = model(img)

            break

    print(f'Shape logits models: {out.shape}')

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(10, 8))

    for i in range(4):
        sample = i

        outx = out.detach().cpu()[sample]

        encoded_mask = encode_segmap(seg[sample].clone())
        decoded_mask = decode_segmap(encoded_mask)

        decoded_outx = decode_segmap(torch.argmax(outx, dim=0))

        ax[i][0].imshow(np.moveaxis(img[sample].cpu().numpy(), 0, 2))
        ax[i][0].set_title('Sample image')
        ax[i][0].axis(False)

        ax[i][1].imshow(decoded_mask)
        ax[i][1].set_title('Truth mask')
        ax[i][1].axis(False)

        ax[i][2].imshow(decoded_outx)
        ax[i][2].set_title(f'Mask {name_model}')
        ax[i][2].axis(False)

    plt.savefig(fname=f'../models/figure_out_results/{name_model}-{num_epoch}.png', format="png")
