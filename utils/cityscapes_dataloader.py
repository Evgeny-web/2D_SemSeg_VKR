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

import albumentations as A
from albumentations.pytorch import ToTensorV2

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.datasets import Cityscapes


data_path = "/media/evgeny/6610D40610D3DB5F/Download_Softwares/PyCharm/Projects/2D_SemSeg_VKR/Datasets/CityScapes"

ignore_index = 255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ["unlabled", "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
               "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus",
               "train", "motorcycle", "bicycle"]

class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes = len(valid_classes)

colors = [[0, 0, 0],
          [128, 64, 128],
          [244, 35, 232],
          [70, 70, 70],
          [102, 102, 156],
          [190, 153, 153],
          [153, 153, 153],
          [250, 170, 30],
          [220, 220, 0],
          [107, 142, 35],
          [152, 251, 152],
          [0, 130, 180],
          [220, 20, 60],
          [255, 0, 0],
          [0, 0, 142],
          [0, 0, 70],
          [0, 60, 100],
          [0, 80, 100],
          [0, 0, 230],
          [119, 11, 32],
          ]

label_colors = dict(zip(range(n_classes), colors))

transform_A = A.Compose(
    [
        A.Resize(256, 512),
        A.HorizontalFlip(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def resize_masks(masks, predict):
    new_masks = torch.zeros([masks.shape[0], predict.shape[1], predict.shape[2]])

    transform_mask = A.Compose(
        [
            A.Resize(predict.shape[1], predict.shape[2]),
        ])

    for i, im in enumerate(masks):
        np_targets = transform_mask(image=im.numpy())
        seg = torch.tensor(np_targets['image'])
        new_masks[i] = seg

    return new_masks

def encode_segmap(mask):
    """
    Переназначаем все нежелательные классы на ignore_index (255)
    а Желательные классы переопределяем на новые значения индексов
    """
    for _voidc in void_classes:
        mask[mask == _voidc] = ignore_index
    for _validc in valid_classes:
        mask[mask == _validc] = class_map[_validc]
    return mask


def decode_segmap(temp):
    # Преобразуем серое изображение в цветное RGB
    temp = temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()

    for i in range(0, n_classes):
        r[temp == i] = label_colors[i][0]
        g[temp == i] = label_colors[i][1]
        b[temp == i] = label_colors[i][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb


class MyClassCityscapes(Cityscapes):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed = transform_A(image=np.array(image), mask=np.array(target))

        return transformed['image'], transformed['mask']


def get_dataloader_cityscapes(batch_size: int = 2):
    """
    :param batch_size:
    :return: train_dataloader, val_dataloader, test_dataloader,
            ignore_index, valid_classes, n_classes, class_map, label_colors
    """

    def get_data_cityscapes_class(data_path: str, split: str, mode: str, target_type: str, transform,
                                  target_transform=None):
        datas = MyClassCityscapes(root=data_path,
                                  split=split,
                                  mode=mode,
                                  target_type=target_type,
                                  transforms=transform,
                                  target_transform=target_transform)

        return datas

    train_data = get_data_cityscapes_class(data_path, "train", 'fine', 'semantic', transform_A)
    val_data = get_data_cityscapes_class(data_path, "val", 'fine', 'semantic', transform_A)
    test_data = get_data_cityscapes_class(data_path, "test", 'fine', 'semantic', transform_A)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=True)

    train_step_vizualise_loss = len(train_dataloader) // 6
    val_step_vizualise_loss = len(val_dataloader) // 5
    test_step_vizualise_loss = len(test_dataloader) // 5

    return train_dataloader, val_dataloader, test_dataloader, \
        train_step_vizualise_loss, val_step_vizualise_loss, test_step_vizualise_loss
