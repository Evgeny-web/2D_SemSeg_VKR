from UNetModel import UNetModel

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchmetrics

from PIL import Image

if __name__ == '__main__':
    train_data = datasets.Cityscapes(root='Datasets/CityScapes/')

