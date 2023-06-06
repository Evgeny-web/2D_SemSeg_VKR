import torch
from torch import nn
from utils.cityscapes_dataloader import decode_segmap
from time import time
from torchmetrics import F1Score, JaccardIndex


def get_unet_optimizer(model):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00004)
    return optimizer


def get_attunet_optimizer(model):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.00004)
    return optimizer


def get_segformer_optimizer(model):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00006, weight_decay=0.01)
    return optimizer


def get_cross_entropy_loss():
    return nn.CrossEntropyLoss()


def iou_metric(logits, targets, device):
    # Jaccard loss
    # start_time = time()
    logits = torch.round(logits)
    num_classes = logits.size(1)

    iou = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)
    result = iou(logits, targets)

    # end_time = time()
    # IoU_time = end_time - start_time
    # print(f'Время затраченное для подсчета IoU метрики:{IoU_time}')

    return round(result.item(), 5)


def pixel_metric(logits, target):
    accuracy = 0.0
    # start_time = time()
    for index in range(len(logits)):
        dec_logits = decode_segmap(torch.argmax(logits[index].cpu(), dim=0))
        dec_target = decode_segmap(target[index].cpu())
        correct_pixels = (dec_logits == dec_target).sum().item()
        total_pixels = dec_target.shape[0] * dec_target.shape[1]
        accuracy += correct_pixels / total_pixels

    accuracy /= len(logits)
    # end_time = time()
    # Pixel_time = end_time - start_time
    # print(f'Время затраченное для подсчета Pixel метрики:{Pixel_time}')

    return round(accuracy, 5)


def f1_metric(logits, target, device):
    # start_time = time()
    # Преобразование предсказанных меток в бинарный формат
    logits = torch.round(logits)
    num_classes = logits.size(1)

    F1 = F1Score(task='multiclass', num_classes=num_classes).to(device)

    result = F1(logits, target)

    # end_time = time()
    # f1_time = end_time - start_time
    # print(f'Время затраченное для подсчета f1 метрики:{f1_time}')

    return round(result.item(), 5)

# x1 = torch.randint(size=(4, 20, 512, 512), high=19, dtype=torch.float)
# x2 = torch.randint(size=(4, 512, 512), high=19, dtype=torch.float)
#
# i = f1_metric(x1, x2)
# print(i)
