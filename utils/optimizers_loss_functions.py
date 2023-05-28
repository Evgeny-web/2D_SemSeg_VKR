import torch
from torch import nn


def get_unet_optimizer(model):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.00004)
    return optimizer

def get_attunet_optimizer(model):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.00004)
    return optimizer

def get_segformer_optimizer(model):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.00006, weight_decay=0.01)
    return optimizer


CrossEntropyloss_fn = nn.CrossEntropyLoss()

def iou_loss(logits, targets):
    # Jaccard loss
    smooth = 1e-5
    num_classes = logits.size(1)
    loss = 0.0

    for i in range(num_classes):
        iflat = logits[:, i].contiguous().view(-1)
        tflat = targets[:].contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        union = (iflat.sum() + tflat.sum()) - intersection
        loss += (intersection + smooth) / (union + smooth)

    result = 1.0 - (loss / num_classes)

    return result

def pixel_accuracy(logits, target):
    _, predicted = torch.max(logits, 1)
    correct_pixels = (predicted == target).sum().item()
    total_pixels = target.numel()
    accuracy = correct_pixels / total_pixels

    return accuracy


def f1_score(logits, target):
    # Преобразование предсказанных меток в бинарный формат
    logits = torch.round(logits)

    # Вычисление TP, FP и FN
    tp = torch.sum(target * logits, dim=(1, 2))
    fp = torch.sum(logits, dim=(1, 2)) - tp
    fn = torch.sum(target, dim=(1, 2)) - tp

    # Вычисление precision, recall и F1-score
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    # Усреднение F1-score по классам
    f1_macro = torch.mean(f1)

    return f1_macro