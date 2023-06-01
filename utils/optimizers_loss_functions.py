import torch
from torch import nn
from utils.cityscapes_dataloader import decode_segmap


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


def iou_metric(logits, targets):
    # Jaccard loss
    smooth = 1e-5
    num_classes = logits.size(1)
    result = 0.0

    for index in range(len(logits)):
        loss = 0.0
        for i in range(num_classes):
            intersection = 0
            iflat = logits[index, i].contiguous().view(-1)
            tflat = targets[index].contiguous().view(-1)

            for j in range(len(tflat)):
                if iflat[j] == i and tflat[j] == i:
                    intersection += 1

            union_i = (iflat == i).sum()
            union_t = (tflat == i).sum()
            union = union_t + union_i - intersection

            loss += (intersection + smooth) / (union + smooth)

        result += loss / num_classes

    result /= len(logits)

    return result


def pixel_metric(logits, target):
    accuracy = 0.0
    for index in range(len(logits)):
        dec_logits = decode_segmap(torch.argmax(logits[index].cpu(), dim=0))
        dec_target = decode_segmap(target[index].cpu())
        correct_pixels = (dec_logits == dec_target).sum().item()
        total_pixels = dec_target.shape[0] * dec_target.shape[1]
        accuracy += correct_pixels / total_pixels

    accuracy /= len(logits)

    return round(accuracy, 2)


def f1_metric(logits, target):
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


# x1 = torch.tensor(data=[[[[1, 1, 1, 1],
#                           [1, 1, 1, 1],
#                           [0, 0, 0, 0],
#                           [0, 0, 0, 0]],
#                          [[0, 0, 0, 0],
#                           [1, 1, 1, 1],
#                           [0, 0, 0, 0],
#                           [1, 1, 1, 1]]
#                          ]],device='cuda')
#
# x2 = torch.tensor(data=[[[1, 1, 1, 1],
#                          [1, 1, 1, 1],
#                          [0, 0, 0, 0],
#                          [0, 0, 0, 0]]
#                         ],device='cuda')
# print(x1.shape)
# print(x2.shape)
# print(f'x2[0]: {x2[0]}')
# print(f'x1[0]: {x1[0]}')
#
# res = iou_metric(x1, x2)
# print(f'res: {res}')
