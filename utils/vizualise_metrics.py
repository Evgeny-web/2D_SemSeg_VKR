import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

list_name_loss = ['IoU_loss.txt', 'Pixel_acc.txt', 'F1_loss.txt']

for name_loss in list_name_loss:
    metric_file = '../models/loss_metrics'
    list_models_iou = [f'SegFormer/{name_loss}', f'LinSegFormer/{name_loss}', f'LongSegFormer/{name_loss}',
                       f'LinAttUnet/{name_loss}', f'LongAttUnet/{name_loss}']

    print(name_loss.split('.')[0])

    list_epochs = []
    segformer_train = []
    segformer_test = []

    linseg_train = []
    linseg_test = []

    longseg_train = []
    longseg_test = []

    linatt_train = []
    linatt_test = []

    longatt_train = []
    longatt_test = []

    for model in list_models_iou:
        data_path = f'{metric_file}/{model}'
        with open(data_path, 'r') as file:
            results = file.readlines()

        index = list_models_iou.index(model)

        for epoch_data in results:
            values = epoch_data.split('-')
            if int(values[0]) == 301:
                break

            if int(values[0]) % 20 != 0:
                continue

            if index == 0:
                list_epochs.append(int(values[0].strip()))
                segformer_train.append(round(float(values[1].strip()), 5))
                segformer_test.append(round(float(values[2].strip()), 5))
            elif index == 1:
                linseg_train.append(round(float(values[1].strip()), 5))
                linseg_test.append(round(float(values[2].strip()), 5))
            elif index == 2:
                longseg_train.append(round(float(values[1].strip()), 5))
                longseg_test.append(round(float(values[2].strip()), 5))
            elif index == 3:
                linatt_train.append(round(float(values[1].strip()), 5))
                linatt_test.append(round(float(values[2].strip()), 5))
            elif index == 4:
                longatt_train.append(round(float(values[1].strip()), 5))
                longatt_test.append(round(float(values[2].strip()), 5))

    # print(len(list_epochs))
    # print(len(segformer_train))
    # print(len(linseg_train))

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    l0 = ax.plot(list_epochs, segformer_train, color='r', label='SegFormer_train', scaley=0.2)
    l1 = ax.plot(list_epochs, linseg_train, color='g', label='linseg_train')
    l2 = ax.plot(list_epochs, longseg_train, color='b', label='longseg_train')
    l3 = ax.plot(list_epochs, linatt_train, color='y', label='linatt_train')
    l4 = ax.plot(list_epochs, longatt_train, color='c', label='longatt_train')

    ax.set_title(name_loss.split('.')[0])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')

    ax.legend()

    plt.savefig(fname=f'../models/figure_out_loss/{name_loss.split(".")[0]}_train.png', format='png')

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    l0 = ax.plot(list_epochs, segformer_test, color='r', label='SegFormer_test', scaley=0.2)
    l1 = ax.plot(list_epochs, linseg_test, color='g', label='linseg_test')
    l2 = ax.plot(list_epochs, longseg_test, color='b', label='longseg_test')
    l3 = ax.plot(list_epochs, linatt_test, color='y', label='linatt_test')
    l4 = ax.plot(list_epochs, longatt_test, color='c', label='longatt_test')

    ax.set_title(name_loss.split('.')[0])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Value')

    ax.legend()

    plt.savefig(fname=f'../models/figure_out_loss/{name_loss.split(".")[0]}_test.png', format='png')
