### Train Loop
import torch
from torch import nn

from utils.cityscapes_dataloader import encode_segmap, decode_segmap, resize_masks
from utils.optimizers_loss_functions import *

iou_loss_fn = iou_loss()
pixel_loss_fn = pixel_accuracy()
f1_loss_fn = f1_score()


def train_step(model, dataloader, loss_fn, optimizer, epoch, device, train_step_vizualise_loss, segformer: bool = False,
               seg_logits: bool = False):
    model.train()
    iou_loss, pixel_loss, f1_loss = 0.0, 0.0, 0.0
    train_loss = 0.0

    for batch, (img, seg) in enumerate(dataloader):
        # Put data on the target device
        img, seg = img.to(device), seg.to(device)

        # Output logits Model
        y_pred = model(img)
        # transform mask in true size for SegFormer
        if segformer:
            seg = resize_masks(seg, y_pred)

        # Encode mask
        seg = encode_segmap(seg.clone())
        # Calculate Loss
        if seg_logits:
            loss = loss_fn(y_pred.logits, seg.long())
            iou_loss += iou_loss_fn(y_pred.logits, seg)
            pixel_loss += pixel_loss_fn(y_pred.logits, seg)
            f1_loss = f1_loss_fn(y_pred.logits, seg)
        else:
            loss = loss_fn(y_pred, seg.long())
            iou_loss += iou_loss_fn(y_pred, seg)
            pixel_loss += pixel_loss_fn(y_pred, seg)
            f1_loss += f1_loss_fn(y_pred, seg)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % train_step_vizualise_loss == 0:
            print(f"Batch is {batch} Loss: {loss:.4f}")

    train_loss /= len(dataloader)
    iou_loss /= len(dataloader)
    pixel_loss /= len(dataloader)
    f1_loss /= len(dataloader)
    print(f"Epoch: {epoch} Train loss: {train_loss:.4f}")
    print(f"Epoch: {epoch} Train IoU loss: {iou_loss:.4f}")
    print(f"Epoch: {epoch} Train Pixel loss: {pixel_loss:.4f}")
    print(f"Epoch: {epoch} Train F1 loss: {f1_loss:.4f}")

    return [train_loss, iou_loss, pixel_loss, f1_loss]


def val_step(model, dataloader, loss_fn, epoch, device, val_step_vizualise_loss, segformer: bool = False,
             seg_logits: bool = False):
    model.eval()
    iou_loss, pixel_loss, f1_loss = 0.0, 0.0, 0.0
    val_loss = 0.0

    # Add a loop through the val batches
    with torch.inference_mode():
        for batch, (img, seg) in enumerate(dataloader):
            # Put data on the target device
            img, seg = img.to(device), seg.to(device)

            # Output logits ModelV1
            val_pred = model(img)
            # transform mask in true size for SegFormer
            if segformer:
                seg = resize_masks(seg, val_pred)

            # Encode mask
            seg = encode_segmap(seg.clone())
            # Calculate Loss
            if seg_logits:
                loss = loss_fn(val_pred.logits, seg.long())
                iou_loss += iou_loss_fn(val_pred.logits, seg)
                pixel_loss += pixel_loss_fn(val_pred.logits, seg)
                f1_loss = f1_loss_fn(val_pred.logits, seg)
            else:
                loss = loss_fn(val_pred, seg.long())
                iou_loss += iou_loss_fn(val_pred, seg)
                pixel_loss += pixel_loss_fn(val_pred, seg)
                f1_loss += f1_loss_fn(val_pred, seg)
            val_loss += loss

            if batch % val_step_vizualise_loss == 0:
                print(f"Batch is {batch} Val Loss: {loss:.4f}")

        val_loss /= len(dataloader)
        iou_loss /= len(dataloader)
        pixel_loss /= len(dataloader)
        f1_loss /= len(dataloader)
        print(f"Epoch: {epoch} Val loss: {val_loss:.4f}")
        print(f"Epoch: {epoch} Val IoU loss: {iou_loss:.4f}")
        print(f"Epoch: {epoch} Val Pixel loss: {pixel_loss:.4f}")
        print(f"Epoch: {epoch} Val F1 loss: {f1_loss:.4f}")

    return [val_loss, iou_loss, pixel_loss, f1_loss]


def test_step(model, dataloader, loss_fn, epoch, device, test_step_vizualise_loss, segformer: bool = False,
              seg_logits: bool = False):
    model.eval()

    iou_loss, pixel_loss, f1_loss = 0.0, 0.0, 0.0
    test_loss = 0.0
    test_list_loss, iou_list_loss, pixel_list_loss, f1_list_loss = [], [], [], []
    batches_list = []

    # Add a loop through the val batches
    with torch.inference_mode():
        for batch, (img, seg) in enumerate(dataloader):
            batches_list.append(batch)
            # Put data on the target device
            img, seg = img.to(device), seg.to(device)

            # Output logits ModelV1
            test_pred = model(img)
            # transform mask in true size for SegFormer
            if segformer:
                seg = resize_masks(seg, test_pred)
            # Encode mask
            seg = encode_segmap(seg.clone())

            # Calculate Loss
            if seg_logits:
                loss = loss_fn(test_pred.logits, seg.long())
                iou_loss += iou_loss_fn(test_pred.logits, seg)
                pixel_loss += pixel_loss_fn(test_pred.logits, seg)
                f1_loss = f1_loss_fn(test_pred.logits, seg)
            else:
                loss = loss_fn(test_pred, seg.long())
                iou_loss += iou_loss_fn(test_pred, seg)
                pixel_loss += pixel_loss_fn(test_pred, seg)
                f1_loss += f1_loss_fn(test_pred, seg)

            test_list_loss.append(loss)
            iou_list_loss.append(iou_loss)
            pixel_list_loss.append(pixel_loss)
            f1_list_loss.append(f1_loss)
            test_loss += loss

            if batch % test_step_vizualise_loss == 0:
                print(f"Batch is {batch} Test Loss: {loss:.4f}")

        test_loss /= len(dataloader)
        iou_loss /= len(dataloader)
        pixel_loss /= len(dataloader)
        f1_loss /= len(dataloader)
        print(f"Average test loss: {test_loss:.4f}")
        print(f"Epoch: {epoch} test IoU loss: {iou_loss:.4f}")
        print(f"Epoch: {epoch} test Pixel loss: {pixel_loss:.4f}")
        print(f"Epoch: {epoch} test F1 loss: {f1_loss:.4f}")

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_average_loss": test_loss,
            "model_list_loss": test_list_loss,
            "model_list_IoU_loss": iou_list_loss,
            "model_list_Pixel_loss": pixel_list_loss,
            "model_list_F1_loss": f1_list_loss,
            "model_list_batches": batches_list}
