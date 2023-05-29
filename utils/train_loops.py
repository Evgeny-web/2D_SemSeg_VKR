### Train Loop
import torch
from torch import nn

from utils.cityscapes_dataloader import encode_segmap, decode_segmap, resize_masks
from utils.optimizers_loss_functions import iou_metric, pixel_metric, f1_metric


def train_step(model, dataloader, loss_fn, optimizer, epoch, device, train_step_vizualise_loss, segformer: bool = False,
               seg_logits: bool = False):
    model.train()
    iou_loss_score, pixel_loss_score, f1_loss_score = 0.0, 0.0, 0.0
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
            iou_loss_score += iou_metric(y_pred.logits, seg)
            pixel_loss_score += pixel_metric(y_pred.logits, seg)
            f1_loss_score += f1_metric(y_pred.logits, seg)
        else:
            loss = loss_fn(y_pred.logits, seg.long())
            iou_loss_score += iou_metric(y_pred.logits, seg)
            pixel_loss_score += pixel_metric(y_pred.logits, seg)
            f1_loss_score += f1_metric(y_pred.logits, seg)
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % train_step_vizualise_loss == 0:
            print(f"Batch is {batch} Loss: {loss:.4f}")

    train_loss /= len(dataloader)
    iou_loss_score /= len(dataloader)
    pixel_loss_score /= len(dataloader)
    f1_loss_score /= len(dataloader)
    print(f"Epoch: {epoch} Train loss: {train_loss:.4f}")
    print(f"Epoch: {epoch} Train IoU loss: {iou_loss_score:.4f}")
    print(f"Epoch: {epoch} Train Pixel loss: {pixel_loss_score:.4f}")
    print(f"Epoch: {epoch} Train F1 loss: {f1_loss_score:.4f}")

    if device == 'cuda':
        return [train_loss.cpu().item(), iou_loss_score.cpu().item(), pixel_loss_score, f1_loss_score.cpu().item()]
    else:
        return [train_loss, iou_loss_score, pixel_loss_score, f1_loss_score]


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
                iou_loss += iou_metric(val_pred.logits, seg)
                pixel_loss += pixel_metric(val_pred.logits, seg)
                f1_loss += f1_metric(val_pred.logits, seg)
            else:
                loss = loss_fn(val_pred, seg.long())
                iou_loss += iou_metric(val_pred, seg)
                pixel_loss += pixel_metric(val_pred, seg)
                f1_loss += f1_metric(val_pred, seg)
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

    if device == 'cuda':
        return [val_loss.cpu().item(), iou_loss.cpu().item(), pixel_loss, f1_loss.cpu().item()]
    else:
        return [val_loss, iou_loss, pixel_loss, f1_loss]


def test_step(model, dataloader, loss_fn, device, segformer: bool = False,
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
                iou_loss = iou_metric(test_pred.logits, seg)
                pixel_loss = pixel_metric(test_pred.logits, seg)
                f1_loss = f1_metric(test_pred.logits, seg.unsqueeze(dim=1))
            else:
                loss = loss_fn(test_pred, seg.long())
                iou_loss = iou_metric(test_pred, seg)
                pixel_loss = pixel_metric(test_pred, seg)
                f1_loss = f1_metric(test_pred, seg.unsqueeze(dim=1))

            test_list_loss.append(loss.cpu().item())
            iou_list_loss.append(iou_loss.cpu().item())
            pixel_list_loss.append(pixel_loss)
            f1_list_loss.append(f1_loss.cpu().item())
            test_loss += loss

        test_loss /= len(dataloader)
        mean_iou_loss = sum(iou_list_loss) / len(dataloader)
        mean_pixel_loss = sum(pixel_list_loss) / len(dataloader)
        mean_f1_loss = sum(f1_list_loss) / len(dataloader)
        print(f"Average test loss: {test_loss:.4f}")
        print(f"Test IoU loss: {mean_iou_loss:.4f}")
        print(f"Test Pixel loss: {mean_pixel_loss:.4f}")
        print(f"Test F1 loss: {mean_f1_loss:.4f}")

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_average_loss": test_loss,
            "model_list_loss": test_list_loss,
            "model_list_IoU_loss": iou_list_loss,
            "model_list_Pixel_loss": pixel_list_loss,
            "model_list_F1_loss": f1_list_loss,
            "model_list_batches": batches_list}
