import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import sigmoid_focal_loss

def LossFunc(pred, mask):
    mask=mask.float()
    bce = F.binary_cross_entropy(pred, mask, reduce=None)

    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)

    return (bce + aiou + mae).mean()

def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=True, use_l1_loss=True
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def compute_loss(predictions, label, stage_weights= [0.5, 0.5, 0.3, 0.3]):
    loss_fn = nn.CrossEntropyLoss()

    # Stage-wise classification losses
    stage_losses = [loss_fn(pred, label) for pred in predictions]

    # Weighted sum of stage losses
    total_loss = sum(w * loss for w, loss in zip(stage_weights, stage_losses))

    return total_loss