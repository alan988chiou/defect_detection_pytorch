import torch
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2,3))
    den = (probs + targets).sum(dim=(2,3)) + eps
    loss = 1 - (num + eps) / den
    return loss.mean()

def focal_loss(logits, targets, gamma=2.5):
    # binary focal on logits
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    w = (1 - pt).pow(gamma)
    return (w * bce).mean()

def combo_loss(logits, targets, dice_w=0.7, focal_w=0.3, gamma=2.5):
    return dice_w * dice_loss(logits, targets) + focal_w * focal_loss(logits, targets, gamma=gamma)
