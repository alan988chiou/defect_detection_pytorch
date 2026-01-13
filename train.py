# train.py
import os
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from aoi.dataset import AoiSegDataset
from aoi.losses import combo_loss
from aoi.utils import ensure_dir, dice_iou_from_logits

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def make_model():
    # Your SMP version doesn't accept "convnext_base" directly.
    # Use timm prefix. If this still fails, switch to "resnet101".
    return smp.UnetPlusPlus(
        encoder_name="resnet34",  # e.g. mobilenet_v2, resnet50, efficientnet-b7
        encoder_weights="imagenet",   # set to "imagenet" if your env supports it
        in_channels=1,
        classes=1,
        activation=None,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    dataset_root = r"dataset"
    run_dir = os.path.join("runs", time.strftime("%Y%m%d_%H%M%S"))
    ensure_dir(run_dir)

    train_ds = AoiSegDataset(dataset_root, split="train", augment=True)
    val_ds = AoiSegDataset(dataset_root, split="val", augment=False)

    # RTX 4080 Laptop: start from 12 for 384 patches; if OOM -> 8 or 6
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True
    )

    model = make_model().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    epochs = 30
    thr = 0.5
    best_val_dice = -1.0

    # (Optional) sanity print once
    # ims0, mks0 = next(iter(train_loader))
    # print("batch:", ims0.shape, ims0.dtype, mks0.shape, mks0.dtype, mks0.min().item(), mks0.max().item())

    for ep in range(1, epochs + 1):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        tr_losses = []

        pbar = tqdm(train_loader, desc=f"Train {ep}/{epochs}", ncols=110)
        for ims, mks in pbar:
            # DataLoader already gives Tensor -> do NOT torch.from_numpy
            ims = ims.to(device, non_blocking=True)
            mks = mks.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(ims)
                loss = combo_loss(logits, mks, dice_w=0.7, focal_w=0.3, gamma=2.5)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # -------------------------
        # Val
        # -------------------------
        model.eval()
        val_losses, val_dices, val_ious = [], [], []

        pbarv = tqdm(val_loader, desc=f"Val   {ep}/{epochs}", ncols=110)
        with torch.no_grad():
            for ims, mks in pbarv:
                ims = ims.to(device, non_blocking=True)
                mks = mks.to(device, non_blocking=True)

                logits = model(ims)
                loss = combo_loss(logits, mks, dice_w=0.7, focal_w=0.3, gamma=2.5)
                d, i = dice_iou_from_logits(logits, mks, thr=thr)

                val_losses.append(loss.item())
                val_dices.append(d)
                val_ious.append(i)

                pbarv.set_postfix(val_loss=f"{loss.item():.4f}", dice=f"{d:.4f}", iou=f"{i:.4f}")

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss = float(np.mean(val_losses)) if val_losses else 0.0
        va_dice = float(np.mean(val_dices)) if val_dices else 0.0
        va_iou = float(np.mean(val_ious)) if val_ious else 0.0

        print(
            f"[{ep:02d}/{epochs}] "
            f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_dice={va_dice:.4f}  val_iou={va_iou:.4f}"
        )

        # -------------------------
        # Save best
        # -------------------------
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_val_dice": best_val_dice,
                "thr": thr,
            }
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            print(f"  -> saved best.pt (val_dice={best_val_dice:.4f})")

    torch.save({"model": model.state_dict(), "thr": thr}, os.path.join(run_dir, "last.pt"))
    print("done:", run_dir)


if __name__ == "__main__":
    main()
