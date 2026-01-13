import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A


def imread_gray_u8(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


def list_pairs(img_dir, msk_dir):
    fns = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"))])
    pairs = []
    for f in fns:
        ip = os.path.join(img_dir, f)
        mp = os.path.join(msk_dir, f)
        if os.path.exists(mp):
            pairs.append((ip, mp))
    return pairs


class AoiSegDataset(Dataset):
    def __init__(self, root, split="train", augment=True):
        self.img_dir = os.path.join(root, split, "images")
        self.msk_dir = os.path.join(root, split, "masks")
        self.pairs = list_pairs(self.img_dir, self.msk_dir)
        assert len(self.pairs) > 0, f"No data in {self.img_dir}"

        self.augment = augment and (split == "train")

        self.tf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(p=0.15),
        ]) if self.augment else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]

        im = imread_gray_u8(ip)
        mk = imread_gray_u8(mp)
        if im is None or mk is None:
            raise RuntimeError(f"Read failed: {ip}")

        im = im.astype(np.float32) / 255.0
        mk = (mk > 0).astype(np.float32)

        if self.tf is not None:
            out = self.tf(image=im, mask=mk)
            im, mk = out["image"], out["mask"]

        # to torch tensor, NCHW = 1xHxW
        im = torch.from_numpy(np.expand_dims(im, 0)).float()
        mk = torch.from_numpy(np.expand_dims(mk, 0)).float()

        return im, mk
