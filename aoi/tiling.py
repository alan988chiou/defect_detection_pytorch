import os
import cv2
import numpy as np
from tqdm import tqdm
from aoi.io_u8 import imread_gray_u8, imwrite_gray_u8

def list_images(folder, exts=(".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")):
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(exts):
            files.append(fn)
    return sorted(files)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def tile_one(image, mask, patch, stride):
    H, W = image.shape[:2]
    for y in range(0, max(1, H - patch + 1), stride):
        for x in range(0, max(1, W - patch + 1), stride):
            yi = min(y, H - patch)
            xi = min(x, W - patch)
            im = image[yi:yi+patch, xi:xi+patch]
            mk = mask[yi:yi+patch, xi:xi+patch]
            yield xi, yi, im, mk

def has_defect(mask_patch, min_pos_pixels=10):
    return int((mask_patch > 0).sum()) >= min_pos_pixels

def dilate_mask(mask_u8, k=3, it=1):
    if it <= 0 or k <= 1:
        return mask_u8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask_u8, kernel, iterations=it)

def save_patch(out_img_path, out_msk_path, im, mk):
    imwrite_gray_u8(out_img_path, im)
    imwrite_gray_u8(out_msk_path, mk)

def build_patches(
    raw_img_dir, raw_msk_dir,
    out_root,
    split_ratios=(0.8, 0.1, 0.1),
    patch=384, stride=192,
    dilate_k=3, dilate_it=1,
    pos_oversample=3,
    min_pos_pixels=10,
    seed=42
):
    rng = np.random.default_rng(seed)

    img_files = list_images(raw_img_dir)
    assert len(img_files) > 0, f"No images found in: {raw_img_dir}"

    for fn in img_files:
        if not os.path.exists(os.path.join(raw_msk_dir, fn)):
            raise FileNotFoundError(f"Mask missing: {fn}")

    rng.shuffle(img_files)

    n = len(img_files)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])
    train_list = img_files[:n_train]
    val_list   = img_files[n_train:n_train+n_val]
    test_list  = img_files[n_train+n_val:]

    splits = {"train": train_list, "val": val_list, "test": test_list}

    for sp in splits:
        ensure_dir(os.path.join(out_root, sp, "images"))
        ensure_dir(os.path.join(out_root, sp, "masks"))

    for sp, files in splits.items():
        idx = 0
        for fn in tqdm(files, desc=f"tiling {sp}", total=len(files)):
            img_path = os.path.join(raw_img_dir, fn)
            msk_path = os.path.join(raw_msk_dir, fn)

            im = imread_gray_u8(img_path)
            mk = imread_gray_u8(msk_path)
            if im is None or mk is None:
                raise RuntimeError(f"Read failed: {fn}")
            if im.shape != mk.shape:
                raise RuntimeError(f"Shape mismatch: {fn} {im.shape} vs {mk.shape}")

            mk = (mk > 0).astype(np.uint8) * 255
            mk = dilate_mask(mk, k=dilate_k, it=dilate_it)

            for x, y, imp, mkp in tile_one(im, mk, patch=patch, stride=stride):
                pos = has_defect(mkp, min_pos_pixels=min_pos_pixels)
                rep = (pos_oversample if pos else 1)

                for r in range(rep):
                    out_name = f"{os.path.splitext(fn)[0]}_x{x}_y{y}_i{idx:06d}_r{r}.png"
                    out_img = os.path.join(out_root, sp, "images", out_name)
                    out_msk = os.path.join(out_root, sp, "masks",  out_name)
                    save_patch(out_img, out_msk, imp, mkp)
                    idx += 1
