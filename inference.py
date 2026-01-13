# infer_big_image.py  (Chinese path safe + average time per image)
import os
import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp

# -------------------------
# Chinese-path-safe IO
# -------------------------
def imread_gray_u8(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img

def imwrite_u8(path: str, img):
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".png"
        path = path + ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"imencode failed: {path}")
    buf.tofile(path)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(folder, exts=(".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")):
    out = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(exts):
            out.append(fn)
    return sorted(out)

# -------------------------
# Model
# -------------------------
def make_model(encoder_name="resnet34"):
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None,
    )

# -------------------------
# Tiled inference
# -------------------------
def hann2d(h, w):
    wy = np.hanning(h)
    wx = np.hanning(w)
    w2 = np.outer(wy, wx).astype(np.float32)
    return np.maximum(w2, 1e-3)

@torch.no_grad()
def infer_big(img_u8, model, patch=320, stride=160, device="cuda"):
    H, W = img_u8.shape
    img = img_u8.astype(np.float32) / 255.0

    weight = hann2d(patch, patch)
    acc = np.zeros((H, W), np.float32)
    wacc = np.zeros((H, W), np.float32)

    for y in range(0, max(1, H - patch + 1), stride):
        for x in range(0, max(1, W - patch + 1), stride):
            yi = min(y, H - patch)
            xi = min(x, W - patch)

            tile = img[yi:yi+patch, xi:xi+patch]
            inp = torch.from_numpy(tile[None, None, ...]).to(device)

            logits = model(inp)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

            acc[yi:yi+patch, xi:xi+patch] += prob * weight
            wacc[yi:yi+patch, xi:xi+patch] += weight

    prob_map = acc / np.maximum(wacc, 1e-6)
    return prob_map

# -------------------------
# Postprocess + overlay
# -------------------------
def postprocess(bin_mask_u8, min_area=30):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask_u8, connectivity=8)
    out = np.zeros_like(bin_mask_u8)
    defects = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out[labels == i] = 255
        defects.append({"id": int(i), "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})
    return out, defects

def overlay(img_u8, mask_u8):
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    red = bgr.copy()
    red[:, :, 2] = np.clip(red[:, :, 2].astype(np.int16) + 120, 0, 255).astype(np.uint8)
    m = (mask_u8 > 0)
    bgr[m] = (0.35 * bgr[m] + 0.65 * red[m]).astype(np.uint8)
    return bgr

# -------------------------
# Main
# -------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: set your run dir
    run_dir = r"runs\20260113_100902"
    ckpt_path = os.path.join(run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # IMPORTANT: must match training encoder
    # If you didn't save it in checkpoint, set it manually here to match training.
    encoder_name = ckpt.get("encoder_name", "resnet34")

    model = make_model(encoder_name=encoder_name)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    thr = float(ckpt.get("thr", 0.5))

    in_dir = r"data_raw\images"
    out_dir = os.path.join(run_dir, "infer")
    ensure_dir(out_dir)

    files = list_images(in_dir)
    all_rows = []
    times = []

    for fn in tqdm(files, desc="infer big", ncols=110):
        ip = os.path.join(in_dir, fn)
        img = imread_gray_u8(ip)
        if img is None:
            print("read failed:", ip)
            continue

        # timing (GPU needs synchronize)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        prob = infer_big(img, model, patch=320, stride=160, device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)

        binm = (prob >= thr).astype(np.uint8) * 255
        binm2, defects = postprocess(binm, min_area=30)

        ov = overlay(img, binm2)

        base = os.path.splitext(fn)[0]
        imwrite_u8(os.path.join(out_dir, f"{base}_prob.png"), (prob * 255).astype(np.uint8))
        imwrite_u8(os.path.join(out_dir, f"{base}_mask.png"), binm2)
        imwrite_u8(os.path.join(out_dir, f"{base}_overlay.png"), ov)

        for d in defects:
            d["file"] = fn
            all_rows.append(d)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            os.path.join(out_dir, "defects.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    if times:
        avg = sum(times) / len(times)
        print(f"Average inference time per image: {avg:.3f} sec ({1.0/avg:.2f} img/s)")

    print("done:", out_dir)

if __name__ == "__main__":
    main()
