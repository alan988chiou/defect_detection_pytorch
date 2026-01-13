# -*- coding: utf-8 -*-
"""
Split all images (including subfolders) into Bright field / Dark field halves (top/bottom),
based on which half is brighter (mean intensity), and save to flat output folders.

Requirements:
  pip install opencv-python numpy

Usage example (Windows / Chinese path supported):
  python split_bf_df.py --in "D:\資料\輸入" --out_bf "D:\輸出\BF" --out_df "D:\輸出\DF" --ext ".bmp,.png,.jpg,.jpeg,.tif,.tiff"
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path


# --------------------------
# Unicode-safe imread/imwrite
# --------------------------
def imread_unicode(path: str, flags=cv2.IMREAD_UNCHANGED):
    """Read image with Unicode path support on Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img

def imwrite_unicode(path: str, img, params=None):
    """Write image with Unicode path support on Windows."""
    if params is None:
        ok, buf = cv2.imencode(Path(path).suffix, img)
    else:
        ok, buf = cv2.imencode(Path(path).suffix, img, params)
    if not ok:
        return False
    buf.tofile(path)
    return True


def is_image_file(p: Path, exts_set):
    return p.is_file() and p.suffix.lower() in exts_set


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_unique_name(out_dir: Path, base_name: str, ext: str):
    """
    Flat output folder: avoid name collision by appending _0001, _0002...
    base_name is stem without extension; ext includes dot (e.g. .bmp)
    """
    candidate = out_dir / f"{base_name}{ext}"
    if not candidate.exists():
        return candidate

    i = 1
    while True:
        candidate = out_dir / f"{base_name}_{i:04d}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def to_gray_mean(img):
    """
    Compute mean intensity in a robust way for:
      - 8-bit/16-bit gray
      - BGR/BGRA
    """
    if img is None:
        return None

    if img.ndim == 2:  # grayscale
        g = img
    else:
        # Convert to gray from BGR/BGRA
        if img.shape[2] == 4:
            bgr = img[:, :, :3]
        else:
            bgr = img
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Use float mean; supports uint8/uint16
    return float(np.mean(g))


def to_gray_median(img, vmin=10, vmax=250):
    """
    Median of gray values within [vmin, vmax]
    (reject black mask & saturated glare pixels)
    """
    if img is None:
        return None

    if img.ndim == 2:
        g = img
    else:
        if img.shape[2] == 4:
            bgr = img[:, :, :3]
        else:
            bgr = img
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 篩選有效亮度區間
    m = (g >= vmin) & (g <= vmax)
    valid = g[m]

    if valid.size == 0:
        return None

    return float(np.median(valid))


def split_bf_df(img):
    """
    Split into top/bottom halves by height.
    Return (bright_half, dark_half, bright_is_top: bool)
    """
    h = img.shape[0]
    mid = h // 2

    top = img[:mid, ...]
    bottom = img[mid:, ...]

    mt = to_gray_median(top, 10, 240)
    mb = to_gray_median(bottom, 10, 240)

    if mt is None or mb is None:
        return None, None, None

    if mt >= mb:
        return top, bottom, True
    else:
        return bottom, top, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Input folder (recursive)")
    ap.add_argument("--out_bf", required=True, help="Output folder for Bright field images (flat)")
    ap.add_argument("--out_df", required=True, help="Output folder for Dark field images (flat)")
    ap.add_argument("--ext", default=".bmp,.png,.jpg,.jpeg,.tif,.tiff",
                    help="Comma-separated extensions to include (case-insensitive)")
    ap.add_argument("--keep_ext", action="store_true",
                    help="Keep original file extension. Default: save as original extension anyway.")
    ap.add_argument("--suffix", default="_BF,_DF",
                    help="Suffixes for BF/DF filenames, e.g. _BF,_DF")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_bf = Path(args.out_bf)
    out_df = Path(args.out_df)

    ensure_dir(out_bf)
    ensure_dir(out_df)

    exts_set = set([e.strip().lower() for e in args.ext.split(",") if e.strip()])
    bf_suffix, df_suffix = [s.strip() for s in args.suffix.split(",")]

    if not in_dir.exists():
        print(f"[ERROR] Input dir not found: {in_dir}")
        sys.exit(1)

    # Gather files recursively
    files = [p for p in in_dir.rglob("*") if is_image_file(p, exts_set)]
    if not files:
        print("[INFO] No images found.")
        return

    ok_cnt = 0
    fail_cnt = 0

    for p in files:
        try:
            img = imread_unicode(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("imread failed (None)")

            bright, dark, bright_is_top = split_bf_df(img)
            if bright is None:
                raise RuntimeError("split failed")

            # Flat output: base name from original stem only
            stem = p.stem
            ext = p.suffix  # keep original extension

            bf_name = make_unique_name(out_bf, stem + bf_suffix, ext)
            df_name = make_unique_name(out_df, stem + df_suffix, ext)

            if not imwrite_unicode(str(bf_name), bright):
                raise RuntimeError("imwrite BF failed")
            if not imwrite_unicode(str(df_name), dark):
                raise RuntimeError("imwrite DF failed")

            ok_cnt += 1
            # Optional: print mapping
            # print(f"[OK] {p} -> BF({ 'top' if bright_is_top else 'bottom' }) {bf_name.name}, DF {df_name.name}")

        except Exception as e:
            fail_cnt += 1
            print(f"[FAIL] {p} : {e}")

    print(f"[DONE] total={len(files)} ok={ok_cnt} fail={fail_cnt}")
    print(f"[OUT] BF: {out_bf}")
    print(f"[OUT] DF: {out_df}")


if __name__ == "__main__":
    main()
