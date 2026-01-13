import os, json, cv2
import numpy as np
from tqdm import tqdm

IMG_DIR = r"raw\images_bf"
LBL_DIR = r"raw\labels_bf"

OUT_IMG_DIR = r"dataset\images"
OUT_MSK_DIR = r"dataset\masks"

DILATE_K = 3
DILATE_IT = 1

def ensure(p):
    os.makedirs(p, exist_ok=True)

def imread_u8(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

def imwrite_u8(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("imencode failed:", path)
    buf.tofile(path)

def main():
    ensure(OUT_IMG_DIR)
    ensure(OUT_MSK_DIR)

    imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".bmp",".png",".jpg",".tif",".tiff"))]

    for fn in tqdm(imgs):
        base = os.path.splitext(fn)[0]
        jp = os.path.join(LBL_DIR, base + ".json")
        if not os.path.exists(jp):
            print("missing json:", fn)
            continue

        with open(jp, "r", encoding="utf-8") as f:
            js = json.load(f)

        ip = os.path.join(IMG_DIR, fn)
        img = imread_u8(ip)
        if img is None:
            print("read failed:", ip)
            continue

        H, W = img.shape
        mask = np.zeros((H, W), np.uint8)

        for shp in js["shapes"]:
            pts = np.array(shp["points"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

        if DILATE_IT > 0 and DILATE_K > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_K, DILATE_K))
            mask = cv2.dilate(mask, k, iterations=DILATE_IT)

        imwrite_u8(os.path.join(OUT_IMG_DIR, fn), img)
        imwrite_u8(os.path.join(OUT_MSK_DIR, fn), mask)

    print("done")

if __name__ == "__main__":
    main()
