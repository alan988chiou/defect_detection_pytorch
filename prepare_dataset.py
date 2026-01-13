from aoi.tiling import build_patches

if __name__ == "__main__":
    build_patches(
        raw_img_dir="data_raw/images",
        raw_msk_dir="data_raw/masks",
        out_root="dataset",
        split_ratios=(0.6, 0.2, 0.2),
        patch=384,
        stride=192,          # overlap 推論一致
        dilate_k=3,
        dilate_it=1,         # 0 表示不膨脹
        pos_oversample=3,    # 正樣本重抽 3x（可調 2~6）
        min_pos_pixels=10
    )
