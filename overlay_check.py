import cv2
import numpy as np
import os
from pathlib import Path

# 设置路径
data_dir = Path("data")
mask_dir = Path("mask_contour")
seg_mask_dir = Path("mask_seg")
overlay_dir = Path("overlay")
seg_overlay_dir = Path("seg_overlay")

# 创建输出目录
overlay_dir.mkdir(exist_ok=True)
seg_overlay_dir.mkdir(exist_ok=True)

# 颜色设置（BGR格式）
mask_color = (0, 0, 255)  # 红色 - 用于mask
seg_mask_color = (0, 255, 0)  # 绿色 - 用于seg_mask
alpha = 0.3  # 透明度

for folder_num in range(1, 113):
    folder_name = f"{folder_num:03d}"

    # 创建对应子文件夹
    (overlay_dir / folder_name).mkdir(exist_ok=True, parents=True)
    (seg_overlay_dir / folder_name).mkdir(exist_ok=True, parents=True)

    # 遍历每个子文件夹中的图片
    data_folder = data_dir / folder_name
    mask_folder = mask_dir / folder_name
    seg_mask_folder = seg_mask_dir / folder_name

    if not data_folder.exists():
        continue

    for img_file in data_folder.iterdir():
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # 读取原始图片
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            # 处理mask叠加
            mask_file = mask_folder / f"{img_file.stem}.png"
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # 创建彩色掩膜
                    colored_mask = np.zeros_like(img)
                    colored_mask[mask > 0] = mask_color
                    # 叠加图像
                    overlay = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)
                    # 保存结果
                    output_path = overlay_dir / folder_name / f"{img_file.stem}.png"
                    cv2.imwrite(str(output_path), overlay)

            # 处理seg_mask叠加
            seg_mask_file = seg_mask_folder / f"{img_file.stem}.png"
            if seg_mask_file.exists():
                seg_mask = cv2.imread(str(seg_mask_file), cv2.IMREAD_GRAYSCALE)
                if seg_mask is not None:
                    # 创建彩色掩膜
                    colored_seg_mask = np.zeros_like(img)
                    colored_seg_mask[seg_mask > 0] = seg_mask_color
                    # 叠加图像
                    seg_overlay = cv2.addWeighted(img, 1 - alpha, colored_seg_mask, alpha, 0)
                    # 保存结果
                    output_path = seg_overlay_dir / folder_name / f"{img_file.stem}.png"
                    cv2.imwrite(str(output_path), seg_overlay)

print("处理完成！")
print(f"overlay结果保存在: {overlay_dir}")
print(f"seg_overlay结果保存在: {seg_overlay_dir}")