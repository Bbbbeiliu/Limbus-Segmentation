import os
import json
import numpy as np
from PIL import Image
import glob
import cv2


def png_to_labelme_json(png_path, original_image_path, output_json_path):
    """
    将PNG分割掩膜转换回Labelme JSON格式

    Args:
        png_path: PNG掩膜文件路径
        original_image_path: 原始图像路径
        output_json_path: 输出JSON文件路径
    """

    # 读取PNG图像
    mask = Image.open(png_path)

    # 获取图像尺寸
    width, height = mask.size

    # 获取原始图像文件名
    original_image_filename = os.path.basename(original_image_path)

    # 构建Labelme JSON结构
    json_data = {
        "version": "5.1.1",
        "flags": {},
        "shapes": [],
        "imagePath": original_image_filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

    mask_array = np.array(mask)
    # 处理调色板图像
    if mask.mode == 'P':
        # 获取调色板
        palette = mask.getpalette()

        # 查找红色对应的索引（R=128, G=0, B=0）
        red_index = None
        for i in range(0, len(palette), 3):
            r, g, b = palette[i], palette[i + 1], palette[i + 2]
            if r == 128 and g == 0 and b == 0:  # 暗红色
                red_index = i // 3
                break

        if red_index is not None:
            # 创建二值掩膜，红色区域为1，其他为0
            binary_mask = (mask_array == red_index).astype(np.uint8)

            # 使用OpenCV查找轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 简化轮廓点（减少点数）
                epsilon = 0.003 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 将轮廓点转换为Labelme格式
                points = []
                for point in approx:
                    x, y = point[0]
                    points.append([float(x), float(y)])

                # 只有当轮廓有足够多的点时才添加
                if len(points) >= 3:
                    shape_data = {
                        "label": "limbus",
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    json_data["shapes"].append(shape_data)

    # 处理RGB图像
    elif mask.mode == 'RGB':
        # 将图像转换为numpy数组
        mask_array = np.array(mask)

        # 查找红色像素 (R=128, G=0, B=0)
        red_pixels = (mask_array[:, :, 0] == 128) & (mask_array[:, :, 1] == 0) & (mask_array[:, :, 2] == 0)

        if np.any(red_pixels):
            # 创建二值掩膜
            binary_mask = red_pixels.astype(np.uint8)

            # 使用OpenCV查找轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # 简化轮廓点
                epsilon = 0.003 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 将轮廓点转换为Labelme格式
                points = []
                for point in approx:
                    x, y = point[0]
                    points.append([float(x), float(y)])

                if len(points) >= 3:
                    shape_data = {
                        "label": "limbus",
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    json_data["shapes"].append(shape_data)

    # 如果没有找到轮廓，创建一个空的形状
    if len(json_data["shapes"]) == 0:
        print(f"Warning: No contours found in {png_path}")

    # 保存JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return json_data


def process_mask_folders():
    """
    处理所有掩膜文件夹，将PNG转换为JSON并保存到原始图像文件夹
    """

    # 设置路径
    base_data_dir = "data"  # 原始图像根目录
    base_mask_dir = "mask"  # 掩膜根目录

    # 处理001-050文件夹
    for folder_num in range(5,8):
        folder_name = f"{folder_num:03d}"  # 001, 002, ..., 050

        # 构建路径
        mask_folder = os.path.join(base_mask_dir, folder_name)
        data_folder = os.path.join(base_data_dir, folder_name)

        # 检查文件夹是否存在
        if not os.path.exists(mask_folder):
            print(f"Mask folder {mask_folder} does not exist, skipping...")
            continue

        if not os.path.exists(data_folder):
            print(f"Data folder {data_folder} does not exist, skipping...")
            continue

        print(f"Processing folder: {folder_name}")

        # 查找所有PNG文件
        png_pattern = os.path.join(mask_folder, "*.png")
        png_files = glob.glob(png_pattern)

        print(f"Found {len(png_files)} PNG files in {folder_name}")

        for png_path in png_files:
            try:
                # 获取PNG文件名（不含扩展名）
                png_filename = os.path.basename(png_path)
                base_name = os.path.splitext(png_filename)[0]

                # 查找对应的原始图像文件（支持多种格式）
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                original_image_path = None

                for ext in image_extensions:
                    possible_path = os.path.join(data_folder, base_name + ext)
                    if os.path.exists(possible_path):
                        original_image_path = possible_path
                        break

                if original_image_path is None:
                    print(f"Warning: No original image found for {png_filename} in {data_folder}")
                    # 如果没有找到原始图像，使用PNG文件名作为图像路径
                    original_image_path = base_name + '.jpg'  # 假设为jpg格式

                # 生成输出JSON路径（保存在原始图像文件夹）
                output_json_path = os.path.join(data_folder, base_name + '.json')

                # 转换PNG到JSON
                png_to_labelme_json(png_path, original_image_path, output_json_path)
                print(f"Converted: {png_filename} -> {os.path.basename(output_json_path)}")

            except Exception as e:
                print(f"Error processing {png_path}: {str(e)}")

        print(f"Completed folder: {folder_name}\n")


def verify_conversion(folder_num, file_index=0):
    """
    验证转换结果

    Args:
        folder_num: 文件夹编号 (1-50)
        file_index: 文件索引
    """

    folder_name = f"{folder_num:03d}"

    # 构建路径
    mask_folder = os.path.join("mask", folder_name)
    data_folder = os.path.join("data", folder_name)

    # 查找PNG文件
    png_files = glob.glob(os.path.join(mask_folder, "*.png"))

    if file_index >= len(png_files):
        print(f"File index {file_index} out of range for folder {folder_name}")
        return

    png_path = png_files[file_index]
    png_filename = os.path.basename(png_path)
    base_name = os.path.splitext(png_filename)[0]

    # 查找对应的JSON文件
    json_path = os.path.join(data_folder, base_name + '.json')

    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 读取PNG文件
    mask = Image.open(png_path)

    print(f"Verification for {png_filename}:")
    print(f"  Image size: {mask.size}")
    print(f"  Number of shapes: {len(json_data['shapes'])}")

    if len(json_data['shapes']) > 0:
        print(f"  Points in first shape: {len(json_data['shapes'][0]['points'])}")
        print(f"  First few points: {json_data['shapes'][0]['points'][:3]}")

    print()


def main():
    """
    主函数
    """

    print("Starting PNG to JSON conversion...")

    # 处理所有文件夹
    process_mask_folders()

    print("All folders processed successfully!")

    # 可选：验证几个文件的转换结果
    print("\nVerifying conversion results...")

    # 验证第一个文件夹的第一个文件
    verify_conversion(1, 3)



if __name__ == "__main__":
    main()