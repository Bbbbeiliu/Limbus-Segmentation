import os
import random
import numpy as np
from PIL import Image, ImageDraw
import glob
import argparse
import shutil


def load_obstacles(obstacle_dir):
    """加载所有障碍物图像"""
    obstacle_files = glob.glob(os.path.join(obstacle_dir, "*.png"))
    obstacles = []

    for file_path in obstacle_files:
        try:
            obstacle = Image.open(file_path).convert('RGBA')
            obstacles.append(obstacle)
        except Exception as e:
            print(f"加载障碍物 {file_path} 时出错: {e}")

    return obstacles


def random_transform_obstacle(obstacle, scale_range=(0.6, 1.4)):
    """对障碍物进行随机变换"""
    # 随机缩放
    scale = random.uniform(scale_range[0], scale_range[1])
    new_width = int(obstacle.width * scale)
    new_height = int(obstacle.height * scale)

    # 随机旋转
    angle = random.uniform(0, 360)

    # 随机拉伸
    stretch_x = random.uniform(0.8, 1.2)
    stretch_y = random.uniform(0.8, 1.2)
    stretch_width = int(new_width * stretch_x)
    stretch_height = int(new_height * stretch_y)

    # 应用变换
    transformed = obstacle.resize((stretch_width, stretch_height), Image.Resampling.LANCZOS)
    transformed = transformed.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    return transformed


def erase_contour_in_region(mask, obstacle, pos_x, pos_y):
    """使用障碍物的精确形状在标签上擦除轮廓"""
    # 创建标签副本
    mask_copy = mask.copy()

    # 获取障碍物的alpha通道（透明度）
    obstacle_alpha = obstacle.split()[3]

    # 将障碍物alpha通道转换为二值掩膜
    obstacle_mask = obstacle_alpha.point(lambda x: 255 if x > 0 else 0)

    # 创建一个与标签同样大小的临时图像
    temp_mask = Image.new('L', mask.size, 0)

    # 将障碍物掩膜粘贴到临时图像上
    temp_mask.paste(obstacle_mask, (pos_x, pos_y))

    # 使用障碍物掩膜在标签上擦除轮廓
    # 将障碍物区域的像素值设为0（黑色）
    mask_array = np.array(mask_copy)
    temp_array = np.array(temp_mask)

    # 只在障碍物区域（temp_array > 0）将标签设为0
    mask_array[temp_array > 0] = 0

    # 转换回PIL图像
    result = Image.fromarray(mask_array, mode='L')

    return result


def apply_obstacle_to_image_and_mask(image, mask, obstacle):
    """将障碍物应用到图像和标签上，返回增强后的图像和标签"""
    # 随机位置
    max_x = max(0, image.width - obstacle.width)
    max_y = max(0, image.height - obstacle.height)

    if max_x <= 0 or max_y <= 0:
        # 如果障碍物太大，调整大小
        scale = min(image.width / obstacle.width, image.height / obstacle.height)
        new_width = int(obstacle.width * scale * 0.8)
        new_height = int(obstacle.height * scale * 0.8)
        obstacle = obstacle.resize((new_width, new_height), Image.Resampling.LANCZOS)
        max_x = max(0, image.width - obstacle.width)
        max_y = max(0, image.height - obstacle.height)

    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)

    # 创建图像副本
    result_image = image.copy()

    # 粘贴障碍物到图像
    result_image.paste(obstacle, (pos_x, pos_y), obstacle)

    # 在标签上擦除障碍物区域的轮廓
    result_mask = erase_contour_in_region(mask, obstacle, pos_x, pos_y)

    return result_image, result_mask


def apply_dynamic_obstacle_augmentation(image, mask, obstacles, probability=0.1):
    """
    动态应用障碍物增强
    在训练过程中对单张图像进行实时增强

    Args:
        image: PIL Image (RGB)
        mask: PIL Image (L)
        obstacles: 障碍物图像列表
        probability: 应用增强的概率

    Returns:
        augmented_image: 增强后的图像
        augmented_mask: 增强后的标签
    """
    if random.random() > probability or not obstacles:
        return image, mask

    try:
        # 转换为RGBA以支持透明度
        image_rgba = image.convert('RGBA')

        # 随机选择一个障碍物并变换
        obstacle = random.choice(obstacles)
        transformed_obstacle = random_transform_obstacle(obstacle)

        # 应用障碍物到图像和标签
        augmented_image, augmented_mask = apply_obstacle_to_image_and_mask(
            image_rgba, mask, transformed_obstacle
        )

        # 转换回RGB
        augmented_image = augmented_image.convert('RGB')

        return augmented_image, augmented_mask

    except Exception as e:
        print(f"动态障碍物增强出错: {e}")
        return image, mask

def clear_previous_augmentations(data_dirs):
    """清除之前生成的增强文件"""
    for data_dir in data_dirs:
        # 清除数据目录中的增强图像
        for file in glob.glob(os.path.join(data_dir, "agument_*")):
            try:
                os.remove(file)
                print(f"已删除: {file}")
            except Exception as e:
                print(f"删除文件 {file} 时出错: {e}")

        # 清除对应的标签目录中的增强标签
        relative_path = os.path.relpath(data_dir, 'data')
        label_dir = os.path.join('mask', relative_path)
        if os.path.exists(label_dir):
            for file in glob.glob(os.path.join(label_dir, "agument_*")):
                try:
                    os.remove(file)
                    print(f"已删除: {file}")
                except Exception as e:
                    print(f"删除文件 {file} 时出错: {e}")


def augment_data(data_dirs, obstacle_dir, probability=0.2):
    """对数据进行增强"""
    obstacles = load_obstacles(obstacle_dir)

    if not obstacles:
        print("错误: 没有找到障碍物图像")
        return

    # 清除之前的增强文件
    print("清除之前的增强文件...")
    clear_previous_augmentations(data_dirs)

    for data_dir in data_dirs:
        if not os.path.exists(data_dir):
            print(f"警告: 目录不存在 {data_dir}")
            continue

        print(f"处理目录: {data_dir}")

        # 查找所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(data_dir, ext)))

        # 过滤掉已经增强的文件
        image_files = [f for f in image_files if not os.path.basename(f).startswith('agument_')]

        for image_path in image_files:
            # 10%概率应用增强
            if random.random() > probability:
                continue

            try:
                # 加载原图像
                original_image = Image.open(image_path).convert('RGBA')
                base_name = os.path.splitext(os.path.basename(image_path))[0]

                # 查找对应的标签文件
                relative_path = os.path.relpath(data_dir, 'data')
                label_dir = os.path.join('mask', relative_path)
                label_path = os.path.join(label_dir, base_name + '.png')

                if not os.path.exists(label_path):
                    print(f"警告: 未找到标签文件 {label_path}")
                    continue

                # 在增强循环中替换这部分代码：

                # 加载原图像和标签
                original_image = Image.open(image_path).convert('RGBA')
                label_image = Image.open(label_path).convert('L')

                # 随机选择一个障碍物并变换
                obstacle = random.choice(obstacles)
                transformed_obstacle = random_transform_obstacle(obstacle)

                # 同时应用到图像和标签
                augmented_image, augmented_label = apply_obstacle_to_image_and_mask(
                    original_image, label_image, transformed_obstacle
                )

                # 保存增强后的图像
                output_image_name = f"agument_{base_name}.jpg"
                output_image_path = os.path.join(data_dir, output_image_name)
                augmented_image.convert('RGB').save(output_image_path, 'JPEG')

                # 保存增强后的标签
                output_label_name = f"agument_{base_name}.png"
                output_label_path = os.path.join(label_dir, output_label_name)
                augmented_label.save(output_label_path, 'PNG')


            except Exception as e:
                print(f"处理 {image_path} 时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用障碍物进行数据增强')
    parser.add_argument('--data_dirs', type=str, nargs='+',
                        default=[f'data/{i:03d}' for i in range(50,51)],
                        # default=['data/050', 'data/051', 'data/052', 'data/053', 'data/054',
                        #          'data/055', 'data/056', 'data/057', 'data/058', 'data/059',
                        #          'data/060', 'data/061', 'data/062', 'data/063', 'data/064',
                        #          'data/065', 'data/066', 'data/067', 'data/068', 'data/069',
                        #          'data/070'],
                        # default=['data/073'],
                        help='数据目录列表')
    parser.add_argument('--obstacle_dir', type=str, default='obstacle/img',
                        help='障碍物图像目录')
    parser.add_argument('--probability', type=float, default=0,
                        help='应用增强的概率 (默认: 0.1)')

    args = parser.parse_args()

    augment_data(args.data_dirs, args.obstacle_dir, args.probability)
    print("数据增强完成!")