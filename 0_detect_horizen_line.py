import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageStitchingDetector:
    def __init__(self, gradient_threshold=30, region_height_ratio=0.0315):
        self.gradient_threshold = gradient_threshold
        self.region_height_ratio = region_height_ratio

    def detect_stitching_line(self, image_path):
        """检测图像中间位置的拼接分界线"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return False, 0, "无法读取图片"

            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            # 计算中间区域的范围
            region_height = int(height * self.region_height_ratio)
            start_row = height // 2 - region_height // 2
            end_row = height // 2 + region_height // 2

            # 提取中间区域
            center_region = gray[start_row:end_row, :]

            # 计算垂直方向的梯度（关注从深到浅的变化）
            # 使用Sobel算子计算垂直梯度
            sobel_y = cv2.Sobel(center_region, cv2.CV_64F, 0, 1, ksize=3)

            # 取绝对值，关注从深到浅和从浅到深的变化
            sobel_y_abs = np.abs(sobel_y)

            # 按行计算平均梯度
            row_gradients = np.mean(sobel_y_abs, axis=1)

            # 找到梯度最大的行
            max_gradient_idx = np.argmax(row_gradients)
            max_gradient = row_gradients[max_gradient_idx]

            # 计算梯度阈值（使用自适应阈值）
            adaptive_threshold = np.mean(row_gradients) + 2 * np.std(row_gradients)

            # 判断是否存在分界线
            has_stitching = max_gradient > max(self.gradient_threshold, adaptive_threshold)

            if has_stitching:
                # 计算分界线在原图中的位置
                stitching_line = start_row + max_gradient_idx
                return True, stitching_line, f"检测到强梯度: {max_gradient:.2f} > {max(self.gradient_threshold, adaptive_threshold):.2f}"
            else:
                return False, 0, f"未检测到强梯度: {max_gradient:.2f} <= {max(self.gradient_threshold, adaptive_threshold):.2f}"

        except Exception as e:
            logger.error(f"检测过程中出错 {image_path}: {e}")
            return False, 0, f"检测错误: {str(e)}"

    def create_visualization(self, image_path, stitching_line):
        """创建可视化结果，不修改原图"""
        try:
            # 读取原图
            original_image = cv2.imread(image_path)
            if original_image is None:
                return

            # 创建副本用于可视化
            vis_image = original_image.copy()
            height, width = vis_image.shape[:2]

            # 标记分界线
            cv2.line(vis_image, (0, stitching_line), (width, stitching_line), (0, 0, 255), 3)
            cv2.putText(vis_image, "Stitching Line", (10, stitching_line - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 标记检测区域
            region_height = int(height * self.region_height_ratio)
            start_row = height // 2 - region_height // 2
            end_row = height // 2 + region_height // 2
            cv2.rectangle(vis_image, (0, start_row), (width, end_row), (255, 0, 0), 2)
            cv2.putText(vis_image, "Detection Region", (10, start_row - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 保存可视化结果
            output_path = Path(image_path).parent / f"visualization_{Path(image_path).name}"
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"可视化结果已保存: {output_path}")

        except Exception as e:
            logger.error(f"创建可视化错误: {e}")


def process_images(source_folder, target_folder, mask_base_folder, wrong_mask_folder,
                   folder_name, move_files=True, visualize=False, gradient_threshold=30):
    """
    处理指定文件夹中的所有JPG图片

    参数:
        source_folder: 源文件夹路径
        target_folder: 目标文件夹路径
        mask_base_folder: mask基础文件夹路径
        wrong_mask_folder: wrong mask文件夹路径
        folder_name: 当前处理的子文件夹名称
        move_files: 是否移动文件(True)或仅复制(False)
        visualize: 是否为检测到的图片生成可视化结果
        gradient_threshold: 梯度阈值，值越小检测越敏感
    """
    try:
        # 创建目标文件夹
        os.makedirs(target_folder, exist_ok=True)
        os.makedirs(wrong_mask_folder, exist_ok=True)

        # 初始化检测器
        detector = ImageStitchingDetector(gradient_threshold=gradient_threshold)

        # 获取所有JPG文件
        jpg_files = list(Path(source_folder).glob("*.jpg"))
        jpg_files.extend(Path(source_folder).glob("*.jpeg"))
        jpg_files.extend(Path(source_folder).glob("*.JPG"))

        if not jpg_files:
            logger.warning(f"在文件夹 {source_folder} 中未找到JPG图片")
            return

        logger.info(f"找到 {len(jpg_files)} 个JPG文件，开始检测...")

        detected_files = []

        for jpg_path in jpg_files:
            try:
                # 检测图片
                has_stitching, stitching_line, reason = detector.detect_stitching_line(str(jpg_path))

                if has_stitching:
                    detected_files.append((jpg_path, stitching_line, reason))
                    logger.info(f"检测到拼接图片: {jpg_path.name} - 分界线位置: {stitching_line} - 原因: {reason}")

                    # 查找对应的JSON文件（在同一目录下）
                    json_path = jpg_path.with_suffix('.json')

                    # 查找对应的PNG文件（在mask目录下的对应子文件夹中）
                    mask_folder_path = Path(mask_base_folder) / folder_name
                    png_path = mask_folder_path / jpg_path.with_suffix('.png').name

                    # 移动或复制文件
                    if move_files:
                        # 移动JPG文件
                        target_jpg = Path(target_folder) / jpg_path.name
                        shutil.move(str(jpg_path), str(target_jpg))

                        # 移动JSON文件（如果存在）
                        if json_path.exists():
                            target_json = Path(target_folder) / json_path.name
                            shutil.move(str(json_path), str(target_json))
                            logger.info(f"已移动文件: {jpg_path.name} 和对应的JSON文件")
                        else:
                            logger.warning(f"未找到对应的JSON文件: {json_path.name}")

                        # 移动PNG文件（如果存在）
                        if png_path.exists():
                            target_png = Path(wrong_mask_folder) / png_path.name
                            shutil.move(str(png_path), str(target_png))
                            logger.info(f"已移动mask文件: {png_path.name} 到 wrong_mask 文件夹")
                        else:
                            logger.warning(f"未找到对应的PNG文件: {png_path}")

                    else:
                        # 复制JPG文件
                        target_jpg = Path(target_folder) / jpg_path.name
                        shutil.copy2(str(jpg_path), str(target_jpg))

                        # 复制JSON文件（如果存在）
                        if json_path.exists():
                            target_json = Path(target_folder) / json_path.name
                            shutil.copy2(str(json_path), str(target_json))
                            logger.info(f"已复制文件: {jpg_path.name} 和对应的JSON文件")
                        else:
                            logger.warning(f"未找到对应的JSON文件: {json_path.name}")

                        # 复制PNG文件（如果存在）
                        if png_path.exists():
                            target_png = Path(wrong_mask_folder) / png_path.name
                            shutil.copy2(str(png_path), str(target_png))
                            logger.info(f"已复制mask文件: {png_path.name} 到 wrong_mask 文件夹")
                        else:
                            logger.warning(f"未找到对应的PNG文件: {png_path}")

                    # 创建可视化结果
                    if visualize:
                        detector.create_visualization(str(target_jpg) if move_files else str(jpg_path), stitching_line)

            except Exception as e:
                logger.error(f"处理文件 {jpg_path} 时出错: {e}")
                continue

        # 生成报告
        logger.info("\n" + "=" * 50)
        logger.info("检测完成!")
        logger.info(f"总共检测 {len(jpg_files)} 个文件")
        logger.info(f"发现 {len(detected_files)} 个疑似拼接图片")

        if detected_files:
            logger.info("疑似拼接图片列表:")
            for file_path, line_pos, reason in detected_files:
                logger.info(f"  - {file_path.name}: 分界线位置 {line_pos}, 原因: {reason}")

        return detected_files

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        return []


def main():
    """主函数"""
    base_source_folder = 'data'  # 基础源文件夹路径
    target_folder = "data/wrong data"  # 目标文件夹路径
    mask_base_folder = "mask_seg"  # mask基础文件夹路径
    wrong_mask_folder = "mask/wrong mask"  # wrong mask文件夹路径

    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)
    os.makedirs(wrong_mask_folder, exist_ok=True)

    # 处理001到050所有文件夹
    for i in range(1,51):
        # 生成文件夹名称，如001, 002, ..., 050
        folder_name = f"{i:03d}"
        source_folder = os.path.join(base_source_folder, folder_name)

        # 检查源文件夹是否存在
        if not os.path.exists(source_folder):
            logger.warning(f"源文件夹不存在: {source_folder}")
            continue

        logger.info(f"正在处理文件夹: {source_folder}")

        # 处理当前文件夹中的图片
        detected_files = process_images(
            source_folder=source_folder,
            target_folder=target_folder,
            mask_base_folder=mask_base_folder,
            wrong_mask_folder=wrong_mask_folder,
            folder_name=folder_name,
            move_files=True,  # 设为False表示复制而不是移动
            visualize=False,  # 为检测到的图片生成可视化结果
            gradient_threshold=50  # 调整这个值来改变检测灵敏度，值越小越敏感
        )

        if detected_files:
            logger.info(f"在文件夹 {folder_name} 中发现 {len(detected_files)} 个疑似拼接图片")
        else:
            logger.info(f"在文件夹 {folder_name} 中未发现疑似拼接图片")

    print("\n所有文件夹处理完成!")


if __name__ == "__main__":
    main()