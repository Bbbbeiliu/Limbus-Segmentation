import json
import numpy as np
from PIL import Image, ImageDraw
import os
import argparse
import math
from valid_region_detector import (detect_valid_region, apply_valid_region_mask)

def safe_float_conversion(value):
    """安全地将值转换为浮点数"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def calculate_edge_lengths(points):
    """计算多边形各边的长度（包括闭合边）"""
    if len(points) < 2.5:
        return []

    # 计算所有边的长度，包括最后一个点与第一个点的闭合边
    edge_lengths = []
    n = len(points)

    for i in range(n):
        # 当前点和下一个点（如果是最后一个点，则下一个点是第一个点）
        current = points[i]
        next_point = points[(i + 1) % n]

        # 计算两点之间的距离
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        edge_lengths.append(distance)

    return edge_lengths


def draw_polygon_with_gaps(draw, points, line_width=5):
    """绘制多边形，但跳过超过75%分位数长度3倍的边"""
    if len(points) < 2:
        return

    # 计算所有边的长度
    edge_lengths = calculate_edge_lengths(points)

    if not edge_lengths:
        return

    # 计算边长的75%分位数
    sorted_lengths = sorted(edge_lengths)
    n = len(sorted_lengths)

    # 计算75%分位数的位置
    q3_index = int(0.6 * n)
    q3_length = sorted_lengths[q3_index]

    threshold = q3_length * 2.5  # 阈值设为75%分位数的3倍

    # 绘制所有不超过阈值的边
    for i in range(len(points)):
        current_point = points[i]
        next_point = points[(i + 1) % len(points)]

        # 如果这条边的长度不超过阈值，则绘制
        if edge_lengths[i] <= threshold:
            draw.line([current_point, next_point], fill=255, width=line_width)

# def draw_polygon_with_gaps(draw, points, line_width=5):
#     """绘制多边形，但跳过超过平均长度3倍的边"""
#     if len(points) < 2:
#         return
#
#     # 计算所有边的长度
#     edge_lengths = calculate_edge_lengths(points)
#
#     if not edge_lengths:
#         return
#
#     # 计算边长的平均值
#     avg_length = sum(edge_lengths) / len(edge_lengths)
#
#     threshold = avg_length * 2.5  # 阈值设为平均长度的3倍
#
#     # 绘制所有不超过阈值的边
#     for i in range(len(points)):
#         current_point = points[i]
#         next_point = points[(i + 1) % len(points)]
#
#         # 如果这条边的长度不超过阈值，则绘制
#         if edge_lengths[i] <= threshold:
#             draw.line([current_point, next_point], fill=255, width=line_width)

def draw_circle_with_line_width(draw, center, point_on_circle, line_width=5):
    """绘制圆形轮廓，使用给定的线宽"""
    # 计算半径：圆心到圆上点的距离
    dx = point_on_circle[0] - center[0]
    dy = point_on_circle[1] - center[1]
    radius = int(round(math.sqrt(dx ** 2 + dy ** 2)))

    # 计算外接矩形的左上角和右下角坐标
    x0 = center[0] - radius
    y0 = center[1] - radius
    x1 = center[0] + radius
    y1 = center[1] + radius

    # 直接使用width参数绘制圆形轮廓
    draw.ellipse([x0, y0, x1, y1], outline=255, width=line_width)


def draw_polyline_with_gaps(draw, points, line_width=5):
    """绘制折线轮廓，跳过超过阈值的线段"""
    if len(points) < 2:
        return

    # 计算所有边的长度（不包括闭合边，因为折线不闭合）
    edge_lengths = []
    for i in range(len(points) - 1):
        current = points[i]
        next_point = points[i + 1]

        # 计算两点之间的距离
        dx = next_point[0] - current[0]
        dy = next_point[1] - current[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)
        edge_lengths.append(distance)

    if not edge_lengths:
        return

    # 计算边长的75%分位数
    sorted_lengths = sorted(edge_lengths)
    n = len(sorted_lengths)

    # 计算75%分位数的位置
    q3_index = int(0.6 * n)
    q3_length = sorted_lengths[q3_index]

    threshold = q3_length * 2.5  # 阈值设为75%分位数的2.5倍

    # 绘制所有不超过阈值的边
    for i in range(len(points) - 1):
        current_point = points[i]
        next_point = points[i + 1]

        # 如果这条边的长度不超过阈值，则绘制
        if i < len(edge_lengths) and edge_lengths[i] <= threshold:
            draw.line([current_point, next_point], fill=255, width=line_width)


def json_to_contour_mask(data_root,json_file, output_path, image_size=None, line_width=5):
    """
    将JSON标注文件转换为轮廓掩膜
    """

    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取图像尺寸 - 适应ISAT格式
        image_path = None
        if image_size is None:
            if 'info' in data:
                # ISAT格式
                image_width = data['info'].get('width', 0)
                image_height = data['info'].get('height', 0)
            else:
                # LabelMe格式
                image_width = data.get('imageWidth', 0)
                image_height = data.get('imageHeight', 0)

            # 如果JSON中没有图像尺寸，尝试从图像文件获取
            if image_width == 0 or image_height == 0:
                # 获取JSON文件所在目录
                json_dir = os.path.dirname(json_file)
                json_base = os.path.splitext(os.path.basename(json_file))[0]

                # 尝试查找多种格式的图像文件
                possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG']

                for ext in possible_extensions:
                    test_path = os.path.join(json_dir, json_base + ext)
                    if os.path.exists(test_path):
                        image_path = test_path
                        print(f"找到图像文件: {image_path}")
                        break

                if image_path and os.path.exists(image_path):
                    with Image.open(image_path) as img:
                        image_width, image_height = img.size
                else:
                    print(f"警告: 无法找到对应的图像文件，使用默认值 256x256")
                    image_width, image_height = 256, 256
        else:
            image_width, image_height = image_size

        # 创建空白图像
        mask = Image.new('L', (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask)

        # 检查是否有形状数据 - 适应ISAT格式
        shapes_processed = 0

        if 'objects' in data:
            # ISAT格式
            if not data['objects']:
                print(f"警告: {json_file} 中没有找到对象数据")
                mask.save(output_path)
                return

            # 处理ISAT格式的对象
            for obj in data['objects']:
                if 'segmentation' in obj and obj['segmentation']:
                    # 获取多边形点
                    raw_points = obj['segmentation']
                    points = []

                    # ISAT格式的点可能是嵌套列表
                    if isinstance(raw_points[0], list) and isinstance(raw_points[0][0], (int, float)):
                        # 格式: [[x1, y1], [x2, y2], ...]
                        for point in raw_points:
                            if len(point) >= 2:
                                x = safe_float_conversion(point[0])
                                y = safe_float_conversion(point[1])
                                points.append((x, y))
                    else:
                        # 格式: [x1, y1, x2, y2, ...]
                        for i in range(0, len(raw_points), 2):
                            if i + 1 < len(raw_points):
                                x = safe_float_conversion(raw_points[i])
                                y = safe_float_conversion(raw_points[i + 1])
                                points.append((x, y))

                    if len(points) < 2:
                        print(f"警告: 对象点数不足，跳过")
                        continue

                    # 将点转换为整数
                    points = [(int(round(x)), int(round(y))) for x, y in points]

                    # 绘制多边形，跳过过长的边
                    try:
                        draw_polygon_with_gaps(draw, points, line_width=line_width)
                        shapes_processed += 1
                    except Exception as e:
                        print(f"绘制形状时出错: {e}")

        elif 'shapes' in data:
            # LabelMe格式
            if not data['shapes']:
                print(f"警告: {json_file} 中没有找到形状数据")
                mask.save(output_path)
                return

            # 处理LabelMe格式的形状
            for shape in data['shapes']:
                shape_type = shape.get('shape_type', '')
                image_path = data['imagePath']

                # 处理多边形
                if shape_type == 'polygon' and 'points' in shape:
                    # 安全地转换坐标点
                    raw_points = shape['points']
                    points = []

                    for point in raw_points:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            x = safe_float_conversion(point[0])
                            y = safe_float_conversion(point[1])
                            points.append((x, y))

                    if len(points) < 2:
                        print(f"警告: 形状点数不足，跳过")
                        continue

                    # 将点转换为整数
                    points = [(int(round(x)), int(round(y))) for x, y in points]

                    # 绘制多边形，跳过过长的边
                    try:
                        draw_polygon_with_gaps(draw, points, line_width=line_width)
                        shapes_processed += 1
                    except Exception as e:
                        print(f"绘制多边形时出错: {e}")

                # 处理圆形
                elif shape_type == 'circle' and 'points' in shape:
                    raw_points = shape['points']
                    if len(raw_points) >= 2:
                        # 第一个点是圆心，第二个点是圆上一点
                        center_point = raw_points[0]
                        circle_point = raw_points[1]

                        if (isinstance(center_point, (list, tuple)) and len(center_point) >= 2 and
                                isinstance(circle_point, (list, tuple)) and len(circle_point) >= 2):

                            # 转换坐标为浮点数
                            center_x = safe_float_conversion(center_point[0])
                            center_y = safe_float_conversion(center_point[1])
                            circle_x = safe_float_conversion(circle_point[0])
                            circle_y = safe_float_conversion(circle_point[1])

                            # 将点转换为整数
                            center = (int(round(center_x)), int(round(center_y)))
                            circle_point = (int(round(circle_x)), int(round(circle_y)))

                            # 绘制圆形
                            try:
                                draw_circle_with_line_width(draw, center, circle_point, line_width=line_width)
                                shapes_processed += 1
                            except Exception as e:
                                print(f"绘制圆形时出错: {e}")
                        else:
                            print(f"警告: 圆形标注点格式错误，跳过")
                    else:
                        print(f"警告: 圆形标注点不足，跳过")

                    image_path = os.path.join(data_root, image_path)
                    print(image_path)
                    # 检测有效区域并清除无效区域的轮廓
                    if image_path and os.path.exists(image_path):
                        try:
                            print(f"image_path是: {image_path}")
                            print(f"开始检测有效区域: {image_path}")
                            # 检测有效区域
                            top_limit, bottom_limit = detect_valid_region(image_path, visualize=False)

                            # 应用有效区域掩膜，清除无效区域的轮廓
                            print(output_path)
                            mask = apply_valid_region_mask(mask, top_limit, bottom_limit)

                            # 打印清除区域信息
                            invalid_top = f"0-{top_limit - 1}" if top_limit > 0 else "无"
                            invalid_bottom = f"{bottom_limit + 1}-{image_height - 1}" if bottom_limit < image_height - 1 else "无"
                            print(f"已清除无效区域轮廓 (上: {invalid_top}, 下: {invalid_bottom})")
                        except Exception as e:
                            print(f"清除无效区域轮廓时出错: {e}")
                    else:
                        print(f"警告: 未找到图像文件，无法清除无效区域轮廓")
                        if not image_path:
                            print(f"图像路径为空，请检查图像文件是否存在")
                        else:
                            print(f"图像文件不存在: {image_path}")

                # 处理折线
                elif shape_type == 'linestrip' and 'points' in shape:
                    # 安全地转换坐标点
                    raw_points = shape['points']
                    points = []

                    for point in raw_points:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            x = safe_float_conversion(point[0])
                            y = safe_float_conversion(point[1])
                            points.append((x, y))

                    if len(points) < 2:
                        print(f"警告: 折线点数不足，跳过")
                        continue

                    # 将点转换为整数
                    points = [(int(round(x)), int(round(y))) for x, y in points]

                    # 绘制折线，跳过过长的线段
                    try:
                        draw_polyline_with_gaps(draw, points, line_width=line_width)
                        shapes_processed += 1
                    except Exception as e:
                        print(f"绘制折线时出错: {e}")

                # 其他形状类型
                elif shape_type:
                    print(f"警告: 不支持的形状类型 '{shape_type}'，跳过")

        else:
            print(f"警告: {json_file} 中未找到有效的标注数据格式")

        if shapes_processed == 0:
            print(f"警告: {json_file} 中没有成功处理任何形状")



        # 保存掩膜
        mask.save(output_path)
        print(f"轮廓掩膜已保存: {output_path} (处理了 {shapes_processed} 个形状)")

    except Exception as e:
        print(f"处理文件 {json_file} 时出错: {e}")
        # 创建一个空的掩膜文件
        try:
            mask = Image.new('L', (256, 256), 0)
            mask.save(output_path)
            print(f"已创建空掩膜: {output_path}")
        except:
            print(f"无法创建空掩膜: {output_path}")

def process_directory(data_root, mask_root, line_width=5, target_subdir=None):
    """批量处理目录中的JSON文件，可以指定特定子目录"""

    # 如果指定了目标子目录，则只处理该子目录
    if target_subdir:
        subdir_path = os.path.join(data_root, target_subdir)

        if not os.path.isdir(subdir_path):
            print(f"错误: 子目录 '{target_subdir}' 不存在")
            return

        print(f"处理目录: {target_subdir}")

        # 在mask目录中创建对应的子目录
        mask_subdir = os.path.join(mask_root, target_subdir)
        os.makedirs(mask_subdir, exist_ok=True)

        # 处理该子目录中的所有JSON文件
        json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]

        for json_file in json_files:
            json_path = os.path.join(subdir_path, json_file)
            output_path = os.path.join(mask_subdir, json_file.replace('.json', '.png'))

            print(f"处理: {target_subdir}/{json_file}")
            json_to_contour_mask(data_root,json_path, output_path, line_width=line_width)

    else:
        # 如果没有指定特定子目录，则使用原来的逻辑
        # 检查data_root目录下是否有子目录
        has_subdirs = False
        for item in os.listdir(data_root):
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path):
                has_subdirs = True
                break

        if has_subdirs:
            # 原逻辑：处理所有子目录
            for subdir in os.listdir(data_root):
                subdir_path = os.path.join(data_root, subdir)

                if not os.path.isdir(subdir_path):
                    continue

                print(f"处理目录: {subdir}")

                mask_subdir = os.path.join(mask_root, subdir)
                os.makedirs(mask_subdir, exist_ok=True)

                json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]

                for json_file in json_files:
                    json_path = os.path.join(subdir_path, json_file)
                    output_path = os.path.join(mask_subdir, json_file.replace('.json', '.png'))

                    print(f"处理: {subdir}/{json_file}")
                    json_to_contour_mask(data_root,json_path, output_path, line_width=line_width)
        else:
            # 直接处理当前目录下的JSON文件
            print(f"处理目录: {data_root}")

            os.makedirs(mask_root, exist_ok=True)

            json_files = [f for f in os.listdir(data_root) if f.endswith('.json')]

            for json_file in json_files:
                json_path = os.path.join(data_root, json_file)
                output_path = os.path.join(mask_root, json_file.replace('.json', '.png'))

                print(f"处理: {json_file}")
                json_to_contour_mask(data_root,json_path, output_path, line_width=line_width)

    print(f"处理完成! 已按照原始目录结构保存掩膜文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将JSON标注转换为轮廓掩膜')

    # 添加批次处理参数
    parser.add_argument('--start', type=int, default=109, help='起始编号')
    parser.add_argument('--end', type=int, default=109, help='结束编号')

    args = parser.parse_args()

    # 批量处理从 start 到 end 的所有目录
    for i in range(args.start, args.end + 1):
        num = f"{i:03d}"
        json_dir = f"data/{num}"
        output_dir = f"mask/{num}"
        # 检查输入目录是否存在
        if not os.path.exists(json_dir):
            print(f"跳过: 目录 {json_dir} 不存在")
            continue
        print(f"正在处理目录: {json_dir} -> {output_dir}")

        # 调用您已有的 process_directory 函数
        process_directory(json_dir, output_dir,5)

    print("批量处理完成！")