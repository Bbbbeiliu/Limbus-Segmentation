import json
import os
import argparse
from PIL import Image, ImageDraw


def extract_obstacles_from_json(json_dir, output_dir):
    """
    从JSON文件中提取障碍物并保存为透明PNG
    """
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for i, json_file in enumerate(json_files):
        json_path = os.path.join(json_dir, json_file)

        try:
            # 读取JSON数据
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取图像路径 - 直接从JSON中读取
            image_path = data.get('imagePath', '')
            print(image_path)
            if not image_path:
                # 如果JSON中没有imagePath，使用相同名称的图像文件
                base_name = os.path.splitext(json_file)[0]
                image_path = base_name + '.jpg'  # 假设是jpg格式

            # 尝试在JSON文件同一目录下查找图像
            json_parent = os.path.dirname(json_path)
            full_image_path = os.path.join(json_parent, image_path)

            if not os.path.exists(full_image_path):
                # 如果不在同一目录，尝试只使用文件名查找
                image_name = os.path.basename(image_path)
                full_image_path = os.path.join(json_parent, image_name)

            if not os.path.exists(full_image_path):
                print(f"警告: 无法找到 {json_file} 对应的图像文件 {image_path}")
                continue

            # 打开原图像
            original_image = Image.open(full_image_path).convert('RGBA')

            # 创建透明图像
            obstacle_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))

            # 处理所有形状
            shapes_processed = 0
            for shape in data.get('shapes', []):
                if shape['shape_type'] == 'polygon':
                    points = [(int(x), int(y)) for x, y in shape['points']]

                    # 创建掩膜
                    mask = Image.new('L', original_image.size, 0)
                    draw = ImageDraw.Draw(mask)
                    draw.polygon(points, fill=255)

                    # 应用掩膜提取障碍物
                    obstacle_region = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
                    obstacle_region.paste(original_image, (0, 0), mask)

                    # 合并到障碍物图像
                    obstacle_image = Image.alpha_composite(obstacle_image, obstacle_region)
                    shapes_processed += 1

            if shapes_processed == 0:
                print(f"警告: {json_file} 中没有找到有效的多边形标注")
                continue

            # 保存障碍物图像
            output_filename = f"obstacle_{i + 1:03d}.png"
            output_path = os.path.join(output_dir, output_filename)
            obstacle_image.save(output_path, 'PNG')
            print(f"已提取: {output_filename}")

        except Exception as e:
            print(f"处理 {json_file} 时出错: {e}")

    print(f"完成! 共处理 {len(json_files)} 个JSON文件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从Labelme JSON文件中提取障碍物')
    parser.add_argument('--json_dir', type=str, default='obstacle/mask',
                        help='JSON文件目录路径')
    parser.add_argument('--output_dir', type=str, default='obstacle/img',
                        help='输出透明PNG目录路径')

    args = parser.parse_args()

    extract_obstacles_from_json(args.json_dir, args.output_dir)