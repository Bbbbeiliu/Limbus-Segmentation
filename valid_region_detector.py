# valid_region_detector.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def detect_valid_region(image_path, gray_threshold=190, change_threshold=0.1, visualize=False):
    """
    检测图像中有效区域的高度上下限

    参数:
        image_path: 图像文件路径
        gray_threshold: 灰色背景的阈值 (0-255)
        change_threshold: 变化阈值比例，用于检测突变
        visualize: 是否可视化检测过程

    返回:
        (top_limit, bottom_limit): 有效区域的上下边界行数
    """
    try:
        # 打开图像
        img = Image.open(image_path)

        # 转换为灰度图
        gray_img = img.convert('L')

        # 获取图像尺寸
        width, height = gray_img.size

        print(f"图像尺寸: {width}x{height}")

        # 计算每行的像素值之和
        row_sums = []
        for y in range(height):
            row_sum = 0
            for x in range(width):
                pixel = gray_img.getpixel((x, y))
                row_sum += pixel
            row_sums.append(row_sum)

        # 归一化行和
        max_sum = max(row_sums)
        min_sum = min(row_sums)
        normalized_sums = [(s - min_sum) / (max_sum - min_sum) if max_sum > min_sum else 0 for s in row_sums]

        # 计算每行平均值用于分析
        row_means = [s / width for s in row_sums]

        # 检测上边界（从上往下）
        top_limit = 0
        for i in range(1, height):
            # 检测突变：当前行与前一行的差值超过阈值
            if abs(normalized_sums[i] - normalized_sums[i - 1]) > change_threshold:
                top_limit = i
                break

        # 检测下边界（从下往上）
        bottom_limit = height - 1
        for i in range(height - 2, -1, -1):
            # 检测突变：当前行与后一行的差值超过阈值
            if abs(normalized_sums[i] - normalized_sums[i + 1]) > change_threshold:
                bottom_limit = i
                break

        print(f"突变检测结果: top={top_limit}, bottom={bottom_limit}")

        # 如果检测失败，使用备用方法：基于灰度阈值
        if top_limit == 0 and bottom_limit == height - 1:
            print("使用备用方法检测...")
            # 备用方法：找到非灰色区域
            non_gray_rows = []
            for y in range(height):
                # 采样部分像素来判断是否是灰色
                sample_points = 20
                step = max(1, width // sample_points)
                gray_count = 0
                total_count = 0

                for x in range(0, width, step):
                    pixel = gray_img.getpixel((x, y))
                    if pixel >= gray_threshold:
                        gray_count += 1
                    total_count += 1

                # 如果灰色像素比例小于80%，认为是非灰色区域
                if gray_count / total_count < 0.8:
                    non_gray_rows.append(y)

            if non_gray_rows:
                # 添加一些缓冲区域
                buffer = max(1, height // 50)
                top_limit = max(0, min(non_gray_rows) - buffer)
                bottom_limit = min(height - 1, max(non_gray_rows) + buffer)
                print(f"基于灰度检测结果: top={top_limit}, bottom={bottom_limit}")
            else:
                # 如果还是没有找到，返回整个图像范围
                top_limit = 0
                bottom_limit = height - 1
                print("未检测到有效区域，使用整个图像范围")

        # 确保边界在合理范围内
        top_limit = max(0, min(top_limit, height - 1))
        bottom_limit = max(0, min(bottom_limit, height - 1))

        # 确保上边界小于下边界
        if top_limit > bottom_limit:
            top_limit, bottom_limit = bottom_limit, top_limit

        # 计算有效区域高度
        valid_height = bottom_limit - top_limit + 1
        print(f"有效区域: 行 {top_limit} 到 {bottom_limit}, 高度={valid_height}像素")

        # 新增：如果检测到的高度过小（可能误检测），强制使用默认固定区域
        if valid_height < 100:
            print("检测到的有效区域高度小于100，可能误检测，使用默认有效区域: 行56到199")
            # 确保默认值在图像范围内
            top_limit = max(0, min(56, height - 1))
            bottom_limit = max(0, min(199, height - 1))
            # 重新计算高度
            valid_height = bottom_limit - top_limit + 1
            print(f"调整后有效区域: 行 {top_limit} 到 {bottom_limit}, 高度={valid_height}像素")

        # 可视化（如果启用）
        if visualize:
            visualize_detection(gray_img, row_means, row_sums, top_limit, bottom_limit, gray_threshold)

        return top_limit, bottom_limit

    except Exception as e:
        print(f"检测有效区域时出错: {e}")
        # 返回整个图像范围作为默认值
        if 'img' in locals():
            width, height = img.size
            return 0, height - 1
        else:
            return 0, 255



def visualize_detection(gray_img, row_means, row_sums, top_limit, bottom_limit, gray_threshold):
    """可视化检测过程"""
    width, height = gray_img.size

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 原始灰度图像
    axes[0, 0].imshow(gray_img, cmap='gray')
    axes[0, 0].axhline(y=top_limit, color='r', linestyle='--', linewidth=2)
    axes[0, 0].axhline(y=bottom_limit, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_title(f'原始图像 (有效区域: {top_limit}-{bottom_limit})')
    axes[0, 0].set_xlabel('宽度')
    axes[0, 0].set_ylabel('高度')

    # 2. 每行平均像素值
    axes[0, 1].plot(row_means, range(height))
    axes[0, 1].set_ylim(height, 0)  # 反转y轴使图像方向正确
    axes[0, 1].axhline(y=top_limit, color='r', linestyle='--', linewidth=2)
    axes[0, 1].axhline(y=bottom_limit, color='r', linestyle='--', linewidth=2)
    axes[0, 1].axvline(x=gray_threshold, color='g', linestyle=':', alpha=0.5)
    axes[0, 1].set_title('每行平均像素值')
    axes[0, 1].set_xlabel('平均像素值')
    axes[0, 1].set_ylabel('行号')

    # 3. 每行像素值总和
    axes[1, 0].plot(row_sums, range(height))
    axes[1, 0].set_ylim(height, 0)
    axes[1, 0].axhline(y=top_limit, color='r', linestyle='--', linewidth=2)
    axes[1, 0].axhline(y=bottom_limit, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('每行像素值总和')
    axes[1, 0].set_xlabel('像素值总和')
    axes[1, 0].set_ylabel('行号')

    # 4. 直方图分析
    axes[1, 1].hist([row_means[:top_limit],
                     row_means[top_limit:bottom_limit + 1],
                     row_means[bottom_limit + 1:]],
                    bins=20, label=['上无效区', '有效区', '下无效区'], alpha=0.7)
    axes[1, 1].axvline(x=gray_threshold, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_title('像素值分布')
    axes[1, 1].set_xlabel('像素值')
    axes[1, 1].set_ylabel('频率')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def apply_valid_region_mask(mask, top_limit, bottom_limit):
    """
    将掩膜中无效区域（上下灰色部分）的像素值设置为0

    参数:
        mask: PIL Image对象 (L模式)
        top_limit: 有效区域上边界
        bottom_limit: 有效区域下边界

    返回:
        处理后的掩膜图像
    """
    try:
        # 转换为numpy数组以便处理
        mask_array = np.array(mask)
        height, width = mask_array.shape

        # 创建无效区域的掩膜
        invalid_mask = np.ones((height, width), dtype=bool)
        invalid_mask[top_limit:bottom_limit + 1, :] = False

        # 将无效区域的像素值设置为0
        mask_array[invalid_mask] = 0

        # 转换回PIL图像
        result_mask = Image.fromarray(mask_array)

        # 统计被清除的像素数量
        cleared_pixels = np.sum(invalid_mask & (mask_array != 0))
        total_pixels = np.sum(mask_array != 0)
        if total_pixels > 0:
            cleared_percentage = cleared_pixels / total_pixels * 100
            print(f"清除了 {cleared_pixels} 个轮廓像素 ({cleared_percentage:.1f}%)")

        return result_mask

    except Exception as e:
        print(f"应用有效区域掩膜时出错: {e}")
        return mask


def test_detection(image_path):
    """测试检测函数"""

    top, bottom = detect_valid_region(image_path, visualize=False)


    return top, bottom


if __name__ == "__main__":
    image_path = r"D:/project/segmentation/data/070/frame_000000.jpg"

    top, bottom = test_detection(image_path)
    print("-" * 50)
    print(f"最终结果: 上边界={top}, 下边界={bottom}")

