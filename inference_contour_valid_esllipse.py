import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from skimage import morphology
import json
import warnings
import matplotlib.pyplot as plt
import random
import collections

warnings.filterwarnings('ignore')

# RANSAC椭圆拟合参数
ELLIPSE_RANSAC_MAX_ITER = 10000
ELLIPSE_RANSAC_CONFIDENCE = 0.999
ELLIPSE_RANSAC_REPROJ_THRESHOLD = 1.0
ELLIPSE_RANSAC_INLIERS_RATIO = 0.5


class ResNetEncoder(nn.Module):
    """ResNet-18 encoder for U-Net"""

    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()

        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 1/2 resolution, 64 channels

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # 1/4 resolution, 64 channels

        x = self.layer2(x)
        features.append(x)  # 1/8 resolution, 128 channels

        x = self.layer3(x)
        features.append(x)  # 1/16 resolution, 256 channels

        x = self.layer4(x)
        features.append(x)  # 1/32 resolution, 512 channels

        return features


class DecoderBlock(nn.Module):
    """Decoder block with skip connection"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)

        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UNetResNet18(nn.Module):
    """U-Net with ResNet-18 encoder"""

    def __init__(self, num_classes=1, pretrained=True):
        super(UNetResNet18, self).__init__()

        self.encoder = ResNetEncoder(pretrained=pretrained)

        self.decoder4 = DecoderBlock(512, 256, 256)  # 1/16
        self.decoder3 = DecoderBlock(256, 128, 128)  # 1/8
        self.decoder2 = DecoderBlock(128, 64, 64)  # 1/4
        self.decoder1 = DecoderBlock(64, 64, 64)  # 1/2

        self.final_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)

        x = self.decoder4(features[4], features[3])  # 512 -> 256
        x = self.decoder3(x, features[2])  # 256 -> 128
        x = self.decoder2(x, features[1])  # 128 -> 64
        x = self.decoder1(x, features[0])  # 64 -> 64

        x = self.final_upconv(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x


def load_contour_model(encoder_path, decoder_path, device):
    """Load the trained contour model from separate encoder and decoder files"""

    # Initialize model
    model = UNetResNet18(num_classes=1, pretrained=False)

    # Load encoder weights
    if os.path.exists(encoder_path):
        encoder_state = torch.load(encoder_path, map_location=device)
        model.encoder.load_state_dict(encoder_state)
        print(f"Loaded contour encoder from: {encoder_path}")
    else:
        raise FileNotFoundError(f"Contour encoder file not found: {encoder_path}")

    # Load decoder weights
    if os.path.exists(decoder_path):
        decoder_state = torch.load(decoder_path, map_location=device)
        model.decoder4.load_state_dict(decoder_state['decoder4'])
        model.decoder3.load_state_dict(decoder_state['decoder3'])
        model.decoder2.load_state_dict(decoder_state['decoder2'])
        model.decoder1.load_state_dict(decoder_state['decoder1'])
        model.final_upconv.load_state_dict(decoder_state['final_upconv'])
        model.final_conv.load_state_dict(decoder_state['final_conv'])
        print(f"Loaded contour decoder from: {decoder_path}")
    else:
        raise FileNotFoundError(f"Contour decoder file not found: {decoder_path}")

    model = model.to(device)
    model.eval()

    return model


def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image for inference"""

    # Load and convert image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # Resize image
    image = image.resize(target_size, Image.BILINEAR)

    # Apply transforms (same as validation in contour training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor, original_size


def preprocess_frame(frame, target_size=(256, 256)):
    """Preprocess OpenCV frame for inference"""

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)
    original_size = (frame.shape[1], frame.shape[0])

    # Resize image
    image = image.resize(target_size, Image.BILINEAR)

    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor, original_size


def skeletonize_mask(mask_array):
    """Convert binary mask to single-pixel width skeleton using morphological operations"""

    # Ensure mask is binary
    binary_mask = (mask_array > 0).astype(np.uint8)

    # Method 1: Using scikit-image skeletonize (recommended)
    try:
        skeleton = morphology.skeletonize(binary_mask)
        return skeleton.astype(np.uint8) * 255
    except ImportError:
        print("scikit-image not available, using OpenCV method")

    # Method 2: Using OpenCV morphological operations as fallback
    kernel = np.ones((3, 3), np.uint8)

    # Create a copy of the mask
    skeleton = binary_mask.copy()

    # Iterative thinning
    while True:
        # Erode
        eroded = cv2.erode(skeleton, kernel)

        # Open operation to remove small components
        temp = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)

        # Subtract from skeleton
        temp = cv2.subtract(skeleton, temp)

        # Stop if no more changes
        if cv2.countNonZero(temp) == 0:
            break

        skeleton = eroded.copy()

    return skeleton * 255


def fit_ellipse_ransac(points, max_iterations=ELLIPSE_RANSAC_MAX_ITER,
                       confidence=ELLIPSE_RANSAC_CONFIDENCE,
                       reproj_threshold=ELLIPSE_RANSAC_REPROJ_THRESHOLD):
    """
    使用RANSAC方法拟合椭圆

    Args:
        points: 轮廓点数组，形状为 (N, 2)
        max_iterations: RANSAC最大迭代次数
        confidence: 置信度
        reproj_threshold: 重投影误差阈值

    Returns:
        ellipse_params: 椭圆参数 (center, axes, angle)
        inliers: 内点索引
    """
    if len(points) < 5:
        return None, None

    points_np = np.array(points, dtype=np.float32)

    # 使用RANSAC方法拟合椭圆
    best_ellipse = None
    best_inliers = []
    best_score = 0

    # 设置随机种子
    random.seed(42)

    for iteration in range(max_iterations):
        # 随机采样5个点拟合椭圆
        if len(points_np) >= 5:
            indices = random.sample(range(len(points_np)), 5)
        else:
            indices = list(range(len(points_np)))

        sample_points = points_np[indices]

        try:
            # 使用采样点拟合椭圆
            sample_ellipse = cv2.fitEllipse(sample_points)
            center, axes, angle = sample_ellipse

            # 计算点到椭圆的距离
            a, b = axes[0] / 2, axes[1] / 2
            angle_rad = np.radians(angle)

            # 旋转矩阵
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

            # 计算所有点到该椭圆的代数距离
            distances = []
            for point in points_np:
                # 转换为椭圆坐标系
                translated = point - center
                rotated = R @ translated

                x, y = rotated
                if abs(a) < 1e-6 or abs(b) < 1e-6:
                    distances.append(float('inf'))
                else:
                    # 椭圆方程: (x/a)^2 + (y/b)^2 = 1
                    dist = (x / a) ** 2 + (y / b) ** 2 - 1
                    distances.append(abs(dist))

            distances = np.array(distances)

            # 内点：距离小于阈值的点
            inliers = distances < reproj_threshold
            score = np.sum(inliers)

            if score > best_score:
                best_score = score
                best_ellipse = sample_ellipse
                best_inliers = np.where(inliers)[0]

                # 如果内点比例足够高，提前退出
                inlier_ratio = score / len(points_np)
                if inlier_ratio > ELLIPSE_RANSAC_INLIERS_RATIO:
                    break

        except Exception as e:
            continue

    # 如果有足够的内点，使用所有内点重新拟合椭圆
    if best_ellipse is not None and len(best_inliers) >= 5:
        try:
            inlier_points = points_np[best_inliers]
            final_ellipse = cv2.fitEllipse(inlier_points)
            return final_ellipse, best_inliers
        except:
            # 如果重新拟合失败，返回原始拟合结果
            return best_ellipse, best_inliers

    # 如果RANSAC完全失败，使用所有点拟合椭圆
    try:
        fallback_ellipse = cv2.fitEllipse(points_np)
        return fallback_ellipse, np.arange(len(points_np))
    except:
        return None, None


def fit_shape_to_contour(mask_array, shape_type='ellipse', min_contour_points=10):
    """
    使用RANSAC方法拟合椭圆到轮廓

    Args:
        mask_array: 二值掩码数组
        shape_type: 拟合形状类型，目前只支持'ellipse'
        min_contour_points: 最小轮廓点数

    Returns:
        shape_params: 形状参数
        contour: 轮廓点
        binary_mask: 二值掩码
    """
    # 转换为二值图像
    binary_mask = (mask_array > 0).astype(np.uint8)

    # 如果掩码为空，直接返回
    if np.sum(binary_mask) == 0:
        return None, None, None

    # 形态学操作：填充小孔洞，连接断点
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None, None

    # 找到最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 检查是否有足够的点
    if len(largest_contour) < min_contour_points:
        return None, None, None

    shape_params = None
    contour = largest_contour

    if shape_type == 'ellipse':
        # 提取轮廓点
        points = largest_contour.reshape(-1, 2)

        # 使用RANSAC拟合椭圆
        ellipse, inlier_indices = fit_ellipse_ransac(points)

        if ellipse is not None:
            (center_x, center_y), (axes_a, axes_b), angle = ellipse
            shape_params = {
                'type': 'ellipse',
                'center': (center_x, center_y),
                'axes': (axes_a, axes_b),
                'angle': angle,
                'inliers': len(inlier_indices) if inlier_indices is not None else 0,
                'total_points': len(points)
            }
            print(f"RANSAC椭圆拟合: 总点数={len(points)}, 内点数={shape_params['inliers']}, "
                  f"中心=({center_x:.1f}, {center_y:.1f}), "
                  f"轴长=({axes_a:.1f}, {axes_b:.1f}), 角度={angle:.1f}°")
        else:
            print("RANSAC椭圆拟合失败")
            return None, None, None

    else:
        # 只支持椭圆拟合
        raise ValueError(f"只支持椭圆拟合, 不支持: {shape_type}")

    return shape_params, contour, binary_mask


def draw_ellipse_mask(ellipse_params, size):
    """Create a binary mask with the fitted ellipse"""

    mask = np.zeros(size, dtype=np.uint8)

    if ellipse_params is None:
        return mask

    # 绘制椭圆轮廓
    center = (int(ellipse_params['center'][0]), int(ellipse_params['center'][1]))
    axes = (int(ellipse_params['axes'][0] / 2), int(ellipse_params['axes'][1] / 2))
    angle = ellipse_params['angle']

    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=1)  # thickness=1 for contour only

    return mask


def create_contour_overlay(original_image_path, mask_image, alpha=0.6, contour_color=(0, 255, 255)):
    """Create an overlay of the original image and contour segmentation mask"""

    # Load original image
    original = Image.open(original_image_path).convert('RGB')

    # Resize mask to match original if needed
    if mask_image.size != original.size:
        mask_image = mask_image.resize(original.size, Image.NEAREST)

    # Convert mask to numpy for processing
    mask_array = np.array(mask_image)

    # Create colored mask
    mask_colored = Image.new('RGB', mask_image.size, contour_color)
    mask_colored_array = np.array(mask_colored)

    # Create overlay
    original_array = np.array(original)
    overlay_array = original_array.copy()

    # Apply mask with transparency - only where mask is non-zero
    mask_indices = mask_array > 0
    overlay_array[mask_indices] = (
            (1 - alpha) * original_array[mask_indices] +
            alpha * mask_colored_array[mask_indices]
    ).astype(np.uint8)

    overlay_image = Image.fromarray(overlay_array)

    return overlay_image


def create_shape_overlay(original_image_path, shape_mask_image, alpha=0.8,
                         shape_color=(0, 255, 0), shape_params=None):
    """Create an overlay of the original image with fitted shape (ellipse)"""

    # Load original image
    original = Image.open(original_image_path).convert('RGB')

    # Resize mask to match original if needed
    if shape_mask_image.size != original.size:
        shape_mask_image = shape_mask_image.resize(original.size, Image.NEAREST)

    # Convert mask to numpy for processing
    shape_mask_array = np.array(shape_mask_image)

    # Create colored shape mask
    shape_colored = Image.new('RGB', shape_mask_image.size, shape_color)
    shape_colored_array = np.array(shape_colored)

    # Create overlay
    original_array = np.array(original)
    overlay_array = original_array.copy()

    # Apply shape mask with transparency - only where shape contour exists
    shape_indices = shape_mask_array > 0
    overlay_array[shape_indices] = (
            (1 - alpha) * original_array[shape_indices] +
            alpha * shape_colored_array[shape_indices]
    ).astype(np.uint8)

    # If we have shape parameters, draw the center point
    if shape_params is not None:
        # Convert PIL image back to numpy for OpenCV drawing
        overlay_array_cv = np.array(overlay_array)
        center_x, center_y = int(shape_params['center'][0]), int(shape_params['center'][1])
        cv2.circle(overlay_array_cv, (center_x, center_y), 5, (255, 0, 0), -1)  # Red center point

        # 绘制内点数量信息
        inlier_text = f"Inliers: {shape_params.get('inliers', 0)}/{shape_params.get('total_points', 0)}"
        cv2.putText(overlay_array_cv, inlier_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        overlay_array = overlay_array_cv

    overlay_image = Image.fromarray(overlay_array)

    return overlay_image


# ==================== 修改的椭圆筛选功能 ====================
def evaluate_ellipse_coverage(ellipse_params, original_mask, min_coverage=0.5):
    """
    评估椭圆轮廓点与分割掩膜的重合度

    Args:
        ellipse_params: 椭圆参数
        original_mask: 原始分割掩膜 (二值图，0-255)
        min_coverage: 最小重合度阈值

    Returns:
        bool: True表示椭圆有效，False表示应过滤掉
        str: 评估结果
        float: 重合度百分比
    """
    if ellipse_params is None:
        return False, "椭圆参数为空", 0.0

    # 检查轴长
    axes = ellipse_params['axes']
    if min(axes) <= 0:
        return False, "轴长为零或负数", 0.0

    # 检查轴长比例（可选，保持原有的几何检查）
    axis_ratio = min(axes) / max(axes)
    if axis_ratio < 0.6:  # 保持相对宽松的比例检查
        return False, f"轴长比例异常: {axis_ratio:.3f}", 0.0

    # 创建椭圆轮廓（不填充）
    ellipse_contour = np.zeros_like(original_mask)
    center = (int(ellipse_params['center'][0]), int(ellipse_params['center'][1]))
    axes_half = (int(axes[0] / 2), int(axes[1] / 2))
    angle = ellipse_params['angle']

    # 绘制椭圆轮廓（线宽为1像素）
    cv2.ellipse(ellipse_contour, center, axes_half, angle, 0, 360, 255, thickness=1)

    # 统计椭圆轮廓上的点
    ellipse_points = np.where(ellipse_contour > 0)
    total_points = len(ellipse_points[0])

    if total_points == 0:
        return False, "无法生成椭圆轮廓", 0.0

    # 统计落在原始分割掩膜上的点
    coverage_count = 0
    for y, x in zip(ellipse_points[0], ellipse_points[1]):
        if original_mask[y, x] > 0:
            coverage_count += 1

    # 计算重合度百分比
    coverage_ratio = coverage_count / total_points

    if coverage_ratio < min_coverage:
        return False, f"重合度过低: {coverage_ratio:.3f}", coverage_ratio

    return True, f"重合度: {coverage_ratio:.3f}", coverage_ratio


# ==================== 椭圆参数平滑器 ====================
class SimpleJitterFilter:
    """修正版抖动过滤器：检测并跳过单帧抖动"""

    def __init__(self, threshold=10, max_fixed_frames=1):
        """
        Args:
            threshold: 像素阈值，超过此值认为是抖动
            max_fixed_frames: 最多连续修正几帧（防止无限卡住）
        """
        self.threshold = threshold
        self.max_fixed_frames = max_fixed_frames

        # 存储历史
        self.prev_center = None
        self.prev_prev_center = None  # 前两帧的中心
        self.fixed_count = 0  # 连续修正的帧数
        self.last_was_fixed = False  # 上一帧是否被修正

    def filter_jitter(self, ellipse_params):
        """过滤抖动，返回修正后的参数或原始参数"""
        if ellipse_params is None:
            return None

        center = ellipse_params['center']

        # 第一帧，只记录不处理
        if self.prev_center is None:
            self.prev_center = center
            return ellipse_params

        # 第二帧，记录前一帧中心
        if self.prev_prev_center is None:
            self.prev_prev_center = self.prev_center
            self.prev_center = center
            return ellipse_params

        # 计算当前帧与前一帧的距离
        dx = center[0] - self.prev_center[0]
        dy = center[1] - self.prev_center[1]
        distance = np.sqrt(dx * dx + dy * dy)

        # 检查是否是单帧抖动（突然跳出去又跳回来）
        if distance > self.threshold:
            print(f"检测到可能抖动: 距离={distance:.1f} > {self.threshold}")

            # 计算与更早一帧的距离
            dx2 = center[0] - self.prev_prev_center[0]
            dy2 = center[1] - self.prev_prev_center[1]
            distance2 = np.sqrt(dx2 * dx2 + dy2 * dy2)

            # 关键逻辑：如果当前帧与前两帧相似，但与前一帧不同，说明是单帧抖动
            if distance2 < self.threshold and self.last_was_fixed == False:
                print(f"确认为单帧抖动: 与前两帧距离={distance2:.1f} < 阈值")
                print(f"跳过该抖动帧，使用前一帧中心")

                # 使用前一帧的中心，但其他参数保持当前帧
                fixed_params = ellipse_params.copy()
                fixed_params['center'] = self.prev_center

                # 重要：不更新prev_center，保持原来的轨迹
                # 但更新prev_prev_center
                self.prev_prev_center = self.prev_center

                self.fixed_count = 1
                self.last_was_fixed = True
                return fixed_params

            # 如果是连续抖动或移动，正常更新
            else:
                print(f"可能是真实移动或连续抖动: 与前两帧距离={distance2:.1f}")
                self.prev_prev_center = self.prev_center
                self.prev_center = center
                self.fixed_count = 0
                self.last_was_fixed = False
                return ellipse_params

        else:
            # 正常移动，更新历史
            self.prev_prev_center = self.prev_center
            self.prev_center = center
            self.fixed_count = 0
            self.last_was_fixed = False
            return ellipse_params

    def reset(self):
        """重置过滤器"""
        self.prev_center = None
        self.prev_prev_center = None
        self.fixed_count = 0
        self.last_was_fixed = False


class EllipseSmoother:
    """使用移动平均平滑椭圆参数"""

    def __init__(self, window_size=3, jitter_threshold=10):
        self.window_size = window_size

        # 存储历史参数
        self.history_centers = collections.deque(maxlen=window_size)
        self.history_axes = collections.deque(maxlen=window_size)
        self.history_angles_sin = collections.deque(maxlen=window_size)
        self.history_angles_cos = collections.deque(maxlen=window_size)
        self.jitter_filter = SimpleJitterFilter(threshold=jitter_threshold)

        self.frame_count = 0

    def smooth_ellipse(self, ellipse_params):
        """平滑椭圆参数"""
        if ellipse_params is None:
            return None

        # 先过滤抖动
        ellipse_params = self.jitter_filter.filter_jitter(ellipse_params)

        # 提取当前参数
        center = ellipse_params['center']
        axes = ellipse_params['axes']
        angle = ellipse_params['angle']

        # 将角度转换为sin和cos进行平滑（处理角度环绕问题）
        angle_rad = np.radians(angle)
        self.history_centers.append(center)
        self.history_axes.append(axes)
        self.history_angles_sin.append(np.sin(angle_rad))
        self.history_angles_cos.append(np.cos(angle_rad))

        self.frame_count += 1

        # 如果历史数据不足，返回原始值
        if self.frame_count < 2 or len(self.history_centers) < 2:
            return ellipse_params

        # 计算移动平均值
        centers_array = np.array(self.history_centers)
        smoothed_center = np.mean(centers_array, axis=0)

        axes_array = np.array(self.history_axes)
        smoothed_axes = np.mean(axes_array, axis=0)

        # 平均sin和cos，然后计算角度
        avg_sin = np.mean(self.history_angles_sin)
        avg_cos = np.mean(self.history_angles_cos)
        smoothed_angle = np.degrees(np.arctan2(avg_sin, avg_cos))

        # 创建平滑后的椭圆参数
        smoothed_params = ellipse_params.copy()
        smoothed_params['center'] = tuple(smoothed_center)
        smoothed_params['axes'] = tuple(smoothed_axes)
        smoothed_params['angle'] = smoothed_angle

        return smoothed_params

    def reset(self):
        """重置平滑器"""
        self.history_centers.clear()
        self.history_axes.clear()
        self.history_angles_sin.clear()
        self.history_angles_cos.clear()
        self.frame_count = 0


def filter_bad_ellipse(ellipse_params, original_mask, min_coverage=0.5):
    """
    基于椭圆轮廓点重合度的筛选函数

    Args:
        ellipse_params: 椭圆参数
        original_mask: 原始分割掩膜
        min_coverage: 最小重合度阈值

    Returns:
        bool: True表示椭圆有效，False表示应过滤掉
        str: 过滤原因
    """
    is_valid, reason, coverage = evaluate_ellipse_coverage(ellipse_params, original_mask, min_coverage)
    return is_valid, reason


def postprocess_contour_mask(mask_tensor, original_size, threshold=0.5,
                             skeletonize=True, fit_shape=True, shape_type='ellipse'):
    """Convert model output to binary contour mask and resize to original size, then fit shape"""

    # 移除批次维度并转换为numpy
    mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # 应用阈值
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # 保存原始二值掩膜（用于筛选）
    original_binary_mask = binary_mask.copy()

    # 应用骨架化来获取单像素宽度的轮廓
    if skeletonize:
        try:
            binary_mask = skeletonize_mask(binary_mask)
        except Exception as e:
            print(f"Skeletonization failed: {e}, using original mask")

    # 转换为PIL Image并调整到原始尺寸
    mask_image = Image.fromarray(binary_mask, mode='L')
    mask_image = mask_image.resize(original_size, Image.NEAREST)

    # 创建单像素掩膜（骨架化后的掩膜）
    skeleton_mask_image = mask_image.copy()

    # 转换回numpy进行形状拟合
    mask_array = np.array(mask_image)

    # 拟合形状
    shape_mask = None
    shape_params = None

    if fit_shape:
        shape_params, contour, _ = fit_shape_to_contour(
            mask_array,
            shape_type=shape_type,
            min_contour_points=5  # 椭圆拟合至少需要5个点
        )

        if shape_params is not None:
            # 创建椭圆掩码
            shape_mask = draw_ellipse_mask(shape_params, (original_size[1], original_size[0]))

            # 转换形状掩码为PIL Image
            shape_mask_image = Image.fromarray(shape_mask, mode='L')
        else:
            # 如果形状拟合失败，返回None
            shape_mask_image = None
    else:
        shape_mask_image = None

    # 返回原始掩膜用于筛选
    original_binary_mask_resized = cv2.resize(original_binary_mask, original_size,
                                              interpolation=cv2.INTER_NEAREST)

    return shape_mask_image, shape_params, skeleton_mask_image, original_binary_mask_resized


def inference_on_folder_contour(input_folder, output_folder, model, device, threshold=0.5, shape_type='ellipse',
                                filter_ellipse=True, min_coverage=0.5, save_comparison=True):
    """Process images in folder: fit shape to contours and create comparison images"""

    # Create output directories
    os.makedirs(output_folder, exist_ok=True)

    # 创建合成对比图文件夹
    comparison_folder = os.path.join(output_folder, 'comparison_images')
    os.makedirs(comparison_folder, exist_ok=True)

    # 仍然创建形状参数文件夹用于保存参数
    shape_params_folder = os.path.join(output_folder, 'shape_parameters')
    os.makedirs(shape_params_folder, exist_ok=True)

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    if not image_files:
        print(f"No image files found in {input_folder}")
        return

    print(f"Found {len(image_files)} images to process")
    print(f"Using RANSAC ellipse fitting")
    if filter_ellipse:
        print(f"Ellipse filtering enabled (min coverage: {min_coverage})")

    print(f"Saving comparison images to: {comparison_folder}")

    # Store shape parameters
    shape_params_dict = {}
    processed_count = 0
    failed_count = 0

    # Process each image
    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                image_path = os.path.join(input_folder, image_file)
                base_name = os.path.splitext(image_file)[0]

                # Preprocess
                image_tensor, original_size = preprocess_image(image_path)
                image_tensor = image_tensor.to(device)

                # Inference
                output = model(image_tensor)

                # Fit shape to contour using RANSAC
                shape_mask_image, shape_params, skeleton_mask_image, original_binary_mask = postprocess_contour_mask(
                    output, original_size, threshold, skeletonize=True, fit_shape=True, shape_type=shape_type
                )

                # 读取原始图像（用于合成图）
                original_frame = cv2.imread(image_path)
                if original_frame is None:
                    print(f"Error reading image: {image_file}")
                    failed_count += 1
                    continue

                # 椭圆筛选
                filtered = False
                filter_reason = ""
                if filter_ellipse and shape_params is not None:
                    is_valid, reason = filter_bad_ellipse(shape_params, original_binary_mask, min_coverage)
                    if not is_valid:
                        filtered = True
                        filter_reason = reason
                        print(f"Filtered ellipse for {image_file}: {reason}")
                        shape_params = None
                        shape_mask_image = None

                # 创建合成对比图
                if save_comparison:
                    try:
                        comparison_img = create_comparison_image(
                            original_frame,  # 使用OpenCV读取的原始图像
                            output,
                            skeleton_mask_image,
                            shape_mask_image,
                            original_size,
                            threshold=threshold,
                            shape_params=shape_params,
                            filter_reason=filter_reason if filtered else None
                        )

                        # 保存合成对比图
                        comparison_path = os.path.join(comparison_folder, f"{base_name}_comparison.png")
                        comparison_img.save(comparison_path)

                    except Exception as e:
                        print(f"Error creating comparison image for {image_file}: {str(e)}")
                        # 创建一个简单的错误提示图
                        error_img = Image.new('RGB', (800, 600), color='white')
                        # 可以在图上添加错误信息，这里简化处理
                        error_path = os.path.join(comparison_folder, f"{base_name}_error.png")
                        error_img.save(error_path)

                # Store shape parameters (even if filtered, but mark as filtered)
                if shape_params is not None:
                    shape_params_dict[image_file] = {
                        'params': shape_params,
                        'filtered': filtered,
                        'filter_reason': filter_reason if filtered else ""
                    }
                else:
                    shape_params_dict[image_file] = {
                        'params': None,
                        'filtered': True,
                        'filter_reason': filter_reason if filter_reason else "No ellipse detected"
                    }

                processed_count += 1

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_count += 1

    # Save shape parameters to JSON
    if shape_params_dict:
        params_path = os.path.join(shape_params_folder, 'shape_parameters.json')
        serializable_params = {}
        for key, data in shape_params_dict.items():
            if data['params'] is not None:
                params = data['params']
                if params['type'] == 'ellipse':
                    serializable_params[key] = {
                        'type': 'ellipse',
                        'center': [float(params['center'][0]), float(params['center'][1])],
                        'axes': [float(params['axes'][0]), float(params['axes'][1])],
                        'angle': float(params['angle']),
                        'inliers': int(params.get('inliers', 0)),
                        'total_points': int(params.get('total_points', 0)),
                        'filtered': data['filtered'],
                        'filter_reason': data['filter_reason']
                    }
            else:
                serializable_params[key] = {
                    'type': None,
                    'center': None,
                    'axes': None,
                    'angle': None,
                    'inliers': 0,
                    'total_points': 0,
                    'filtered': True,
                    'filter_reason': data['filter_reason']
                }

        with open(params_path, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        print(f"Shape parameters saved to: {params_path}")

    print(f"\nImage processing completed!")
    print(f"Processed: {processed_count} images")
    print(f"Failed: {failed_count} images")
    print(f"Comparison images saved to: {comparison_folder}")
    print(f"Shape parameters saved to: {shape_params_folder}")


def process_video_contour(input_video_path, output_folder, model, device, threshold=0.5, fps=None,
                          shape_type='ellipse', filter_ellipse=True, min_coverage=0.5):
    """Process video file: fit shape to contours and create output video with overlay"""

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # 新增：创建对比图文件夹（每隔2帧）
    comparison_folder = os.path.join(output_folder, 'B2_comparison_frames')
    os.makedirs(comparison_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # Get video properties
    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) if fps is None else fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Info:")
    print(f"  FPS: {video_fps}")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  Total frames: {total_frames}")
    print(f"  Using RANSAC ellipse fitting")
    if filter_ellipse:
        print(f"  Ellipse filtering enabled (min coverage: {min_coverage})")
    print(f"  Comparison frames will be saved every 2 frames to: {comparison_folder}")

    # 新增：初始化椭圆平滑器
    print(f"  Ellipse smoothing enabled (window size: 5)")
    ellipse_smoother = EllipseSmoother(window_size=3)

    # Create output video
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = os.path.join(output_folder, f"C2_{video_name}_shape_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))

    # Store shape parameters
    shape_params_list = []
    frame_count = 0
    filtered_count = 0

    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc="Processing video frames")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            image_tensor, original_size = preprocess_frame(frame)
            image_tensor = image_tensor.to(device)

            # Inference
            output = model(image_tensor)

            # Fit shape to contour using RANSAC
            shape_mask_image, shape_params, skeleton_mask_image, original_binary_mask = postprocess_contour_mask(
                output, original_size, threshold, skeletonize=True, fit_shape=True, shape_type=shape_type
            )

            # 椭圆筛选
            draw_ellipse = False
            filter_reason = ""
            if shape_params is not None and filter_ellipse:
                is_valid, reason = filter_bad_ellipse(shape_params, original_binary_mask, min_coverage)
                if is_valid:
                    draw_ellipse = True
                    # 新增：应用椭圆平滑
                    shape_params = ellipse_smoother.smooth_ellipse(shape_params)
                else:
                    filtered_count += 1
                    filter_reason = reason
                    shape_params = None  # 不绘制这个椭圆
                    # 新增：重置平滑器，因为当前帧无效
                    ellipse_smoother.reset()
            elif shape_params is not None:
                draw_ellipse = True
                # 新增：应用椭圆平滑
                shape_params = ellipse_smoother.smooth_ellipse(shape_params)
            else:
                # 新增：如果椭圆参数为None，重置平滑器
                ellipse_smoother.reset()

            # 新增：每隔2帧生成对比图
            if frame_count % 2 == 0 and skeleton_mask_image is not None:
                # 创建对比图
                comparison_img = create_comparison_image(
                    frame,
                    output,
                    skeleton_mask_image,
                    shape_mask_image,
                    original_size,
                    threshold=threshold,
                    shape_params=shape_params,
                    filter_reason=filter_reason if filter_reason else None
                )

                # 保存对比图
                comparison_path = os.path.join(
                    comparison_folder,
                    f"{video_name}_frame_{frame_count:06d}_comparison.png"
                )
                comparison_img.save(comparison_path)

                print(f"Saved comparison image for frame {frame_count}: {comparison_path}")

            # Create overlay
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            overlay = frame_rgb.copy()

            # Draw shape if fitting succeeded and passed filter
            if shape_params is not None and draw_ellipse:
                if shape_params['type'] == 'ellipse':
                    # Draw ellipse contour
                    center = (int(shape_params['center'][0]), int(shape_params['center'][1]))
                    axes = (int(shape_params['axes'][0] / 2), int(shape_params['axes'][1] / 2))
                    angle = shape_params['angle']
                    cv2.ellipse(overlay, center, axes, angle, 0, 360, (0, 255, 0), 2)  # Green ellipse
                    cv2.circle(overlay, center, 5, (255, 0, 0), -1)  # Red center point

                    # 添加内点信息文本
                    inliers = shape_params.get('inliers', 0)
                    total_points = shape_params.get('total_points', 0)
                    inlier_text = f"Inliers: {inliers}/{total_points}"
                    cv2.putText(overlay, inlier_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 如果椭圆被过滤，显示原因
            if filter_reason:
                cv2.putText(overlay, f"Filtered: {filter_reason}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Write frame to output video
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            out.write(overlay_bgr)

            # Store shape parameters
            if shape_params is not None:
                if shape_params['type'] == 'ellipse':
                    shape_params_list.append({
                        'frame': frame_count,
                        'type': 'ellipse',
                        'center': [float(shape_params['center'][0]), float(shape_params['center'][1])],
                        'axes': [float(shape_params['axes'][0]), float(shape_params['axes'][1])],
                        'angle': float(shape_params['angle']),
                        'inliers': int(shape_params.get('inliers', 0)),
                        'total_points': int(shape_params.get('total_points', 0)),
                        'filtered': not draw_ellipse,
                        'filter_reason': filter_reason if filter_reason else ""
                    })
            else:
                shape_params_list.append({
                    'frame': frame_count,
                    'type': None,
                    'center': None,
                    'axes': None,
                    'angle': None,
                    'inliers': 0,
                    'total_points': 0,
                    'filtered': True,
                    'filter_reason': filter_reason if filter_reason else "No ellipse detected"
                })

            frame_count += 1
            pbar.update(1)

        pbar.close()

    # Cleanup
    cap.release()
    out.release()

    # Save shape parameters
    if shape_params_list:
        params_path = os.path.join(output_folder, 'shape_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(shape_params_list, f, indent=2)
        print(f"Shape parameters saved to: {params_path}")

    print(f"Video processing completed!")
    print(f"Processed {frame_count} frames")
    print(f"Filtered {filtered_count} ellipses")
    print(f"Output video saved to: {output_video_path}")
    print(f"Comparison frames saved to: {comparison_folder} (every 10 frames)")


def create_comparison_image(frame, model_output, skeleton_mask_image, shape_mask_image,
                            original_size, threshold=0.5, shape_params=None, filter_reason=None):
    """创建包含原图、分割掩膜、骨架化掩膜和RANSAC拟合椭圆的对比图"""

    # 将OpenCV BGR图像转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 获取原始分割掩膜（未骨架化）
    mask = model_output.squeeze(0).squeeze(0).cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    binary_mask_image = Image.fromarray(binary_mask, mode='L')
    binary_mask_image = binary_mask_image.resize(original_size, Image.NEAREST)

    # 将PIL图像转换为numpy数组
    binary_mask_array = np.array(binary_mask_image)
    skeleton_mask_array = np.array(skeleton_mask_image) if skeleton_mask_image is not None else np.zeros(
        (original_size[1], original_size[0]), dtype=np.uint8)

    # 确保骨架化掩膜和形状掩膜大小一致
    if skeleton_mask_array.shape[:2] != original_size[::-1] and skeleton_mask_image is not None:
        skeleton_mask_image = skeleton_mask_image.resize(original_size, Image.NEAREST)
        skeleton_mask_array = np.array(skeleton_mask_image)

    if shape_mask_image is not None:
        shape_mask_array = np.array(shape_mask_image)
        if shape_mask_array.shape[:2] != original_size[::-1]:
            shape_mask_image = shape_mask_image.resize(original_size, Image.NEAREST)
            shape_mask_array = np.array(shape_mask_image)
    else:
        shape_mask_array = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)

    # 创建彩色掩膜（绿色表示分割掩膜，红色表示骨架，蓝色表示拟合椭圆）
    seg_colored = np.zeros((*original_size[::-1], 3), dtype=np.uint8)
    seg_colored[binary_mask_array > 0] = [0, 255, 0]  # 绿色

    skeleton_colored = np.zeros((*original_size[::-1], 3), dtype=np.uint8)
    skeleton_colored[skeleton_mask_array > 0] = [255, 0, 0]  # 红色

    shape_colored = np.zeros((*original_size[::-1], 3), dtype=np.uint8)
    shape_colored[shape_mask_array > 0] = [0, 0, 255]  # 蓝色

    # 创建掩膜叠加图
    overlay_seg = cv2.addWeighted(frame_rgb, 0.7, seg_colored, 0.3, 0)
    overlay_skeleton = cv2.addWeighted(frame_rgb, 0.7, skeleton_colored, 0.3, 0)
    overlay_shape = cv2.addWeighted(frame_rgb, 0.7, shape_colored, 0.3, 0)
    y_offset = 30

    # 如果有椭圆参数，在拟合椭圆图上绘制椭圆和中心点
    if shape_params is not None:
        center = (int(shape_params['center'][0]), int(shape_params['center'][1]))
        axes = (int(shape_params['axes'][0] / 2), int(shape_params['axes'][1] / 2))
        angle = shape_params['angle']

        # 在拟合椭圆图上绘制椭圆
        cv2.ellipse(overlay_shape, center, axes, angle, 0, 360, (255, 255, 0), 2)  # 青色椭圆
        cv2.circle(overlay_shape, center, 5, (255, 0, 255), -1)  # 洋红色中心点

        # 添加参数文本
        param_text = f"Center: ({center[0]}, {center[1]})"
        axes_text = f"Axes: ({axes[0] * 2:.1f}, {axes[1] * 2:.1f})"
        angle_text = f"Angle: {angle:.1f}°"
        inlier_text = f"Inliers: {shape_params.get('inliers', 0)}/{shape_params.get('total_points', 0)}"


        for text in [param_text, axes_text, angle_text, inlier_text]:
            cv2.putText(overlay_shape, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

    # 如果椭圆被过滤，添加过滤原因
    if filter_reason:
        cv2.putText(overlay_shape, f"Filtered: {filter_reason}", (10, y_offset + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 创建4个子图的对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 原图
    axes[0, 0].imshow(frame_rgb)
    axes[0, 0].set_title('Original Frame', fontsize=12)
    axes[0, 0].axis('off')

    # 分割掩膜叠加
    axes[0, 1].imshow(overlay_seg)
    axes[0, 1].set_title('Segmentation Mask (Green)', fontsize=12)
    axes[0, 1].axis('off')

    # 骨架化掩膜叠加
    axes[1, 0].imshow(overlay_skeleton)
    axes[1, 0].set_title('Skeletonized Mask (Red)', fontsize=12)
    axes[1, 0].axis('off')

    # 拟合椭圆叠加
    axes[1, 1].imshow(overlay_shape)
    if shape_params is not None:
        axes[1, 1].set_title('RANSAC Fitted Ellipse (Blue)', fontsize=12)
    else:
        axes[1, 1].set_title('No Ellipse Detected', fontsize=12)
    axes[1, 1].axis('off')

    # 调整布局
    plt.tight_layout()

    # 将matplotlib图形转换为PIL图像
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    # 转换为RGB（去掉alpha通道）
    comparison_img = Image.fromarray(img_array).convert('RGB')

    return comparison_img


def main():
    parser = argparse.ArgumentParser(description='U-Net ResNet18 Contour Segmentation with RANSAC Ellipse Fitting')

    # 主要输入输出参数
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to folder containing images OR path to video file')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to folder where results will be saved')

    # 模型参数
    parser.add_argument('--model_dir', type=str, default='./checkpoints_contour_unet/result11',
                        help='Path to directory containing model weights')
    parser.add_argument('--encoder_file', type=str, default='resnet18_encoder_contour.pth',
                        help='Filename of encoder weights')
    parser.add_argument('--decoder_file', type=str, default='unet_decoder_contour.pth',
                        help='Filename of decoder weights')

    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')

    # 处理模式参数
    parser.add_argument('--mode', type=str, required=True, choices=['image', 'video'],
                        help='Processing mode: image (process image folder) or video (process video file)')

    # RANSAC参数
    parser.add_argument('--ransac_max_iter', type=int, default=ELLIPSE_RANSAC_MAX_ITER,
                        help='RANSAC maximum iterations')
    parser.add_argument('--ransac_confidence', type=float, default=ELLIPSE_RANSAC_CONFIDENCE,
                        help='RANSAC confidence')
    parser.add_argument('--ransac_threshold', type=float, default=ELLIPSE_RANSAC_REPROJ_THRESHOLD,
                        help='RANSAC reprojection threshold')

    # 处理参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation (default: 0.5)')
    parser.add_argument('--no_comparison', action='store_true',
                        help='Skip creating comparison images (image mode only)')
    parser.add_argument('--fps', type=int, default=None,
                        help='Output video FPS (video mode only, default: use input video FPS)')

    # ==================== 修改的椭圆筛选参数 ====================
    parser.add_argument('--filter_ellipse', action='store_true',
                        help='Filter out bad ellipses based on coverage')
    parser.add_argument('--min_coverage', type=float, default=0.5,
                        help='Minimum coverage threshold for ellipse filtering (default: 0.5)')

    # 新增：平滑参数
    parser.add_argument('--smooth_window', type=int, default=3,
                        help='Window size for ellipse smoothing (default: 5, set to 0 to disable)')

    args = parser.parse_args([
        '--input_path', 'F:/media/video01.mp4',
        '--output_folder', 'F:/result/result11',
        '--mode', 'video',
        '--filter_ellipse',  # 启用椭圆筛选


        '--min_coverage', '0.5',  # 设置覆盖率阈值
    ])

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Using RANSAC ellipse fitting with:")
    print(f"  max_iterations: {args.ransac_max_iter}")
    print(f"  confidence: {args.ransac_confidence}")
    print(f"  reprojection threshold: {args.ransac_threshold}")
    if args.filter_ellipse:
        print(f"Ellipse filtering enabled with min coverage: {args.min_coverage}")

    # Model paths
    encoder_path = os.path.join(args.model_dir, args.encoder_file)
    decoder_path = os.path.join(args.model_dir, args.decoder_file)

    # Load model
    print("Loading model...")
    model = load_contour_model(encoder_path, decoder_path, device)
    print("Model loaded successfully!")

    print(f"Processing mode: {args.mode}")

    if args.mode == 'video':
        # 处理视频
        if not os.path.isfile(args.input_path):
            print(f"Error: Video file not found: {args.input_path}")
            return

        process_video_contour(
            input_video_path=args.input_path,
            output_folder=args.output_folder,
            model=model,
            device=device,
            threshold=args.threshold,
            fps=args.fps,
            shape_type='ellipse',  # 只使用椭圆拟合
            filter_ellipse=args.filter_ellipse,
            min_coverage=args.min_coverage
        )
    else:
        # 处理图片
        if not os.path.isdir(args.input_path):
            print(f"Error: Image folder not found: {args.input_path}")
            return

        inference_on_folder_contour(
            input_folder=args.input_path,
            output_folder=args.output_folder,
            model=model,
            device=device,
            threshold=args.threshold,
            shape_type='ellipse',  # 只使用椭圆拟合
            filter_ellipse=args.filter_ellipse,
            min_coverage=args.min_coverage,
            save_comparison=not args.no_comparison
        )


if __name__ == "__main__":
    main()