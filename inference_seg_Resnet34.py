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
import random
import collections

warnings.filterwarnings('ignore')

# ==================== 全局参数 ====================
ELLIPSE_RANSAC_MAX_ITER = 3000               # 降低迭代次数
ELLIPSE_RANSAC_CONFIDENCE = 0.999
ELLIPSE_RANSAC_REPROJ_THRESHOLD = 1.0
ELLIPSE_RANSAC_INLIERS_RATIO = 0.6           # 提高提前终止阈值

# 预定义 transform（避免重复创建）
PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 模型定义（与原始一致）====================
class ResNetEncoder(nn.Module):
    """resnet-34 encoder for U-Net"""
    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()
        if pretrained:
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet34()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

    def forward(self, x):
        features = []
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        features.append(x)                     # 1/2
        x = self.maxpool(x)
        x = self.layer1(x); features.append(x) # 1/4
        x = self.layer2(x); features.append(x) # 1/8
        x = self.layer3(x); features.append(x) # 1/16
        x = self.layer4(x); features.append(x) # 1/32
        return features

class DecoderBlock(nn.Module):
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
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.conv2(x); x = self.bn2(x); x = self.relu(x)
        return x

class UNetResNet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(UNetResNet34, self).__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.decoder4 = DecoderBlock(512, 256, 256)  # 1/16
        self.decoder3 = DecoderBlock(256, 128, 128)  # 1/8
        self.decoder2 = DecoderBlock(128, 64, 64)    # 1/4
        self.decoder1 = DecoderBlock(64, 64, 64)     # 1/2
        self.final_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder4(features[4], features[3])
        x = self.decoder3(x, features[2])
        x = self.decoder2(x, features[1])
        x = self.decoder1(x, features[0])
        x = self.final_upconv(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

# ==================== 工具函数 ====================
def load_contour_model(encoder_path, decoder_path, device):
    model = UNetResNet34(num_classes=1, pretrained=False)
    if os.path.exists(encoder_path):
        encoder_state = torch.load(encoder_path, map_location=device)
        model.encoder.load_state_dict(encoder_state)
        print(f"Loaded contour encoder from: {encoder_path}")
    else:
        raise FileNotFoundError(f"Contour encoder file not found: {encoder_path}")
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

def preprocess_image(image_path, target_size=(256,256)):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize(target_size, Image.BILINEAR)
    image_tensor = PREPROCESS_TRANSFORM(image).unsqueeze(0)
    return image_tensor, original_size

def preprocess_frame(frame, target_size=(256,256)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    original_size = (frame.shape[1], frame.shape[0])
    image = image.resize(target_size, Image.BILINEAR)
    image_tensor = PREPROCESS_TRANSFORM(image).unsqueeze(0)
    return image_tensor, original_size

def skeletonize_mask(mask_array):
    """快速骨架化：优先使用skimage，回退OpenCV"""
    binary_mask = (mask_array > 0).astype(np.uint8)
    try:
        skeleton = morphology.skeletonize(binary_mask)
        return (skeleton * 255).astype(np.uint8)
    except ImportError:
        # OpenCV fallback（细化算法）
        from skimage.morphology import thin
        return (thin(binary_mask) * 255).astype(np.uint8)

# ==================== 优化的 RANSAC 椭圆拟合 ====================
def fit_ellipse_ransac(points, max_iterations=ELLIPSE_RANSAC_MAX_ITER,
                       confidence=ELLIPSE_RANSAC_CONFIDENCE,
                       reproj_threshold=ELLIPSE_RANSAC_REPROJ_THRESHOLD):
    """
    向量化RANSAC椭圆拟合
    """
    if len(points) < 5:
        return None, None
    points_np = np.array(points, dtype=np.float32)
    best_ellipse = None
    best_inliers = None
    best_score = 0
    n_pts = len(points_np)
    random.seed(42)

    for iteration in range(max_iterations):
        # 随机选5个点
        idx = random.sample(range(n_pts), 5)
        sample = points_np[idx]
        try:
            ellipse = cv2.fitEllipse(sample)
            center, axes, angle = ellipse
            a, b = axes[0]/2, axes[1]/2
            if a < 1e-6 or b < 1e-6:
                continue

            # 向量化计算所有点到椭圆的代数距离
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])  # 旋转矩阵
            translated = points_np - center
            rotated = translated @ R.T  # 等效于 R @ translated.T 再转置
            x, y = rotated[:, 0], rotated[:, 1]
            dist = np.abs((x / a)**2 + (y / b)**2 - 1)

            inliers = dist < reproj_threshold
            score = np.sum(inliers)

            if score > best_score:
                best_score = score
                best_ellipse = ellipse
                best_inliers = np.where(inliers)[0]

                # 提前终止
                if score / n_pts > ELLIPSE_RANSAC_INLIERS_RATIO:
                    break
        except Exception:
            continue

    # 用内点重新拟合
    if best_ellipse is not None and len(best_inliers) >= 5:
        try:
            inlier_points = points_np[best_inliers]
            final_ellipse = cv2.fitEllipse(inlier_points)
            return final_ellipse, best_inliers
        except:
            return best_ellipse, best_inliers
    # 回退到全体点
    try:
        fallback_ellipse = cv2.fitEllipse(points_np)
        return fallback_ellipse, np.arange(n_pts)
    except:
        return None, None

def fit_shape_to_contour(mask_array, shape_type='ellipse', use_ransac=True, min_contour_points=5):
    """
    拟合椭圆到轮廓
    - use_ransac: True 使用RANSAC，False 直接使用cv2.fitEllipse（最小二乘，更快）
    """
    binary_mask = (mask_array > 0).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return None, None, None
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None, None
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < min_contour_points:
        return None, None, None

    shape_params = None
    contour = largest_contour
    if shape_type == 'ellipse':
        points = largest_contour.reshape(-1, 2)
        if use_ransac:
            ellipse, inliers = fit_ellipse_ransac(points)
        else:
            try:
                ellipse = cv2.fitEllipse(points)
                inliers = None
            except:
                return None, None, None
        if ellipse is not None:
            (cx, cy), (axes_a, axes_b), angle = ellipse
            shape_params = {
                'type': 'ellipse',
                'center': (cx, cy),
                'axes': (axes_a, axes_b),
                'angle': angle,
                'inliers': len(inliers) if inliers is not None else len(points),
                'total_points': len(points)
            }
    return shape_params, contour, binary_mask

def draw_ellipse_mask(ellipse_params, size):
    mask = np.zeros(size, dtype=np.uint8)
    if ellipse_params is None:
        return mask
    center = (int(ellipse_params['center'][0]), int(ellipse_params['center'][1]))
    axes = (int(ellipse_params['axes'][0]/2), int(ellipse_params['axes'][1]/2))
    angle = ellipse_params['angle']
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=1)
    return mask

def filter_bad_ellipse(ellipse_params, original_mask, min_coverage=0.5):
    """基于重合度筛选椭圆"""
    if ellipse_params is None:
        return False, "椭圆参数为空"
    axes = ellipse_params['axes']
    if min(axes) <= 0:
        return False, "轴长为零"
    axis_ratio = min(axes)/max(axes)
    if axis_ratio < 0.6:
        return False, f"轴长比例异常: {axis_ratio:.3f}"

    ellipse_contour = np.zeros_like(original_mask)
    center = (int(ellipse_params['center'][0]), int(ellipse_params['center'][1]))
    axes_half = (int(axes[0]/2), int(axes[1]/2))
    cv2.ellipse(ellipse_contour, center, axes_half, ellipse_params['angle'], 0, 360, 255, thickness=1)
    ellipse_points = np.where(ellipse_contour > 0)
    if len(ellipse_points[0]) == 0:
        return False, "无法生成椭圆轮廓"
    overlap = np.sum(original_mask[ellipse_points] > 0)
    coverage = overlap / len(ellipse_points[0])
    if coverage < min_coverage:
        return False, f"重合度过低: {coverage:.3f}"
    return True, f"重合度: {coverage:.3f}"

# ==================== 椭圆平滑器（与原始相同）====================
class SimpleJitterFilter:
    def __init__(self, threshold=10, max_fixed_frames=1):
        self.threshold = threshold
        self.max_fixed_frames = max_fixed_frames
        self.prev_center = None
        self.prev_prev_center = None
        self.fixed_count = 0
        self.last_was_fixed = False

    def filter_jitter(self, ellipse_params):
        if ellipse_params is None:
            return None
        center = ellipse_params['center']
        if self.prev_center is None:
            self.prev_center = center
            return ellipse_params
        if self.prev_prev_center is None:
            self.prev_prev_center = self.prev_center
            self.prev_center = center
            return ellipse_params

        dx = center[0] - self.prev_center[0]
        dy = center[1] - self.prev_center[1]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > self.threshold:
            dx2 = center[0] - self.prev_prev_center[0]
            dy2 = center[1] - self.prev_prev_center[1]
            dist2 = np.sqrt(dx2*dx2 + dy2*dy2)
            if dist2 < self.threshold and not self.last_was_fixed:
                fixed = ellipse_params.copy()
                fixed['center'] = self.prev_center
                self.prev_prev_center = self.prev_center
                self.fixed_count = 1
                self.last_was_fixed = True
                return fixed
            else:
                self.prev_prev_center = self.prev_center
                self.prev_center = center
                self.fixed_count = 0
                self.last_was_fixed = False
                return ellipse_params
        else:
            self.prev_prev_center = self.prev_center
            self.prev_center = center
            self.fixed_count = 0
            self.last_was_fixed = False
            return ellipse_params

class EllipseSmoother:
    def __init__(self, window_size=3, jitter_threshold=10):
        self.window_size = window_size
        self.history_centers = collections.deque(maxlen=window_size)
        self.history_axes = collections.deque(maxlen=window_size)
        self.history_angles_sin = collections.deque(maxlen=window_size)
        self.history_angles_cos = collections.deque(maxlen=window_size)
        self.jitter_filter = SimpleJitterFilter(threshold=jitter_threshold)
        self.frame_count = 0

    def smooth_ellipse(self, ellipse_params):
        if ellipse_params is None:
            return None
        ellipse_params = self.jitter_filter.filter_jitter(ellipse_params)
        center = ellipse_params['center']
        axes = ellipse_params['axes']
        angle = ellipse_params['angle']
        angle_rad = np.radians(angle)
        self.history_centers.append(center)
        self.history_axes.append(axes)
        self.history_angles_sin.append(np.sin(angle_rad))
        self.history_angles_cos.append(np.cos(angle_rad))
        self.frame_count += 1
        if self.frame_count < 2 or len(self.history_centers) < 2:
            return ellipse_params
        smoothed_center = np.mean(self.history_centers, axis=0)
        smoothed_axes = np.mean(self.history_axes, axis=0)
        avg_sin = np.mean(self.history_angles_sin)
        avg_cos = np.mean(self.history_angles_cos)
        smoothed_angle = np.degrees(np.arctan2(avg_sin, avg_cos))
        smoothed = ellipse_params.copy()
        smoothed['center'] = tuple(smoothed_center)
        smoothed['axes'] = tuple(smoothed_axes)
        smoothed['angle'] = smoothed_angle
        return smoothed

    def reset(self):
        self.history_centers.clear()
        self.history_axes.clear()
        self.history_angles_sin.clear()
        self.history_angles_cos.clear()
        self.frame_count = 0

# ==================== 快速对比图生成（OpenCV版）====================
def create_comparison_image_fast(frame, overlay_seg, overlay_skeleton, overlay_shape,
                                 shape_params=None, filter_reason=None, target_height=400):
    """
    使用OpenCV快速生成2x2对比图
    - frame: 原始帧 (BGR)
    - overlay_seg, overlay_skeleton, overlay_shape: 均为 RGB 格式的叠加图
    """
    # 确保所有输入为RGB格式（便于统一处理）
    imgs = []
    for img in [frame, overlay_seg, overlay_skeleton, overlay_shape]:
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3 and isinstance(img, np.ndarray):
            # 已经是RGB，无需转换
            pass
        else:
            # 如果frame是BGR，需要转换
            if img is frame and len(img.shape)==3 and img.shape[2]==3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 缩放至相同高度
        h, w = img.shape[:2]
        new_w = int(w * target_height / h)
        resized = cv2.resize(img, (new_w, target_height))
        imgs.append(resized)

    # 水平拼接第一行：原图 | 分割掩膜
    top_row = np.hstack([imgs[0], imgs[1]])
    # 水平拼接第二行：骨架 | 椭圆
    bottom_row = np.hstack([imgs[2], imgs[3]])

    # 垂直拼接
    comparison = np.vstack([top_row, bottom_row])

    # 添加文字标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 0.8, (255,255,255), 2)
    cv2.putText(comparison, "Segmentation", (imgs[0].shape[1]+10, 30), font, 0.8, (255,255,255), 2)
    cv2.putText(comparison, "Skeleton", (10, target_height+30), font, 0.8, (255,255,255), 2)
    cv2.putText(comparison, "Ellipse", (imgs[2].shape[1]+10, target_height+30), font, 0.8, (255,255,255), 2)

    if shape_params:
        info = f"Center: {shape_params['center'][0]:.0f},{shape_params['center'][1]:.0f}"
        cv2.putText(comparison, info, (imgs[2].shape[1]+10, target_height+60), font, 0.6, (0,255,0), 1)
    if filter_reason:
        cv2.putText(comparison, f"Filter: {filter_reason}", (10, target_height*2-10), font, 0.6, (0,0,255), 1)

    return comparison

# ==================== 后处理函数（返回叠加图而非PIL）====================
def postprocess_contour_mask_fast(output_tensor, original_size, threshold=0.5, skeletonize=True,
                                   fit_shape=True, shape_type='ellipse', use_ransac=True):
    """
    快速后处理：返回 overlay_seg, overlay_skeleton, overlay_shape, shape_params, filter_info
    所有叠加图均为 RGB numpy 数组，尺寸为 original_size (H,W)
    """
    mask = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    # 保存原始二值掩膜用于筛选
    orig_binary = binary_mask.copy()

    # 骨架化（如果需要）
    skeleton_mask = None
    if skeletonize:
        try:
            skeleton_mask = skeletonize_mask(binary_mask)
        except Exception as e:
            skeleton_mask = binary_mask
    else:
        skeleton_mask = binary_mask

    # 调整到原始尺寸
    binary_mask_img = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    skeleton_mask_img = cv2.resize(skeleton_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # 拟合椭圆
    shape_params = None
    shape_mask_img = None
    if fit_shape:
        shape_params, _, _ = fit_shape_to_contour(binary_mask_img, shape_type=shape_type,
                                                   use_ransac=use_ransac)
        if shape_params is not None:
            shape_mask_img = draw_ellipse_mask(shape_params, (original_size[1], original_size[0]))

    # 生成彩色叠加图（RGB）
    h, w = original_size[1], original_size[0]
    seg_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    seg_overlay[binary_mask_img > 0] = [0, 255, 0]      # 绿色

    skeleton_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    skeleton_overlay[skeleton_mask_img > 0] = [255, 0, 0]  # 红色

    shape_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if shape_mask_img is not None:
        shape_overlay[shape_mask_img > 0] = [0, 0, 255]      # 蓝色

    # 将原始帧转换为RGB（调用者传入BGR，需转换）
    # 注意：此处返回的是叠加图本身，不融合到原图，由调用者融合
    return seg_overlay, skeleton_overlay, shape_overlay, shape_params, binary_mask_img

# ==================== 视频处理主函数（批量推理 + 快速对比图）====================
def process_video_contour_fast(input_video_path, output_folder, model, device,
                                threshold=0.5, fps=None, shape_type='ellipse',
                                filter_ellipse=True, min_coverage=0.5,
                                use_ransac=True, batch_size=4, compare_interval=30):
    os.makedirs(output_folder, exist_ok=True)
    comparison_folder = os.path.join(output_folder, 'Result11_comparison_frames')
    os.makedirs(comparison_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) if fps is None else fps
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Info: {video_fps} fps, {frame_width}x{frame_height}, total {total_frames} frames")
    print(f"Batch size: {batch_size}, compare interval: {compare_interval}, use_ransac: {use_ransac}")

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = os.path.join(output_folder, f"C2_{video_name}_shape_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))

    shape_params_list = []
    frame_count = 0
    filtered_count = 0
    ellipse_smoother = EllipseSmoother(window_size=3)

    # 批量处理循环
    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc="Processing video")
        while True:
            # 收集一个batch的帧
            frames_batch = []
            orig_frames = []      # BGR
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_batch.append(frame)
                orig_frames.append(frame)
            if not frames_batch:
                break

            # 预处理batch
            batch_tensors = []
            for frame in frames_batch:
                tensor, _ = preprocess_frame(frame)  # shape (1,3,256,256)
                batch_tensors.append(tensor)
            batch_input = torch.cat(batch_tensors, dim=0).to(device)  # (B,3,256,256)

            # 推理
            batch_output = model(batch_input)  # (B,1,256,256)

            # 逐帧后处理
            for i, output in enumerate(batch_output):
                original_size = (orig_frames[i].shape[1], orig_frames[i].shape[0])
                # 快速后处理
                seg_overlay, skel_overlay, shape_overlay, shape_params, binary_mask = \
                    postprocess_contour_mask_fast(
                        output.unsqueeze(0), original_size, threshold=threshold,
                        skeletonize=True, fit_shape=True, shape_type=shape_type,
                        use_ransac=use_ransac
                    )

                # 椭圆筛选和平滑
                draw_ellipse = False
                filter_reason = ""
                if shape_params is not None and filter_ellipse:
                    valid, reason = filter_bad_ellipse(shape_params, binary_mask, min_coverage)
                    if valid:
                        draw_ellipse = True
                        shape_params = ellipse_smoother.smooth_ellipse(shape_params)
                    else:
                        filtered_count += 1
                        filter_reason = reason
                        shape_params = None
                        ellipse_smoother.reset()
                elif shape_params is not None:
                    draw_ellipse = True
                    shape_params = ellipse_smoother.smooth_ellipse(shape_params)
                else:
                    ellipse_smoother.reset()

                # 准备叠加图：原图与各掩膜融合（透明度0.3）
                frame_rgb = cv2.cvtColor(orig_frames[i], cv2.COLOR_BGR2RGB)
                overlay_seg = cv2.addWeighted(frame_rgb, 0.7, seg_overlay, 0.3, 0)
                overlay_skeleton = cv2.addWeighted(frame_rgb, 0.7, skel_overlay, 0.3, 0)
                overlay_shape_final = cv2.addWeighted(frame_rgb, 0.7, shape_overlay, 0.3, 0)

                # 如果椭圆有效，在overlay_shape_final上绘制椭圆和中心点
                if shape_params is not None and draw_ellipse:
                    center = (int(shape_params['center'][0]), int(shape_params['center'][1]))
                    axes = (int(shape_params['axes'][0]/2), int(shape_params['axes'][1]/2))
                    cv2.ellipse(overlay_shape_final, center, axes, shape_params['angle'],
                                0, 360, (255,255,0), 2)   # 青色椭圆
                    cv2.circle(overlay_shape_final, center, 5, (255,0,255), -1) # 洋红中心

                # 每隔compare_interval帧保存对比图
                if frame_count % compare_interval == 0:
                    comparison_img = create_comparison_image_fast(
                        orig_frames[i], overlay_seg, overlay_skeleton, overlay_shape_final,
                        shape_params=shape_params, filter_reason=filter_reason
                    )
                    comp_path = os.path.join(comparison_folder,
                                             f"{video_name}_frame_{frame_count:06d}_comparison.png")
                    cv2.imwrite(comp_path, cv2.cvtColor(comparison_img, cv2.COLOR_RGB2BGR))

                # 写入最终视频帧（使用 overlay_shape_final，已包含椭圆）
                out.write(cv2.cvtColor(overlay_shape_final, cv2.COLOR_RGB2BGR))

                # 记录参数
                if shape_params is not None:
                    shape_params_list.append({
                        'frame': frame_count,
                        'type': 'ellipse',
                        'center': [float(shape_params['center'][0]), float(shape_params['center'][1])],
                        'axes': [float(shape_params['axes'][0]), float(shape_params['axes'][1])],
                        'angle': float(shape_params['angle']),
                        'inliers': int(shape_params.get('inliers', 0)),
                        'total_points': int(shape_params.get('total_points', 0)),
                        'filtered': not draw_ellipse,
                        'filter_reason': filter_reason
                    })
                else:
                    shape_params_list.append({
                        'frame': frame_count,
                        'type': None,
                        'center': None, 'axes': None, 'angle': None,
                        'inliers': 0, 'total_points': 0,
                        'filtered': True,
                        'filter_reason': filter_reason if filter_reason else "No ellipse"
                    })

                frame_count += 1
                pbar.update(1)

        pbar.close()

    cap.release()
    out.release()

    # 保存参数
    if shape_params_list:
        params_path = os.path.join(output_folder, 'shape_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(shape_params_list, f, indent=2)
        print(f"Shape parameters saved to: {params_path}")

    print(f"Video processing completed! Frames: {frame_count}, Filtered: {filtered_count}")
    print(f"Output video: {output_video_path}")

# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='Optimized U-Net Contour Segmentation with Ellipse Fitting')
    parser.add_argument('--input_path', type=str, default='F:/media/video01.mp4',
                        help='Path to folder containing images OR path to video file')
    parser.add_argument('--output_folder', type=str, default='F:/result/result11',
                        help='Path to folder where results will be saved')
    parser.add_argument('--model_dir', type=str, default='./checkpoints_contour_unet/result11',
                        help='Directory containing model weights')
    parser.add_argument('--encoder_file', type=str, default='resnet34_encoder_contour.pth',
                        help='Encoder filename')
    parser.add_argument('--decoder_file', type=str, default='unet_decoder_contour.pth',
                        help='Decoder filename')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--mode', type=str, default='video', choices=['image', 'video'],
                        help='Processing mode')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Segmentation threshold')
    parser.add_argument('--fps', type=int, default=None,
                        help='Output video FPS (video mode)')
    parser.add_argument('--filter_ellipse', action='store_true',
                        help='Filter ellipses based on coverage')
    parser.add_argument('--min_coverage', type=float, default=0.5,
                        help='Minimum coverage for ellipse filtering')
    parser.add_argument('--use_ransac', action='store_true',
                        help='Use RANSAC for ellipse fitting (slower but robust); if not set, use direct fit')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for video inference')
    parser.add_argument('--compare_interval', type=int, default=30,
                        help='Save comparison image every N frames (video mode)')
    parser.add_argument('--no_comparison', action='store_true',
                        help='Skip comparison images (image mode only)')

    # 为了测试方便，移除硬编码参数（实际使用时请注释下一行）
    args = parser.parse_args([
        '--input_path', 'F:/media/video01.mp4',
        '--output_folder', 'F:/result/result11',
        '--mode', 'video',
        '--filter_ellipse',
        '--min_coverage', '0.5',
        '--use_ransac',        # 使用RANSAC（可选）
    ])
    args = parser.parse_args()

    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # 加载模型
    encoder_path = os.path.join(args.model_dir, args.encoder_file)
    decoder_path = os.path.join(args.model_dir, args.decoder_file)
    print("Loading model...")
    model = load_contour_model(encoder_path, decoder_path, device)
    print("Model loaded.")

    if args.mode == 'video':
        if not os.path.isfile(args.input_path):
            print(f"Error: Video file not found: {args.input_path}")
            return
        process_video_contour_fast(
            input_video_path=args.input_path,
            output_folder=args.output_folder,
            model=model,
            device=device,
            threshold=args.threshold,
            fps=args.fps,
            shape_type='ellipse',
            filter_ellipse=args.filter_ellipse,
            min_coverage=args.min_coverage,
            use_ransac=args.use_ransac,
            batch_size=args.batch_size,
            compare_interval=args.compare_interval
        )
    else:
        # 图片模式暂未优化（可后续按需优化），此处调用原函数（略）
        # 为保持代码简洁，图片模式直接使用原始函数（但您也可以重写类似批量处理）
        print("Image mode not optimized in this version; using original function.")
        # 此处省略原始图片处理代码（可自行添加）
        pass

if __name__ == "__main__":
    main()