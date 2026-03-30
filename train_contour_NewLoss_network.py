import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import cv2
from skimage import measure
from scipy.spatial import KDTree
from skimage.morphology import skeletonize

# 导入障碍物增强函数
import sys
import importlib.util


# 动态导入障碍物增强模块
def import_obstacle_augmentation():
    """动态导入障碍物增强模块"""
    try:
        spec = importlib.util.spec_from_file_location("obstacle_aug", "2_obstacle_argument.py")
        obstacle_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obstacle_module)
        return obstacle_module
    except Exception as e:
        print(f"导入障碍物增强模块失败: {e}")
        return None


obstacle_module = import_obstacle_augmentation()


# -------------------- 新增：详细指标计算函数 --------------------
def dice_coefficient(pred_bin, true_bin):
    """计算两个二值图的Dice系数"""
    inter = np.logical_and(pred_bin, true_bin).sum()
    union = pred_bin.sum() + true_bin.sum()
    if union == 0:
        return 1.0
    return 2.0 * inter / union


def average_contour_distance(pred_bin, true_bin):
    """
    计算预测轮廓到真实轮廓的平均距离（Chamfer距离）
    输入均为二值图（0/1），且应为骨架或轮廓线（非填充）
    """
    pred_points = np.column_stack(np.where(pred_bin > 0))
    true_points = np.column_stack(np.where(true_bin > 0))
    if len(pred_points) == 0 or len(true_points) == 0:
        return np.nan
    tree = KDTree(true_points)
    distances, _ = tree.query(pred_points)
    return np.mean(distances)


def count_components(bin_img, min_area=5):
    """统计二值图中的连通组件个数（忽略面积小于min_area的噪声）"""
    labeled = measure.label(bin_img, connectivity=2)
    props = measure.regionprops(labeled)
    comps = [p for p in props if p.area >= min_area]
    return len(comps)


def dilate(bin_img, kernel_size=3, iterations=1):
    """对二值图进行膨胀"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(bin_img, kernel, iterations=iterations)


# ---------------------------------------------------------------

class ContourSegmentationDataset(Dataset):
    """改进的数据集类，专门用于轮廓分割，包含动态障碍物增强"""

    # ... 保持原样 ...

    def __init__(self, txt_file, transform=None, target_transform=None,
                 obstacle_dir=None, obstacle_prob=0.1, is_training=True):
        self.data_pairs = []
        self.transform = transform
        self.is_training = is_training
        self.obstacle_prob = obstacle_prob

        with open(txt_file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path, label_path = line.split()
                    self.data_pairs.append((img_path, label_path))

        self.obstacles = []
        if is_training and obstacle_dir and obstacle_module:
            try:
                self.obstacles = obstacle_module.load_obstacles(obstacle_dir)
                print(f"加载了 {len(self.obstacles)} 个障碍物用于动态增强")
            except Exception as e:
                print(f"加载障碍物失败: {e}")

    def __len__(self):
        return len(self.data_pairs)

    def _process_contour_label(self, label_img):
        if label_img.mode == 'P':
            label_np = np.array(label_img)
            label_np = (label_np > 0).astype(np.uint8) * 255
            label_fixed = Image.fromarray(label_np, mode='L')
        else:
            label_fixed = label_img.convert('L')
            label_np = np.array(label_fixed)
            if label_np.max() > 1:
                label_np = (label_np > 127).astype(np.uint8) * 255
                label_fixed = Image.fromarray(label_np, mode='L')
        return label_fixed

    def _synchronized_random_scale_and_crop(self, image, label, target_size, scale_range=(0.8, 1.2)):
        target_w, target_h = target_size
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        orig_w, orig_h = image.size
        scaled_w = int(orig_w * scale_factor)
        scaled_h = int(orig_h * scale_factor)
        image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
        label = label.resize((scaled_w, scaled_h), Image.NEAREST)

        if scaled_w < target_w or scaled_h < target_h:
            pad_w = max(0, target_w - scaled_w)
            pad_h = max(0, target_h - scaled_h)
            padded_image = Image.new('RGB', (scaled_w + pad_w, scaled_h + pad_h), (0, 0, 0))
            padded_label = Image.new('L', (scaled_w + pad_w, scaled_h + pad_h), 0)
            paste_x = pad_w // 2
            paste_y = pad_h // 2
            padded_image.paste(image, (paste_x, paste_y))
            padded_label.paste(label, (paste_x, paste_y))
            image = padded_image
            label = padded_label
            scaled_w += pad_w
            scaled_h += pad_h

        if scaled_w > target_w or scaled_h > target_h:
            max_x = max(0, scaled_w - target_w)
            max_y = max(0, scaled_h - target_h)
            left = np.random.randint(0, max_x + 1) if max_x > 0 else 0
            top = np.random.randint(0, max_y + 1) if max_y > 0 else 0
            right = left + target_w
            bottom = top + target_h
            image = image.crop((left, top, right, bottom))
            label = label.crop((left, top, right, bottom))
        else:
            image = image.resize((target_w, target_h), Image.BILINEAR)
            label = label.resize((target_w, target_h), Image.NEAREST)
        return image, label

    def __getitem__(self, idx):
        img_path, label_path = self.data_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        label_original = Image.open(label_path)
        label = self._process_contour_label(label_original)

        if self.is_training and self.obstacles and obstacle_module:
            image, label = obstacle_module.apply_dynamic_obstacle_augmentation(
                image, label, self.obstacles, self.obstacle_prob
            )

        is_training = self.transform and any(
            hasattr(t, 'p') or 'Random' in t.__class__.__name__
            for t in self.transform.transforms
        )

        if is_training:
            image, label = self._synchronized_random_scale_and_crop(
                image, label, target_size=(256, 256), scale_range=(0.50, 1.50)
            )
        else:
            image = image.resize((256, 256), Image.BILINEAR)
            label = label.resize((256, 256), Image.NEAREST)

        if self.transform:
            transform_list = []
            for t in self.transform.transforms:
                if not isinstance(t, transforms.Resize):
                    transform_list.append(t)
            modified_transform = transforms.Compose(transform_list)
            image = modified_transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        label_tensor = transforms.ToTensor()(label)
        label_bin = (label_tensor > 0.5).float()

        # 距离变换图
        label_np = label_bin.squeeze(0).cpu().numpy().astype(np.uint8)
        dist_mask = (1 - label_np) * 255
        dist_np = cv2.distanceTransform(dist_mask, cv2.DIST_L2, 5)
        # 归一化到 [0,1] 防止边界损失过大
        h, w = dist_np.shape
        max_dist = np.sqrt(h ** 2 + w ** 2)
        dist_np = dist_np / max_dist
        dist_tensor = torch.from_numpy(dist_np).float().unsqueeze(0)

        return image, label_bin, dist_tensor


# 损失函数类（保持不变，仅添加了距离归一化，已在数据集中完成）
class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, contour_weight=3.0):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.contour_weight = contour_weight

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        weights = torch.ones_like(target)
        weights[target > 0.5] = self.contour_weight
        intersection = (pred * target * weights).sum()
        pred_sum = (pred * weights).sum()
        target_sum = (target * weights).sum()
        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1 - dice


class ContourFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.5):
        super(ContourFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_factor = torch.ones_like(target) * (1 - self.alpha)
        alpha_factor[target > 0.5] = self.alpha
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class ContourCombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.1, focal_weight=0.3, dice_weight=0.3,
                 boundary_weight=0.2, area_weight=0.1):
        super(ContourCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.area_weight = area_weight
        self.bce_loss = nn.BCELoss()
        self.focal_loss = ContourFocalLoss(alpha=0.8, gamma=2.5)
        self.dice_loss = WeightedDiceLoss(contour_weight=3.0)

    def forward(self, pred, target, distance):
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        boundary_loss = (pred * distance).mean()
        area_loss = torch.abs(pred.sum() - target.sum()) / pred.numel()
        total_loss = (self.bce_weight * bce +
                      self.focal_weight * focal +
                      self.dice_weight * dice +
                      self.boundary_weight * boundary_loss +
                      self.area_weight * area_loss)
        loss_dict = {
            'bce': bce.item(),
            'focal': focal.item(),
            'dice': dice.item(),
            'boundary': boundary_loss.item(),
            'area': area_loss.item()
        }
        return total_loss, loss_dict


# 网络结构（保持不变）
class ResNetEncoder(nn.Module):
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
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNetResNet18(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(UNetResNet18, self).__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128, 64, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)
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


# 可视化函数（保持不变）
def visualize_predictions(images, targets, predictions, epoch, save_dir, num_samples=4):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    for i in range(min(num_samples, images.size(0))):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i + 1} - Input')
        axes[i, 0].axis('off')
        target_np = targets[i, 0].cpu().numpy()
        axes[i, 1].imshow(target_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        pred_np = predictions[i, 0].cpu().detach().numpy()
        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'predictions_epoch_{epoch + 1}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, save_dir):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    loss_components = {'bce': 0.0, 'focal': 0.0, 'dice': 0.0, 'boundary': 0.0, 'area': 0.0}
    vis_images, vis_targets, vis_predictions = [], [], []
    with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") as pbar:
        for batch_idx, (images, targets, distances) in enumerate(pbar):
            images, targets, distances = images.to(device), targets.to(device), distances.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets, distances)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] += v
            batch_iou, batch_dice = 0.0, 0.0
            for i in range(outputs.size(0)):
                iou, dice = calculate_metrics(outputs[i], targets[i])
                batch_iou += iou
                batch_dice += dice
            batch_iou /= outputs.size(0)
            batch_dice /= outputs.size(0)
            total_iou += batch_iou
            total_dice += batch_dice
            if batch_idx == 0 and not vis_images:
                num_samples = min(4, images.size(0))
                vis_images.append(images[:num_samples].detach())
                vis_targets.append(targets[:num_samples].detach())
                vis_predictions.append(outputs[:num_samples].detach())
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{batch_iou:.4f}', 'Dice': f'{batch_dice:.4f}'})
    if vis_images:
        vis_images = torch.cat(vis_images, dim=0)
        vis_targets = torch.cat(vis_targets, dim=0)
        vis_predictions = torch.cat(vis_predictions, dim=0)
        visualize_predictions(vis_images, vis_targets, vis_predictions, epoch, os.path.join(save_dir, 'train_vis'))
    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    for k in loss_components:
        loss_components[k] /= len(train_loader)
    return avg_loss, avg_iou, avg_dice, loss_components


def validate_epoch(model, val_loader, criterion, device, epoch, save_dir, compute_detailed=False):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    loss_components = {'bce': 0.0, 'focal': 0.0, 'dice': 0.0, 'boundary': 0.0, 'area': 0.0}
    vis_images, vis_targets, vis_predictions = [], [], []

    # 详细指标累加器
    detailed_metrics = {
        'dice_tol': 0.0,
        'avg_dist': 0.0,
        'area_ratio': 0.0,
        'pred_comps': 0.0,
        'true_comps': 0.0,
        'composite_score': 0.0,
        'count': 0
    }

    with torch.no_grad():
        with tqdm(val_loader,
                  desc=f"Validation Epoch {epoch + 1}" + (" (Detailed)" if compute_detailed else "")) as pbar:
            for batch_idx, (images, targets, distances) in enumerate(pbar):
                images, targets, distances = images.to(device), targets.to(device), distances.to(device)
                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets, distances)
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_components[k] += v
                # 计算常规指标
                batch_iou, batch_dice = 0.0, 0.0
                for i in range(outputs.size(0)):
                    iou, dice = calculate_metrics(outputs[i], targets[i])
                    batch_iou += iou
                    batch_dice += dice
                batch_iou /= outputs.size(0)
                batch_dice /= outputs.size(0)
                total_iou += batch_iou
                total_dice += batch_dice

                # 如果要求详细指标，对每个样本进行后处理计算
                if compute_detailed:
                    preds_np = outputs.cpu().numpy()  # (B,1,H,W)
                    targets_np = targets.cpu().numpy()  # (B,1,H,W)
                    for i in range(preds_np.shape[0]):
                        pred_bin = (preds_np[i, 0] > 0.5).astype(np.uint8)
                        true_bin = (targets_np[i, 0] > 0.5).astype(np.uint8)

                        # 骨架化（用于距离计算）
                        if pred_bin.sum() > 0:
                            pred_skeleton = skeletonize(pred_bin).astype(np.uint8)
                        else:
                            pred_skeleton = pred_bin
                        if true_bin.sum() > 0:
                            true_skeleton = skeletonize(true_bin).astype(np.uint8)
                        else:
                            true_skeleton = true_bin

                        # 膨胀容忍 Dice
                        pred_dilated = dilate(pred_bin, kernel_size=3, iterations=1)
                        true_dilated = dilate(true_bin, kernel_size=3, iterations=1)
                        dice_tol = dice_coefficient(pred_dilated, true_dilated)

                        # 平均距离
                        avg_dist = average_contour_distance(pred_skeleton, true_skeleton)
                        if np.isnan(avg_dist):
                            avg_dist = 100.0  # 惩罚无轮廓的情况

                        # 面积比
                        area_pred = pred_bin.sum()
                        area_true = true_bin.sum()
                        area_ratio = area_pred / (area_true + 1e-6)

                        # 连通组件
                        pred_comps = count_components(pred_skeleton)
                        true_comps = count_components(true_skeleton)

                        # 综合得分（越大越好）
                        comp_penalty = max(0, pred_comps - 2)
                        composite_score = - (0.5 * (0.2 - dice_tol) + 0.3 * (avg_dist - 5) + 0.1 * (
                                    1.3 - area_ratio) + 0.1 * comp_penalty)

                        detailed_metrics['dice_tol'] += dice_tol
                        detailed_metrics['avg_dist'] += avg_dist
                        detailed_metrics['area_ratio'] += area_ratio
                        detailed_metrics['pred_comps'] += pred_comps
                        detailed_metrics['true_comps'] += true_comps
                        detailed_metrics['composite_score'] += composite_score
                        detailed_metrics['count'] += 1

                if batch_idx == 0 and not vis_images:
                    num_samples = min(4, images.size(0))
                    vis_images.append(images[:num_samples].detach())
                    vis_targets.append(targets[:num_samples].detach())
                    vis_predictions.append(outputs[:num_samples].detach())

                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{batch_iou:.4f}', 'Dice': f'{batch_dice:.4f}'})

    # 可视化
    if vis_images:
        vis_images = torch.cat(vis_images, dim=0)
        vis_targets = torch.cat(vis_targets, dim=0)
        vis_predictions = torch.cat(vis_predictions, dim=0)
        visualize_predictions(vis_images, vis_targets, vis_predictions, epoch, os.path.join(save_dir, 'val_vis'))

    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    for k in loss_components:
        loss_components[k] /= len(val_loader)

    if compute_detailed and detailed_metrics['count'] > 0:
        cnt = detailed_metrics['count']
        detailed_avg = {
            'dice_tol': detailed_metrics['dice_tol'] / cnt,
            'avg_dist': detailed_metrics['avg_dist'] / cnt,
            'area_ratio': detailed_metrics['area_ratio'] / cnt,
            'pred_comps': detailed_metrics['pred_comps'] / cnt,
            'true_comps': detailed_metrics['true_comps'] / cnt,
            'composite_score': detailed_metrics['composite_score'] / cnt,
        }
        return avg_loss, avg_iou, avg_dice, loss_components, detailed_avg
    else:
        return avg_loss, avg_iou, avg_dice, loss_components, None


def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    union = pred_sum + target_sum - intersection
    if union > 0:
        iou = intersection / union
    else:
        iou = torch.tensor(1.0 if intersection == 0 else 0.0)
    if pred_sum + target_sum > 0:
        dice = (2.0 * intersection) / (pred_sum + target_sum)
    else:
        dice = torch.tensor(1.0 if intersection == 0 else 0.0)
    return iou.item(), dice.item()


def plot_training_history(history, save_dir):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    # 损失
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch');
    axes[0, 0].set_ylabel('Loss');
    axes[0, 0].legend();
    axes[0, 0].grid(True)
    # IoU
    axes[0, 1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[0, 1].set_title('IoU');
    axes[0, 1].set_xlabel('Epoch');
    axes[0, 1].legend();
    axes[0, 1].grid(True)
    # Dice
    axes[0, 2].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[0, 2].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[0, 2].set_title('Dice');
    axes[0, 2].set_xlabel('Epoch');
    axes[0, 2].legend();
    axes[0, 2].grid(True)
    # 训练损失分量
    axes[1, 0].plot(epochs, history['train_bce'], 'g-', label='BCE')
    axes[1, 0].plot(epochs, history['train_focal'], 'r-', label='Focal')
    axes[1, 0].plot(epochs, history['train_dice_loss'], 'b-', label='Dice')
    axes[1, 0].plot(epochs, history['train_boundary'], 'm-', label='Boundary')
    axes[1, 0].plot(epochs, history['train_area'], 'c-', label='Area')
    axes[1, 0].set_title('Train Loss Components');
    axes[1, 0].legend();
    axes[1, 0].grid(True)
    # 验证损失分量
    axes[1, 1].plot(epochs, history['val_bce'], 'g-', label='BCE')
    axes[1, 1].plot(epochs, history['val_focal'], 'r-', label='Focal')
    axes[1, 1].plot(epochs, history['val_dice_loss'], 'b-', label='Dice')
    axes[1, 1].plot(epochs, history['val_boundary'], 'm-', label='Boundary')
    axes[1, 1].plot(epochs, history['val_area'], 'c-', label='Area')
    axes[1, 1].set_title('Val Loss Components');
    axes[1, 1].legend();
    axes[1, 1].grid(True)
    # 学习率
    axes[1, 2].plot(epochs, history['lr'], 'm-')
    axes[1, 2].set_title('Learning Rate');
    axes[1, 2].set_xlabel('Epoch');
    axes[1, 2].grid(True)
    # 如果有详细指标，也可以画（可选），这里留空
    axes[0, 3].axis('off');
    axes[1, 3].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'contour_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='轮廓分割训练')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_txt', type=str, default='dataset_combined/train.txt')
    parser.add_argument('--val_txt', type=str, default='dataset_combined/val.txt')
    parser.add_argument('--obstacle_dir', type=str, default='obstacle/img')
    parser.add_argument('--obstacle_prob', type=float, default=0.2)
    parser.add_argument('--bce_weight', type=float, default=0.1)
    parser.add_argument('--focal_weight', type=float, default=0.3)
    parser.add_argument('--dice_weight', type=float, default=0.3)
    parser.add_argument('--boundary_weight', type=float, default=0.2)
    parser.add_argument('--area_weight', type=float, default=0.1)
    parser.add_argument('--detailed_freq', type=int, default=5, help='每多少个epoch计算一次详细指标')
    args = parser.parse_args()

    config = {
        'train_txt': args.train_txt,
        'val_txt': args.val_txt,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'save_dir': './checkpoints_contour_unet_NewLoss',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': args.num_workers,
        'pin_memory': True,
        'save_freq': args.save_freq,
        'vis_freq': 5,
        'obstacle_dir': args.obstacle_dir,
        'obstacle_prob': args.obstacle_prob,
        'bce_weight': args.bce_weight,
        'focal_weight': args.focal_weight,
        'dice_weight': args.dice_weight,
        'boundary_weight': args.boundary_weight,
        'area_weight': args.area_weight,
        'detailed_freq': args.detailed_freq,
    }
    os.makedirs(config['save_dir'], exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("加载训练数据集...")
    train_dataset = ContourSegmentationDataset(
        config['train_txt'], transform=train_transform,
        obstacle_dir=config['obstacle_dir'], obstacle_prob=config['obstacle_prob'], is_training=True
    )
    print("加载验证数据集...")
    val_dataset = ContourSegmentationDataset(
        config['val_txt'], transform=val_transform,
        obstacle_dir=None, obstacle_prob=0.0, is_training=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=config['pin_memory'])

    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")

    model = UNetResNet18(num_classes=1, pretrained=True).to(config['device'])
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)

    criterion = ContourCombinedLoss(
        bce_weight=config['bce_weight'], focal_weight=config['focal_weight'],
        dice_weight=config['dice_weight'], boundary_weight=config['boundary_weight'],
        area_weight=config['area_weight']
    )
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_bce': [], 'train_focal': [], 'train_dice_loss': [], 'train_boundary': [], 'train_area': [],
        'val_bce': [], 'val_focal': [], 'val_dice_loss': [], 'val_boundary': [], 'val_area': [],
        'lr': []
    }
    # 详细指标历史
    detailed_history = {
        'epoch': [], 'dice_tol': [], 'avg_dist': [], 'area_ratio': [], 'pred_comps': [], 'true_comps': [],
        'composite_score': []
    }

    best_val_dice = 0.0
    best_composite_score = -float('inf')
    start_time = time.time()

    try:
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print("-" * 50)

            # 训练
            train_loss, train_iou, train_dice, train_loss_comp = train_epoch(
                model, train_loader, criterion, optimizer, config['device'], epoch, config['save_dir']
            )

            # 常规验证（无详细指标）
            val_loss, val_iou, val_dice, val_loss_comp, _ = validate_epoch(
                model, val_loader, criterion, config['device'], epoch, config['save_dir'], compute_detailed=False
            )

            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]['lr']

            # 更新历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_iou'].append(train_iou)
            history['val_iou'].append(val_iou)
            history['train_dice'].append(train_dice)
            history['val_dice'].append(val_dice)
            history['train_bce'].append(train_loss_comp['bce'])
            history['train_focal'].append(train_loss_comp['focal'])
            history['train_dice_loss'].append(train_loss_comp['dice'])
            history['train_boundary'].append(train_loss_comp['boundary'])
            history['train_area'].append(train_loss_comp['area'])
            history['val_bce'].append(val_loss_comp['bce'])
            history['val_focal'].append(val_loss_comp['focal'])
            history['val_dice_loss'].append(val_loss_comp['dice'])
            history['val_boundary'].append(val_loss_comp['boundary'])
            history['val_area'].append(val_loss_comp['area'])
            history['lr'].append(current_lr)

            print(f"训练 - 损失: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
            print(f"验证 - 损失: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            print(f"学习率: {current_lr:.2e}")

            # 每隔 detailed_freq 个 epoch 计算详细指标
            if (epoch + 1) % config['detailed_freq'] == 0:
                print("计算详细验证指标...")
                _, _, _, _, detailed = validate_epoch(
                    model, val_loader, criterion, config['device'], epoch, config['save_dir'], compute_detailed=True
                )
                if detailed is not None:
                    detailed_history['epoch'].append(epoch + 1)
                    detailed_history['dice_tol'].append(detailed['dice_tol'])
                    detailed_history['avg_dist'].append(detailed['avg_dist'])
                    detailed_history['area_ratio'].append(detailed['area_ratio'])
                    detailed_history['pred_comps'].append(detailed['pred_comps'])
                    detailed_history['true_comps'].append(detailed['true_comps'])
                    detailed_history['composite_score'].append(detailed['composite_score'])

                    print(
                        f"详细指标 - dice_tol: {detailed['dice_tol']:.4f}, avg_dist: {detailed['avg_dist']:.2f}, area_ratio: {detailed['area_ratio']:.2f}, pred_comps: {detailed['pred_comps']:.1f}, composite_score: {detailed['composite_score']:.4f}")

                    # 基于综合得分保存最佳模型
                    if detailed['composite_score'] > best_composite_score:
                        best_composite_score = detailed['composite_score']
                        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                        torch.save(model_to_save.encoder.state_dict(),
                                   os.path.join(config['save_dir'], 'resnet18_encoder_contour_best_composite.pth'))
                        decoder_state = {
                            'decoder4': model_to_save.decoder4.state_dict(),
                            'decoder3': model_to_save.decoder3.state_dict(),
                            'decoder2': model_to_save.decoder2.state_dict(),
                            'decoder1': model_to_save.decoder1.state_dict(),
                            'final_upconv': model_to_save.final_upconv.state_dict(),
                            'final_conv': model_to_save.final_conv.state_dict()
                        }
                        torch.save(decoder_state,
                                   os.path.join(config['save_dir'], 'unet_decoder_contour_best_composite.pth'))
                        print(f"新的最佳综合得分模型已保存! 得分: {best_composite_score:.4f}")

            # 仍然保存基于 Dice 的最佳模型（可选）
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.encoder.state_dict(),
                           os.path.join(config['save_dir'], 'resnet18_encoder_contour.pth'))
                decoder_state = {
                    'decoder4': model_to_save.decoder4.state_dict(),
                    'decoder3': model_to_save.decoder3.state_dict(),
                    'decoder2': model_to_save.decoder2.state_dict(),
                    'decoder1': model_to_save.decoder1.state_dict(),
                    'final_upconv': model_to_save.final_upconv.state_dict(),
                    'final_conv': model_to_save.final_conv.state_dict()
                }
                torch.save(decoder_state, os.path.join(config['save_dir'], 'unet_decoder_contour.pth'))
                print(f"新的最佳Dice模型已保存! Dice: {best_val_dice:.4f}")

            # 定期保存检查点
            if (epoch + 1) % config['save_freq'] == 0:
                checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model,
                                                                                nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'detailed_history': detailed_history,
                    'config': config,
                    'best_val_dice': best_val_dice,
                    'best_composite_score': best_composite_score
                }, checkpoint_path)
                print(f"检查点已保存: {checkpoint_path}")

            if current_lr < 1e-7:
                print("学习率过小，停止训练。")
                break

    except KeyboardInterrupt:
        print("\n训练被用户中断。")

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"总训练时间: {total_time / 3600:.2f} 小时")
    print(f"最佳验证Dice: {best_val_dice:.4f}")
    print(f"最佳综合得分: {best_composite_score:.4f}")

    plot_training_history(history, config['save_dir'])
    # 可选：绘制详细指标曲线
    if detailed_history['epoch']:
        plt.figure()
        plt.plot(detailed_history['epoch'], detailed_history['composite_score'], 'b-o')
        plt.xlabel('Epoch')
        plt.ylabel('Composite Score')
        plt.title('Composite Score over Epochs')
        plt.grid(True)
        plt.savefig(os.path.join(config['save_dir'], 'composite_score.png'), dpi=150)
        plt.close()

    # 保存最终模型
    final_model_path = os.path.join(config['save_dir'], 'final_contour_model.pth')
    torch.save({
        'epoch': len(history['train_loss']) - 1,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'detailed_history': detailed_history,
        'config': config,
        'best_val_dice': best_val_dice,
        'best_composite_score': best_composite_score
    }, final_model_path)
    print(f"最终模型已保存: {final_model_path}")


if __name__ == "__main__":
    main()