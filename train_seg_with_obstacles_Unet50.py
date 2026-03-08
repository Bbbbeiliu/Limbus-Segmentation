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
import random

# -------------------- 障碍物增强函数 --------------------
def load_obstacles(obstacle_dir):
    """
    从指定目录加载所有 PNG 图像作为障碍物
    返回 PIL Image 列表（保留原始模式，通常为 RGBA）
    """
    obstacles = []
    if not os.path.isdir(obstacle_dir):
        print(f"警告: 障碍物目录不存在 {obstacle_dir}")
        return obstacles
    for fname in os.listdir(obstacle_dir):
        if fname.lower().endswith('.png'):
            path = os.path.join(obstacle_dir, fname)
            try:
                img = Image.open(path).convert('RGBA')  # 统一转为 RGBA 以处理透明度
                obstacles.append(img)
            except Exception as e:
                print(f"加载障碍物 {path} 失败: {e}")
    print(f"成功加载 {len(obstacles)} 个障碍物")
    return obstacles

def apply_dynamic_obstacle_augmentation(image, label, obstacles, prob=0.2):
    """
    以概率 prob 在 image 和 label 上同步粘贴随机障碍物
    障碍物区域对应的 label 像素被置为 0（背景）
    参数:
        image: PIL Image (RGB)
        label: PIL Image (L 模式，二值 0/255)
        obstacles: PIL Image 列表 (RGBA)
        prob: 应用增强的概率
    返回:
        增强后的 image 和 label (仍为 PIL 图像)
    """
    if random.random() > prob or not obstacles:
        return image, label

    # 随机选择一个障碍物
    obs = random.choice(obstacles).copy()  # 复制一份避免修改原图

    # 随机缩放障碍物 (原始图像大小的 0.1 ~ 0.3 倍)
    img_w, img_h = image.size
    scale = random.uniform(0.1, 0.3)
    new_w = max(10, int(img_w * scale))
    new_h = max(10, int(img_h * scale))
    obs = obs.resize((new_w, new_h), Image.BILINEAR)

    # 随机旋转（可选，增加多样性）
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        obs = obs.rotate(angle, expand=True, resample=Image.BICUBIC)

    # 随机粘贴位置 (确保障碍物不超出图像边界)
    max_left = img_w - obs.width
    max_top = img_h - obs.height
    if max_left < 0 or max_top < 0:
        # 如果障碍物太大（缩放后仍可能超过），直接返回原图（或重新缩放）
        return image, label
    left = random.randint(0, max_left)
    top = random.randint(0, max_top)

    # 创建粘贴用的 mask：障碍物的 alpha 通道
    if obs.mode == 'RGBA':
        r, g, b, a = obs.split()
        obs_rgb = Image.merge('RGB', (r, g, b))
        mask = a
    else:
        obs_rgb = obs.convert('RGB')
        mask = Image.new('L', obs.size, 255)  # 不透明，全贴

    # 将障碍物 RGB 粘贴到图像上
    image.paste(obs_rgb, (left, top), mask)

    # 在标签上，将障碍物覆盖的区域置 0 (背景)
    label_patch = Image.new('L', (obs.width, obs.height), 0)
    label.paste(label_patch, (left, top), mask)

    return image, label


class SegmentationDataset(Dataset):
    """Dataset class for loading paired images and segmentation masks with palette mode support
       新增障碍物增强功能（仅训练时使用）
    """

    def __init__(self, txt_file, transform=None, target_transform=None,
                 obstacle_dir=None, obstacle_prob=0.1, is_training=True):
        """
        Args:
            txt_file (str): Path to txt file with image and label paths
            transform: Transform to apply to images
            target_transform: Transform to apply to labels (deprecated, handled internally)
            obstacle_dir: 障碍物图像目录（仅训练时使用）
            obstacle_prob: 应用障碍物增强的概率
            is_training: 是否为训练模式，训练时才应用障碍物增强
        """
        self.data_pairs = []
        self.transform = transform
        self.is_training = is_training
        self.obstacle_prob = obstacle_prob

        # Read the txt file
        with open(txt_file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path, label_path = line.split()
                    self.data_pairs.append((img_path, label_path))

        # 加载障碍物图像（仅在训练时）
        self.obstacles = []
        if is_training and obstacle_dir:
            self.obstacles = load_obstacles(obstacle_dir)

    def _fix_palette_label(self, label_img):
        """
        统一将各种格式的掩膜转换为二值图像（前景255，背景0）
        支持：调色板模式、灰度图、0-1二值图、0-255灰度图
        """
        if label_img.mode == 'P':
            label_np = np.array(label_img)
        else:
            label_img = label_img.convert('L')
            label_np = np.array(label_img)

        label_np = (label_np > 0).astype(np.uint8) * 255
        return Image.fromarray(label_np, mode='L')

    def __len__(self):
        return len(self.data_pairs)

    def _synchronized_random_scale_and_crop(self, image, label, target_size, scale_range=(0.8, 1.2)):
        """Apply synchronized random scaling (zoom) and random crop to both image and label"""
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

    def _synchronized_random_crop(self, image, label, crop_size):
        """Apply synchronized random crop to both image and label (fallback method)"""
        w, h = image.size
        crop_w, crop_h = crop_size

        if w < crop_w or h < crop_h:
            scale = max(crop_w / w, crop_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            label = label.resize((new_w, new_h), Image.NEAREST)
            w, h = new_w, new_h

        left = np.random.randint(0, w - crop_w + 1)
        top = np.random.randint(0, h - crop_h + 1)
        right = left + crop_w
        bottom = top + crop_h
        image = image.crop((left, top, right, bottom))
        label = label.crop((left, top, right, bottom))

        return image, label

    def __getitem__(self, idx):
        img_path, label_path = self.data_pairs[idx]

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 加载标签并二值化
        label_original = Image.open(label_path)
        label = self._fix_palette_label(label_original)

        # 动态障碍物增强（仅在训练时）
        if self.is_training and self.obstacles:
            image, label = apply_dynamic_obstacle_augmentation(
                image, label, self.obstacles, self.obstacle_prob
            )

        # 判断是否为训练数据（根据 transform 是否含随机操作）
        is_training = self.transform and any(
            hasattr(t, 'p') or 'Random' in t.__class__.__name__
            for t in self.transform.transforms
        )

        if is_training:
            # 训练: 随机缩放和裁剪
            image, label = self._synchronized_random_scale_and_crop(
                image, label,
                target_size=(256, 256),
                scale_range=(0.50, 1.50)
            )
        else:
            # 验证: 直接调整大小
            image = image.resize((256, 256), Image.BILINEAR)
            label = label.resize((256, 256), Image.NEAREST)

        # 应用图像变换（排除 Resize）
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

        # 标签转张量，并二值化为 0/1
        label = transforms.ToTensor()(label)
        label = (label > 0.5).float()

        return image, label


# 以下模型定义、损失函数、训练辅助函数等与原文件完全一致，仅复制保留
# （ResNetEncoder, DecoderBlock, UNetResNet50, DiceLoss, FocalLoss, CombinedLoss,
#  calculate_metrics, train_epoch, validate_epoch, plot_training_history 均不变）

class ResNetEncoder(nn.Module):
    """ResNet-50 encoder for U-Net"""

    def __init__(self, pretrained=True):
        super(ResNetEncoder, self).__init__()

        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

    def forward(self, x):
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 1/2 resolution, 64 channels

        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # 1/4 resolution, 256 channels

        x = self.layer2(x)
        features.append(x)  # 1/8 resolution, 512 channels

        x = self.layer3(x)
        features.append(x)  # 1/16 resolution, 1024 channels

        x = self.layer4(x)
        features.append(x)  # 1/32 resolution, 2048 channels

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


class UNetResNet50(nn.Module):
    """U-Net with ResNet-50 encoder"""

    def __init__(self, num_classes=1, pretrained=True):
        super(UNetResNet50, self).__init__()

        self.encoder = ResNetEncoder(pretrained=pretrained)

        self.decoder4 = DecoderBlock(2048, 1024, 512)   # 1/16 -> 1/8
        self.decoder3 = DecoderBlock(512, 512, 256)     # 1/8 -> 1/4
        self.decoder2 = DecoderBlock(256, 256, 128)     # 1/4 -> 1/2
        self.decoder1 = DecoderBlock(128, 64, 64)       # 1/2 -> 原图

        self.final_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)

        x = self.decoder4(features[4], features[3])  # 2048+1024 -> 512
        x = self.decoder3(x, features[2])            # 512+512   -> 256
        x = self.decoder2(x, features[1])            # 256+256   -> 128
        x = self.decoder1(x, features[0])            # 128+64    -> 64

        x = self.final_upconv(x)                     # 64 -> 32
        x = self.final_conv(x)                        # 32 -> 1
        x = self.sigmoid(x)

        return x


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined BCE, Focal and Dice loss"""

    def __init__(self, bce_weight=0.3, focal_weight=0.3, dice_weight=0.4):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        total_loss = self.bce_weight * bce + self.focal_weight * focal + self.dice_weight * dice
        return total_loss, bce, focal, dice


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
        iou = torch.tensor(0.0)

    if pred_sum + target_sum > 0:
        dice = (2.0 * intersection) / (pred_sum + target_sum)
    else:
        dice = torch.tensor(0.0)

    return iou.item(), dice.item()


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_bce = 0.0
    total_focal = 0.0
    total_dice_loss = 0.0

    with tqdm(train_loader, desc="Training") as pbar:
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss, bce, focal, dice_loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_focal += focal.item()
            total_dice_loss += dice_loss.item()

            batch_iou = 0.0
            batch_dice = 0.0
            for i in range(outputs.size(0)):
                iou, dice = calculate_metrics(outputs[i], targets[i])
                batch_iou += iou
                batch_dice += dice

            batch_iou /= outputs.size(0)
            batch_dice /= outputs.size(0)

            total_iou += batch_iou
            total_dice += batch_dice

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{batch_iou:.4f}',
                'Dice': f'{batch_dice:.4f}'
            })

    n_batches = len(train_loader)
    return (total_loss / n_batches,
            total_iou / n_batches,
            total_dice / n_batches,
            total_bce / n_batches,
            total_focal / n_batches,
            total_dice_loss / n_batches)


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_bce = 0.0
    total_focal = 0.0
    total_dice_loss = 0.0

    with torch.no_grad():
        with tqdm(val_loader, desc="Validation") as pbar:
            for images, targets in pbar:
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)
                loss, bce, focal, dice_loss = criterion(outputs, targets)

                total_loss += loss.item()
                total_bce += bce.item()
                total_focal += focal.item()
                total_dice_loss += dice_loss.item()

                batch_iou = 0.0
                batch_dice = 0.0
                for i in range(outputs.size(0)):
                    iou, dice = calculate_metrics(outputs[i], targets[i])
                    batch_iou += iou
                    batch_dice += dice

                batch_iou /= outputs.size(0)
                batch_dice /= outputs.size(0)

                total_iou += batch_iou
                total_dice += batch_dice

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{batch_iou:.4f}',
                    'Dice': f'{batch_dice:.4f}'
                })

    n_batches = len(val_loader)
    return (total_loss / n_batches,
            total_iou / n_batches,
            total_dice / n_batches,
            total_bce / n_batches,
            total_focal / n_batches,
            total_dice_loss / n_batches)


def plot_training_history(history, save_dir):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[0, 1].set_title('Training and Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[0, 2].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[0, 2].set_title('Training and Validation Dice')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    axes[1, 0].plot(epochs, history['train_bce'], 'g-', label='BCE')
    axes[1, 0].plot(epochs, history['train_focal'], 'r-', label='Focal')
    axes[1, 0].plot(epochs, history['train_dice_loss'], 'b-', label='Dice')
    axes[1, 0].set_title('Training Loss Components')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, history['val_bce'], 'g-', label='BCE')
    axes[1, 1].plot(epochs, history['val_focal'], 'r-', label='Focal')
    axes[1, 1].plot(epochs, history['val_dice_loss'], 'b-', label='Dice')
    axes[1, 1].set_title('Validation Loss Components')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[1, 2].plot(epochs, history['lr'], 'm-')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'contour_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='分割训练（带障碍物增强）')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_freq', type=int, default=10, help='保存频率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--train_txt', type=str, default='dataset_combined/train_seg.txt', help='训练集文件')
    parser.add_argument('--val_txt', type=str, default='dataset_combined/val_seg.txt', help='验证集文件')
    parser.add_argument('--obstacle_dir', type=str, default='obstacle/img', help='障碍物图像目录（PNG）')
    parser.add_argument('--obstacle_prob', type=float, default=0.2, help='障碍物增强概率')
    args = parser.parse_args()

    config = {
        'train_txt': args.train_txt,
        'val_txt': args.val_txt,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'save_dir': './checkpoints_unet_resnet50_obstacle',   # 修改保存目录以区分
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': args.num_workers,
        'pin_memory': True,
        'save_freq': args.save_freq,
        'obstacle_dir': args.obstacle_dir,
        'obstacle_prob': args.obstacle_prob,
    }

    os.makedirs(config['save_dir'], exist_ok=True)

    # Transforms
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

    # Datasets and loaders
    train_dataset = SegmentationDataset(
        config['train_txt'],
        transform=train_transform,
        obstacle_dir=config['obstacle_dir'],
        obstacle_prob=config['obstacle_prob'],
        is_training=True
    )
    val_dataset = SegmentationDataset(
        config['val_txt'],
        transform=val_transform,
        obstacle_dir=None,
        obstacle_prob=0.0,
        is_training=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=config['pin_memory'])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Model setup
    model = UNetResNet50(num_classes=1, pretrained=True)
    model = model.to(config['device'])

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Optimizer and scheduler
    criterion = CombinedLoss(bce_weight=0.3, focal_weight=0.3, dice_weight=0.4)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

    # Extended history dictionary
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_bce': [], 'val_bce': [],
        'train_focal': [], 'val_focal': [],
        'train_dice_loss': [], 'val_dice_loss': [],
        'lr': []
    }

    best_val_dice = 0.0
    start_time = time.time()

    try:
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print("-" * 50)

            train_loss, train_iou, train_dice, train_bce, train_focal, train_dice_loss = train_epoch(
                model, train_loader, criterion, optimizer, config['device'])

            val_loss, val_iou, val_dice, val_bce, val_focal, val_dice_loss = validate_epoch(
                model, val_loader, criterion, config['device'])

            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_iou'].append(train_iou)
            history['val_iou'].append(val_iou)
            history['train_dice'].append(train_dice)
            history['val_dice'].append(val_dice)
            history['train_bce'].append(train_bce)
            history['val_bce'].append(val_bce)
            history['train_focal'].append(train_focal)
            history['val_focal'].append(val_focal)
            history['train_dice_loss'].append(train_dice_loss)
            history['val_dice_loss'].append(val_dice_loss)
            history['lr'].append(current_lr)

            print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print(f"Train Loss Components - BCE: {train_bce:.4f}, Focal: {train_focal:.4f}, Dice: {train_dice_loss:.4f}")
            print(f"Val Loss Components   - BCE: {val_bce:.4f}, Focal: {val_focal:.4f}, Dice: {val_dice_loss:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model

                torch.save(model_to_save.encoder.state_dict(),
                           os.path.join(config['save_dir'], 'resnet50_encoder.pth'))

                decoder_state = {
                    'decoder4': model_to_save.decoder4.state_dict(),
                    'decoder3': model_to_save.decoder3.state_dict(),
                    'decoder2': model_to_save.decoder2.state_dict(),
                    'decoder1': model_to_save.decoder1.state_dict(),
                    'final_upconv': model_to_save.final_upconv.state_dict(),
                    'final_conv': model_to_save.final_conv.state_dict()
                }
                torch.save(decoder_state, os.path.join(config['save_dir'], 'unet_decoder.pth'))

                print(f"New best model saved (encoder and decoder separately) with Dice: {best_val_dice:.4f}!")

            if (epoch + 1) % config['save_freq'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history,
                    'config': config
                }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch + 1}.pth'))
                print(f"Checkpoint saved at epoch {epoch + 1}")

            if current_lr < 1e-7:
                print("Learning rate too small, stopping training.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Total training time: {total_time / 3600:.2f} hours")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Final learning rate: {current_lr:.2e}")

    plot_training_history(history, config['save_dir'])
    print(f"Training history plot saved to {os.path.join(config['save_dir'], 'seg_training_history.png')}")

    final_model_path = os.path.join(config['save_dir'], 'final_model.pth')
    torch.save({
        'epoch': len(history['train_loss']) - 1,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'config': config
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    history_csv_path = os.path.join(config['save_dir'], 'training_history.csv')
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")


if __name__ == "__main__":
    main()