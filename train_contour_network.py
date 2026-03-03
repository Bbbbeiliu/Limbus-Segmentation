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

# 导入障碍物增强函数
import sys
import importlib.util

# 动态导入障碍物增强模块
def import_obstacle_augmentation():
    """动态导入障碍物增强模块"""
    try:
        # 假设 2_obstacle_argument.py 在同一目录下
        spec = importlib.util.spec_from_file_location("obstacle_aug", "2_obstacle_argument.py")
        obstacle_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obstacle_module)
        return obstacle_module
    except Exception as e:
        print(f"导入障碍物增强模块失败: {e}")
        return None

# 导入模块
obstacle_module = import_obstacle_augmentation()

class ContourSegmentationDataset(Dataset):
    """改进的数据集类，专门用于轮廓分割，包含动态障碍物增强"""

    def __init__(self, txt_file, transform=None, target_transform=None,
                 obstacle_dir=None, obstacle_prob=0.1, is_training=True):
        self.data_pairs = []
        self.transform = transform
        self.is_training = is_training
        self.obstacle_prob = obstacle_prob

        # 读取txt文件
        with open(txt_file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path, label_path = line.split()
                    self.data_pairs.append((img_path, label_path))

        # 加载障碍物图像（仅在训练时）
        self.obstacles = []
        if is_training and obstacle_dir and obstacle_module:
            try:
                self.obstacles = obstacle_module.load_obstacles(obstacle_dir)
                print(f"加载了 {len(self.obstacles)} 个障碍物用于动态增强")
            except Exception as e:
                print(f"加载障碍物失败: {e}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.data_pairs)

    def _process_contour_label(self, label_img):
        """处理轮廓标签，确保为二值图像"""
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
        """同步随机缩放和裁剪"""
        target_w, target_h = target_size

        # 随机缩放因子
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])

        orig_w, orig_h = image.size
        scaled_w = int(orig_w * scale_factor)
        scaled_h = int(orig_h * scale_factor)

        # 应用缩放
        image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
        label = label.resize((scaled_w, scaled_h), Image.NEAREST)

        # 如果缩放后图像小于目标尺寸，进行填充
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

        # 随机裁剪到目标尺寸
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

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 加载轮廓标签
        label_original = Image.open(label_path)
        label = self._process_contour_label(label_original)

        # 动态障碍物增强（仅在训练时）
        if self.is_training and self.obstacles and obstacle_module:
            image, label = obstacle_module.apply_dynamic_obstacle_augmentation(
                image, label, self.obstacles, self.obstacle_prob
            )

        # 判断是否为训练数据
        is_training = self.transform and any(
            hasattr(t, 'p') or 'Random' in t.__class__.__name__
            for t in self.transform.transforms
        )

        if is_training:
            # 训练时应用随机变换
            image, label = self._synchronized_random_scale_and_crop(
                image, label,
                target_size=(256, 256),
                scale_range=(0.50, 1.50)
            )
        else:
            # 验证时直接调整大小
            image = image.resize((256, 256), Image.BILINEAR)
            label = label.resize((256, 256), Image.NEAREST)

        # 应用图像变换
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

        # 标签转换为张量
        label = transforms.ToTensor()(label)
        label = (label > 0.5).float()

        return image, label



# 改进的损失函数 - 针对轮廓分割优化
class WeightedDiceLoss(nn.Module):
    """加权Dice损失，增强对轮廓的关注"""

    def __init__(self, smooth=1e-6, contour_weight=3.0):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.contour_weight = contour_weight

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        # 计算权重
        weights = torch.ones_like(target)
        weights[target > 0.5] = self.contour_weight

        intersection = (pred * target * weights).sum()
        pred_sum = (pred * weights).sum()
        target_sum = (target * weights).sum()

        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        return 1 - dice


class ContourFocalLoss(nn.Module):
    """针对轮廓的Focal Loss"""

    def __init__(self, alpha=0.8, gamma=2.5):
        super(ContourFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)

        # 对正样本（轮廓）给予更高权重
        alpha_factor = torch.ones_like(target) * (1 - self.alpha)
        alpha_factor[target > 0.5] = self.alpha

        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class ContourCombinedLoss(nn.Module):
    """针对轮廓分割的联合损失函数"""

    def __init__(self, bce_weight=0.2, focal_weight=0.5, dice_weight=0.3):
        super(ContourCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()
        self.focal_loss = ContourFocalLoss(alpha=0.8, gamma=2.5)
        self.dice_loss = WeightedDiceLoss(contour_weight=3.0)

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)

        total_loss = (self.bce_weight * bce +
                      self.focal_weight * focal +
                      self.dice_weight * dice)

        return total_loss, {'bce': bce.item(), 'focal': focal.item(), 'dice': dice.item()}


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
        self.relu = resnet.relu  # (batch,64,128,128)              第一层：通道数变成64，尺寸减半
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels  (batch,64,64,64)  第二层：maxPool+layer1，通道数不变，尺寸减半
        self.layer2 = resnet.layer2  # 128 channels (batch,128,32,32) 第三层：通道数加倍，尺寸减半
        self.layer3 = resnet.layer3  # 256 channels (batch,256,16,16) 第四层：通道数加倍，尺寸减半
        self.layer4 = resnet.layer4  # 512 channels (batch,512,8,8)   第五层：通道数加倍，尺寸减半

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

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  # 上采样，尺寸加倍（转置卷积）
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)  # 跳跃连接+卷积，尺寸不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # 再次卷积，尺寸不变
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)

        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)  # 检查匹配跳跃连接和上一个特征图

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

    def __init__(self, num_classes=1, pretrained=True):  # 基于预训练设置初始参数
        super(UNetResNet18, self).__init__()

        self.encoder = ResNetEncoder(pretrained=pretrained)

        self.decoder4 = DecoderBlock(512, 256, 256)  # 1/16 通道数减半，尺寸加倍 8->16
        self.decoder3 = DecoderBlock(256, 128, 128)  # 1/8  通道数减半，尺寸加倍 16->32
        self.decoder2 = DecoderBlock(128, 64, 64)  # 1/4  通道数减半，尺寸加倍 32->64
        self.decoder1 = DecoderBlock(64, 64, 64)  # 1/2  通道数不变，尺寸不变 64->128

        self.final_upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 1/1  通道数减半，尺寸不变 128->256
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


# 可视化函数
def visualize_predictions(images, targets, predictions, epoch, save_dir, num_samples=4):
    """可视化预测结果"""
    os.makedirs(save_dir, exist_ok=True)

    # 设置字体避免中文问题
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(min(num_samples, images.size(0))):
        # 原始图像
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Sample {i + 1} - Input')
        axes[i, 0].axis('off')

        # 真实轮廓
        target_np = targets[i, 0].cpu().numpy()
        axes[i, 1].imshow(target_np, cmap='gray')
        axes[i, 1].set_title(f'Sample {i + 1} - Ground Truth')
        axes[i, 1].axis('off')

        # 预测轮廓
        pred_np = predictions[i, 0].cpu().detach().numpy()
        axes[i, 2].imshow(pred_np, cmap='gray')
        axes[i, 2].set_title(f'Sample {i + 1} - Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'predictions_epoch_{epoch + 1}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# 修改训练和验证函数以包含可视化
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, save_dir):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    loss_components = {'bce': 0.0, 'focal': 0.0, 'dice': 0.0}

    # 用于可视化的样本
    vis_images, vis_targets, vis_predictions = [], [], []

    with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}") as pbar:
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] += v

            # 计算指标
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

            # 只在第一个batch且还没有采样时进行采样
            if batch_idx == 0 and not vis_images:
                # 直接取前4个样本
                num_samples = min(4, images.size(0))
                vis_images.append(images[:num_samples].detach())
                vis_targets.append(targets[:num_samples].detach())
                vis_predictions.append(outputs[:num_samples].detach())

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{batch_iou:.4f}',
                'Dice': f'{batch_dice:.4f}'
            })

    # 可视化训练结果
    if vis_images:
        vis_images = torch.cat(vis_images, dim=0)
        vis_targets = torch.cat(vis_targets, dim=0)
        vis_predictions = torch.cat(vis_predictions, dim=0)
        visualize_predictions(vis_images, vis_targets, vis_predictions,
                              epoch, os.path.join(save_dir, 'train_vis'))

    avg_loss = total_loss / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    avg_dice = total_dice / len(train_loader)

    # 平均损失分量
    for k in loss_components:
        loss_components[k] /= len(train_loader)

    return avg_loss, avg_iou, avg_dice, loss_components


def validate_epoch(model, val_loader, criterion, device, epoch, save_dir):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    loss_components = {'bce': 0.0, 'focal': 0.0, 'dice': 0.0}

    vis_images, vis_targets, vis_predictions = [], [], []

    with torch.no_grad():
        with tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}") as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)
                loss, loss_dict = criterion(outputs, targets)

                total_loss += loss.item()
                for k, v in loss_dict.items():
                    loss_components[k] += v

                # 计算指标
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

                if batch_idx == 0 and not vis_images:
                    # 直接取前4个样本
                    num_samples = min(4, images.size(0))
                    vis_images.append(images[:num_samples].detach())
                    vis_targets.append(targets[:num_samples].detach())
                    vis_predictions.append(outputs[:num_samples].detach())

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{batch_iou:.4f}',
                    'Dice': f'{batch_dice:.4f}'
                })

    # 可视化验证结果
    if vis_images:
        vis_images = torch.cat(vis_images, dim=0)
        vis_targets = torch.cat(vis_targets, dim=0)
        vis_predictions = torch.cat(vis_predictions, dim=0)
        visualize_predictions(vis_images, vis_targets, vis_predictions,
                              epoch, os.path.join(save_dir, 'val_vis'))

    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_dice = total_dice / len(val_loader)

    for k in loss_components:
        loss_components[k] /= len(val_loader)

    return avg_loss, avg_iou, avg_dice, loss_components


def main():
    """主训练函数"""
    import argparse
    parser = argparse.ArgumentParser(description='轮廓分割训练')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小(初始16)')
    parser.add_argument('--num_epochs', type=int, default=50 , help='训练轮数（初始100）')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率(初始1e-4)')
    parser.add_argument('--save_freq', type=int, default=10, help='保存频率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载进程数')
    parser.add_argument('--train_txt', type=str, default='/project/segmentation/dataset_combined/train.txt', help='训练集文件')
    parser.add_argument('--val_txt', type=str, default='/project/segmentation/dataset_combined/val.txt', help='验证集文件')
    parser.add_argument('--obstacle_dir', type=str, default='obstacle/img', help='障碍物图像目录')
    parser.add_argument('--obstacle_prob', type=float, default=0.2, help='障碍物增强概率')

    args = parser.parse_args()
    # Configuration
    config = {
        'train_txt': args.train_txt,
        'val_txt': args.val_txt,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'save_dir': './checkpoints_contour_unet_NewTemp',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': args.num_workers,
        'pin_memory': True,
        'save_freq': args.save_freq,
        'vis_freq': 5,
        'obstacle_dir': args.obstacle_dir,
        'obstacle_prob': args.obstacle_prob,
    }
    os.makedirs(config['save_dir'], exist_ok=True)


    # 数据变换
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

    # 数据集和数据加载器
    print("加载训练数据集...")
    train_dataset = ContourSegmentationDataset(
        config['train_txt'],
        transform=train_transform,
        obstacle_dir=config['obstacle_dir'],
        obstacle_prob=config['obstacle_prob'],
        is_training=True
    )

    print("加载验证数据集...")
    val_dataset = ContourSegmentationDataset(
        config['val_txt'],
        transform=val_transform,
        obstacle_dir=None,  # 验证集不使用障碍物增强
        obstacle_prob=0.0,
        is_training=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=config['pin_memory'])

    print(f"训练样本: {len(train_dataset)}")
    print(f"验证样本: {len(val_dataset)}")

    # 模型设置
    model = UNetResNet18(num_classes=1, pretrained=True)
    model = model.to(config['device'])

    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)

    # 优化器和损失函数
    criterion = ContourCombinedLoss(bce_weight=0.2, focal_weight=0.5, dice_weight=0.3)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_bce': [], 'train_focal': [], 'train_dice_loss': [],
        'val_bce': [], 'val_focal': [], 'val_dice_loss': [],
        'lr': []
    }

    best_val_dice = 0.0
    start_time = time.time()

    try:
        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print("-" * 50)

            # 训练
            train_loss, train_iou, train_dice, train_loss_components = train_epoch(
                model, train_loader, criterion, optimizer, config['device'], epoch, config['save_dir']
            )

            # 验证
            val_loss, val_iou, val_dice, val_loss_components = validate_epoch(
                model, val_loader, criterion, config['device'], epoch, config['save_dir']
            )

            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]['lr']

            # 更新历史记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_iou'].append(train_iou)
            history['val_iou'].append(val_iou)
            history['train_dice'].append(train_dice)
            history['val_dice'].append(val_dice)
            history['train_bce'].append(train_loss_components['bce'])
            history['train_focal'].append(train_loss_components['focal'])
            history['train_dice_loss'].append(train_loss_components['dice'])
            history['val_bce'].append(val_loss_components['bce'])
            history['val_focal'].append(val_loss_components['focal'])
            history['val_dice_loss'].append(val_loss_components['dice'])
            history['lr'].append(current_lr)

            print(f"训练 - 损失: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
            print(f"验证 - 损失: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
            print(f"学习率: {current_lr:.2e}")
            print(f"训练损失分量 - BCE: {train_loss_components['bce']:.4f}, "
                  f"Focal: {train_loss_components['focal']:.4f}, "
                  f"Dice: {train_loss_components['dice']:.4f}")

            # 保存最佳模型
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model

                # 保存编码器
                torch.save(model_to_save.encoder.state_dict(),
                           os.path.join(config['save_dir'], 'resnet18_encoder_contour.pth'))

                # 保存解码器
                decoder_state = {
                    'decoder4': model_to_save.decoder4.state_dict(),
                    'decoder3': model_to_save.decoder3.state_dict(),
                    'decoder2': model_to_save.decoder2.state_dict(),
                    'decoder1': model_to_save.decoder1.state_dict(),
                    'final_upconv': model_to_save.final_upconv.state_dict(),
                    'final_conv': model_to_save.final_conv.state_dict()
                }
                torch.save(decoder_state, os.path.join(config['save_dir'], 'unet_decoder_contour.pth'))

                print(f"新的最佳模型已保存! Dice: {best_val_dice:.4f}")

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
                    'config': config,
                    'best_val_dice': best_val_dice
                }, checkpoint_path)
                print(f"检查点已保存: {checkpoint_path}")

            # 学习率过小时停止训练
            if current_lr < 1e-7:
                print("学习率过小，停止训练。")
                break

    except KeyboardInterrupt:
        print("\n训练被用户中断。")

    total_time = time.time() - start_time
    print("\n" + "=" * 50)
    print("训练完成!")
    print("=" * 50)
    print(f"总训练时间: {total_time / 3600:.2f} 小时")
    print(f"最佳验证Dice: {best_val_dice:.4f}")

    # 绘制训练历史
    plot_training_history(history, config['save_dir'])

    # 保存最终模型
    final_model_path = os.path.join(config['save_dir'], 'final_contour_model.pth')
    torch.save({
        'epoch': len(history['train_loss']) - 1,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'config': config,
        'best_val_dice': best_val_dice
    }, final_model_path)
    print(f"最终模型已保存: {final_model_path}")


# 其他辅助函数保持不变 (calculate_metrics, plot_training_history等)
def calculate_metrics(pred, target, threshold=0.5):
    """计算IoU和Dice指标"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)

    intersection = (pred_flat * target_flat).sum()
    pred_sum = pred_flat.sum()
    target_sum = target_flat.sum()
    union = pred_sum + target_sum - intersection

    # IoU计算
    if union > 0:
        iou = intersection / union
    else:
        iou = torch.tensor(1.0 if intersection == 0 else 0.0)

    # Dice计算
    if pred_sum + target_sum > 0:
        dice = (2.0 * intersection) / (pred_sum + target_sum)
    else:
        dice = torch.tensor(1.0 if intersection == 0 else 0.0)

    return iou.item(), dice.item()


def plot_training_history(history, save_dir):
    """绘制并保存训练历史"""
    # 设置字体避免中文问题
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 损失
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # IoU
    axes[0, 1].plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    axes[0, 1].set_title('Training and Validation IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Dice
    axes[0, 2].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[0, 2].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[0, 2].set_title('Training and Validation Dice')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 损失分量 - 训练
    axes[1, 0].plot(epochs, history['train_bce'], 'g-', label='BCE')
    axes[1, 0].plot(epochs, history['train_focal'], 'r-', label='Focal')
    axes[1, 0].plot(epochs, history['train_dice_loss'], 'b-', label='Dice')
    axes[1, 0].set_title('Training Loss Components')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 损失分量 - 验证
    axes[1, 1].plot(epochs, history['val_bce'], 'g-', label='BCE')
    axes[1, 1].plot(epochs, history['val_focal'], 'r-', label='Focal')
    axes[1, 1].plot(epochs, history['val_dice_loss'], 'b-', label='Dice')
    axes[1, 1].set_title('Validation Loss Components')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # 学习率
    axes[1, 2].plot(epochs, history['lr'], 'm-')
    axes[1, 2].set_title('Learning Rate')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'contour_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()