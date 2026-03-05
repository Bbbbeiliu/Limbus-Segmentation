import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

# ================== 配置区域 ==================
config = {
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'pin_memory': True,
    'save_freq': 10,
    'early_stop_patience': 10,
    'cv_folds_dir': './cv_folds',      # 划分文件所在目录（由 split_data_cv.py 生成）
    'output_dir': './cv_results',       # 结果保存目录
}
os.makedirs(config['output_dir'], exist_ok=True)

# ================== 数据增强 ==================
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

# ================== 修正后的数据集类 ==================
class SegmentationDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data_pairs = []
        with open(txt_file, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    img_path, label_path = line.split()
                    self.data_pairs.append((img_path, label_path))
        self.transform = transform

    def _fix_palette_label(self, label_img):
        """
        统一将各种格式的掩膜转换为二值图像（前景255，背景0）
        支持：调色板模式、灰度图、0-1二值图、0-255灰度图
        """
        # 转换为灰度图数组
        if label_img.mode == 'P':
            label_np = np.array(label_img)
        else:
            # 其他模式（RGB、RGBA、L等）都先转为L模式
            label_img = label_img.convert('L')
            label_np = np.array(label_img)

        # 二值化：所有大于0的像素视为前景
        label_np = (label_np > 0).astype(np.uint8) * 255
        return Image.fromarray(label_np, mode='L')

    def _synchronized_random_scale_and_crop(self, image, label, target_size, scale_range=(0.50, 1.50)):
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
        label = self._fix_palette_label(label_original)

        # 判断是否为训练模式（transform中包含随机增强）
        is_training = self.transform and any(
            hasattr(t, 'p') or 'Random' in t.__class__.__name__
            for t in self.transform.transforms
        )

        if is_training:
            image, label = self._synchronized_random_scale_and_crop(
                image, label, target_size=(256, 256), scale_range=(0.50, 1.50))
        else:
            image = image.resize((256, 256), Image.BILINEAR)
            label = label.resize((256, 256), Image.NEAREST)

        if self.transform:
            # 移除transform中的Resize（已手动处理）
            transform_list = []
            for t in self.transform.transforms:
                if not isinstance(t, transforms.Resize):
                    transform_list.append(t)
            modified_transform = transforms.Compose(transform_list)
            image = modified_transform(image)
        else:
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        label = transforms.ToTensor()(label)        # 将255映射为1.0，0映射为0.0
        label = (label > 0.5).float()               # 确保严格二值（以防浮点误差）
        return image, label

    def __len__(self):
        return len(self.data_pairs)

# ================== 模型定义 ==================
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
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # 1/4
        x = self.layer2(x)
        features.append(x)  # 1/8
        x = self.layer3(x)
        features.append(x)  # 1/16
        x = self.layer4(x)
        features.append(x)  # 1/32
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

# ================== 损失函数 ==================
class DiceLoss(nn.Module):
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

# ================== 指标计算（全背景返回0.0）==================
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

# ================== 训练一个epoch（含诊断）==================
def train_epoch(model, train_loader, criterion, optimizer, device, epoch=None):
    # 诊断：统计第一个batch的前景样本数（只在第一个epoch打印）
    if epoch == 0:
        first_images, first_targets = next(iter(train_loader))
        first_targets = first_targets.to(device)
        foreground = sum(1 for t in first_targets if t.sum() > 0)
        print(f"训练集第一个batch中有前景的样本数: {foreground} / {first_targets.size(0)}")

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

    n = len(train_loader)
    return (total_loss/n, total_iou/n, total_dice/n,
            total_bce/n, total_focal/n, total_dice_loss/n)

# ================== 验证一个epoch（含诊断）==================
def validate_epoch(model, val_loader, criterion, device, epoch=None, save_images=False):
    # 诊断：打印第一个batch的输出范围，并可选择保存调试图像
    first_images, first_targets = next(iter(val_loader))
    first_images = first_images.to(device)
    first_targets = first_targets.to(device)
    with torch.no_grad():
        first_outputs = model(first_images)
        print(f"原始输出范围: [{first_outputs.min():.4f}, {first_outputs.max():.4f}], 均值: {first_outputs.mean():.4f}")

        if save_images and epoch == 0:   # 只在第一个epoch保存图像
            import torchvision.utils as vutils
            # 反归一化以便正确显示RGB图像
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
            denorm_images = first_images * std + mean
            vutils.save_image(denorm_images[:4], f'debug_images_epoch{epoch}.png')
            vutils.save_image(first_targets[:4], f'debug_masks_epoch{epoch}.png')
            vutils.save_image(first_outputs[:4], f'debug_preds_raw_epoch{epoch}.png')
            pred_binary = (first_outputs > 0.5).float()
            vutils.save_image(pred_binary[:4], f'debug_preds_binary_epoch{epoch}.png')
            print(f"调试图像已保存 (epoch {epoch})")

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

    n = len(val_loader)
    return (total_loss/n, total_iou/n, total_dice/n,
            total_bce/n, total_focal/n, total_dice_loss/n)

# ================== 绘制训练历史 ==================
def plot_training_history(history, save_dir, fold):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1,3,2)
    plt.plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    plt.plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(); plt.grid(True)
    plt.subplot(1,3,3)
    plt.plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    plt.plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'fold{fold}_training_curves.png'), dpi=300)
    plt.close()

# ================== 主函数 ==================
def main():
    # 设置多进程启动方式（Windows兼容）
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    fold_results = {
        'val_dice': [],
        'val_iou': [],
        'best_epoch': [],
        'train_history': []
    }

    try:
        for fold in range(1, 6):
            print(f"\n{'='*50}\n开始训练 Fold {fold}\n{'='*50}")

            train_txt = os.path.join(config['cv_folds_dir'], f'fold{fold}_train.txt')
            val_txt   = os.path.join(config['cv_folds_dir'], f'fold{fold}_val.txt')
            if not os.path.exists(train_txt) or not os.path.exists(val_txt):
                raise FileNotFoundError(f"找不到划分文件: {train_txt} 或 {val_txt}")

            train_dataset = SegmentationDataset(train_txt, transform=train_transform)
            val_dataset   = SegmentationDataset(val_txt,   transform=val_transform)

            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                      num_workers=config['num_workers'], pin_memory=config['pin_memory'], drop_last=True)
            val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False,
                                      num_workers=config['num_workers'], pin_memory=config['pin_memory'])

            print(f"训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

            model = UNetResNet18(num_classes=1, pretrained=True).to(config['device'])
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                print(f"使用 {torch.cuda.device_count()} 块GPU")

            criterion = CombinedLoss(bce_weight=0.3, focal_weight=0.3, dice_weight=0.4)
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

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
            best_epoch = 0
            early_stop_counter = 0
            fold_start_time = time.time()

            for epoch in range(config['num_epochs']):
                print(f"\nFold {fold} - Epoch {epoch+1}/{config['num_epochs']}")
                print("-" * 50)

                train_loss, train_iou, train_dice, train_bce, train_focal, train_dice_loss = train_epoch(
                    model, train_loader, criterion, optimizer, config['device'], epoch=epoch)
                val_loss, val_iou, val_dice, val_bce, val_focal, val_dice_loss = validate_epoch(
                    model, val_loader, criterion, config['device'], epoch=epoch, save_images=True)  # 第一个epoch保存图像

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
                print(f"LR: {current_lr:.2e}")

                # 保存最佳模型
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    best_epoch = epoch + 1
                    early_stop_counter = 0

                    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                    fold_best_dir = os.path.join(config['output_dir'], f'fold{fold}_best')
                    os.makedirs(fold_best_dir, exist_ok=True)

                    torch.save(model_to_save.encoder.state_dict(),
                               os.path.join(fold_best_dir, 'resnet18_encoder.pth'))
                    decoder_state = {
                        'decoder4': model_to_save.decoder4.state_dict(),
                        'decoder3': model_to_save.decoder3.state_dict(),
                        'decoder2': model_to_save.decoder2.state_dict(),
                        'decoder1': model_to_save.decoder1.state_dict(),
                        'final_upconv': model_to_save.final_upconv.state_dict(),
                        'final_conv': model_to_save.final_conv.state_dict()
                    }
                    torch.save(decoder_state, os.path.join(fold_best_dir, 'unet_decoder.pth'))
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'history': history,
                        'config': config
                    }, os.path.join(fold_best_dir, 'checkpoint.pth'))
                    print(f"新最佳模型保存 (Dice: {best_val_dice:.4f})")
                else:
                    early_stop_counter += 1

                if (epoch + 1) % config['save_freq'] == 0:
                    ckpt_dir = os.path.join(config['output_dir'], f'fold{fold}_checkpoints')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                        'history': history,
                    }, os.path.join(ckpt_dir, f'epoch_{epoch+1}.pth'))
                    print(f"检查点保存于 epoch {epoch+1}")

                if early_stop_counter >= config['early_stop_patience']:
                    print(f"验证Dice连续 {config['early_stop_patience']} 轮未提升，停止。")
                    break
                if current_lr < 1e-7:
                    print("学习率过小，停止。")
                    break

            fold_results['val_dice'].append(best_val_dice)
            fold_results['val_iou'].append(max(history['val_iou']))
            fold_results['best_epoch'].append(best_epoch)
            fold_results['train_history'].append(history)

            plot_training_history(history, config['output_dir'], fold)
            fold_time = time.time() - fold_start_time
            print(f"Fold {fold} 完成，耗时 {fold_time/60:.2f} 分钟，最佳Dice: {best_val_dice:.4f}")

    except KeyboardInterrupt:
        print("\n\n用户中断训练，正在保存已完成的折结果...")

    # 汇总绘图（无论正常结束还是中断都会执行）
    if fold_results['val_dice']:
        print("\n" + "="*50)
        print("交叉验证完成！")
        avg_dice = np.mean(fold_results['val_dice'])
        std_dice = np.std(fold_results['val_dice'])
        avg_iou  = np.mean(fold_results['val_iou'])
        print(f"5折验证 Dice 均值: {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"各折最佳Dice: {[f'{d:.4f}' for d in fold_results['val_dice']]}")

        # 绘制5折验证曲线对比
        plt.figure(figsize=(10,6))
        for idx, hist in enumerate(fold_results['train_history']):
            plt.plot(range(1, len(hist['val_dice'])+1), hist['val_dice'], label=f'Fold {idx+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Dice')
        plt.title('5-Fold Validation Dice Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config['output_dir'], 'cv_dice_curves.png'), dpi=300)
        plt.close()

        # 平均曲线带标准差
        min_epochs = min(len(h['val_dice']) for h in fold_results['train_history'])
        avg_val_dice = np.mean([h['val_dice'][:min_epochs] for h in fold_results['train_history']], axis=0)
        std_val_dice = np.std([h['val_dice'][:min_epochs] for h in fold_results['train_history']], axis=0)
        plt.figure(figsize=(10,6))
        epochs = range(1, min_epochs+1)
        plt.plot(epochs, avg_val_dice, 'r-', label='Avg Val Dice')
        plt.fill_between(epochs, avg_val_dice - std_val_dice, avg_val_dice + std_val_dice, color='r', alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.title('Average 5-Fold Validation Dice with Std Dev')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config['output_dir'], 'avg_dice_curves.png'), dpi=300)
        plt.close()

        with open(os.path.join(config['output_dir'], 'cv_summary.txt'), 'w') as f:
            f.write("5-Fold Cross Validation Results\n")
            f.write("================================\n")
            f.write(f"Dice per fold: {', '.join([f'{d:.4f}' for d in fold_results['val_dice']])}\n")
            f.write(f"Mean Dice: {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Mean IoU: {avg_iou:.4f}\n")
            f.write(f"Best epochs: {fold_results['best_epoch']}\n")

        print(f"汇总结果已保存至 {config['output_dir']}")
    else:
        print("未完成任何折的训练，无法生成汇总结果。")

if __name__ == '__main__':
    # mp.freeze_support()  # 如需打包可执行文件，取消注释
    main()