# -*- coding: utf-8 -*-
"""
批量图片分割工具
基于 inference_contour_modified_espllipse.py 中的模型定义和权重
从输入文件夹读取图片，输出二值分割掩膜（白色为前景）
"""

import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


# ==================== 模型定义（与原始脚本完全一致）====================
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
        self.layer1 = resnet.layer1  # 64
        self.layer2 = resnet.layer2  # 128
        self.layer3 = resnet.layer3  # 256
        self.layer4 = resnet.layer4  # 512

    def forward(self, x):
        features = []
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x)
        features.append(x)  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x);
        features.append(x)  # 1/4
        x = self.layer2(x);
        features.append(x)  # 1/8
        x = self.layer3(x);
        features.append(x)  # 1/16
        x = self.layer4(x);
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
        x = self.conv1(x);
        x = self.bn1(x);
        x = self.relu(x)
        x = self.conv2(x);
        x = self.bn2(x);
        x = self.relu(x)
        return x


class UNetResNet18(nn.Module):
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
        x = self.decoder4(features[4], features[3])
        x = self.decoder3(x, features[2])
        x = self.decoder2(x, features[1])
        x = self.decoder1(x, features[0])
        x = self.final_upconv(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x


# ==================== 预处理（与原始一致）====================
PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(encoder_path, decoder_path, device):
    """加载模型权重（与原始脚本 load_contour_model 相同）"""
    model = UNetResNet18(num_classes=1, pretrained=False)
    if os.path.exists(encoder_path):
        encoder_state = torch.load(encoder_path, map_location=device)
        model.encoder.load_state_dict(encoder_state)
        print(f"Loaded encoder from: {encoder_path}")
    else:
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
    if os.path.exists(decoder_path):
        decoder_state = torch.load(decoder_path, map_location=device)
        model.decoder4.load_state_dict(decoder_state['decoder4'])
        model.decoder3.load_state_dict(decoder_state['decoder3'])
        model.decoder2.load_state_dict(decoder_state['decoder2'])
        model.decoder1.load_state_dict(decoder_state['decoder1'])
        model.final_upconv.load_state_dict(decoder_state['final_upconv'])
        model.final_conv.load_state_dict(decoder_state['final_conv'])
        print(f"Loaded decoder from: {decoder_path}")
    else:
        raise FileNotFoundError(f"Decoder file not found: {decoder_path}")
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, target_size=(256, 256)):
    """读取图片并预处理为网络输入 tensor"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    image = image.resize(target_size, Image.BILINEAR)
    image_tensor = PREPROCESS_TRANSFORM(image).unsqueeze(0)  # (1,3,H,W)
    return image_tensor, original_size


def predict_mask(model, image_tensor, device, threshold=0.5, return_original_size=False, original_size=None):
    """推理得到二值掩膜（0/255 uint8）"""
    with torch.no_grad():
        output = model(image_tensor.to(device))  # (1,1,H,W)
        prob = output.squeeze(0).squeeze(0).cpu().numpy()  # (H,W)
    binary = (prob > threshold).astype(np.uint8) * 255
    if return_original_size and original_size is not None:
        # 缩放回原始尺寸（使用最近邻插值保持二值性）
        binary = cv2.resize(binary, original_size, interpolation=cv2.INTER_NEAREST)
    return binary


# ==================== 批量处理函数 ====================
def segment_images_from_folder(
        input_folder,
        output_folder,
        model_dir,
        encoder_file='resnet18_encoder_seg.pth',
        decoder_file='unet_decoder_seg.pth',
        device='auto',
        threshold=0.5,
        target_size=(256, 256),
        image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tif'),
        save_format='png'
):
    """
    从指定文件夹读取图片，输出分割掩膜到 output_folder

    参数：
        input_folder   : 输入图片文件夹路径
        output_folder  : 输出掩膜文件夹路径（自动创建）
        model_dir      : 包含 encoder 和 decoder 权重文件的目录
        encoder_file   : encoder 权重文件名
        decoder_file   : decoder 权重文件名
        device         : 'cuda', 'cpu' 或 'auto'
        threshold      : 二值化阈值 (0~1)
        target_size    : 网络输入尺寸 (H,W)
        image_extensions: 支持的图片扩展名元组
        save_format    : 输出格式 'png' 或 'jpg'（png 无损）
    """
    # 设置设备
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载模型
    encoder_path = os.path.join(model_dir, encoder_file)
    decoder_path = os.path.join(model_dir, decoder_file)
    model = load_model(encoder_path, decoder_path, device)

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 收集所有图片文件
    image_files = []
    for f in os.listdir(input_folder):
        if f.lower().endswith(image_extensions):
            image_files.append(f)
    if not image_files:
        print(f"No image files found in {input_folder}")
        return

    print(f"Found {len(image_files)} images. Processing...")
    for img_name in tqdm(image_files):
        img_path = os.path.join(input_folder, img_name)
        try:
            # 预处理
            tensor, original_size = preprocess_image(img_path, target_size)
            # 推理并得到原始尺寸的掩膜
            mask = predict_mask(model, tensor, device, threshold,
                                return_original_size=True,
                                original_size=original_size)
            # 保存掩膜
            base_name = os.path.splitext(img_name)[0]
            if save_format.lower() == 'png':
                out_path = os.path.join(output_folder, f"{base_name}.png")
                cv2.imwrite(out_path, mask)  # 保存为灰度图
            else:
                out_path = os.path.join(output_folder, f"{base_name}.jpg")
                cv2.imwrite(out_path, mask, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print(f"All masks saved to: {output_folder}")


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Batch segmentation for images')
    parser.add_argument('--input_folder', type=str, default='D:/project/navigation/XFeat/assets',
                        help='Folder containing input images')
    parser.add_argument('--output_folder', type=str, default='D:/project/navigation/result/assets',
                        help='Folder to save output masks')
    parser.add_argument('--model_dir', type=str, default='./checkpoints_seg_unet_obstacle/Result6',
                        help='Directory containing model weights')
    parser.add_argument('--encoder_file', type=str, default='resnet18_encoder_seg.pth',
                        help='Encoder filename')
    parser.add_argument('--decoder_file', type=str, default='unet_decoder_seg.pth',
                        help='Decoder filename')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, auto')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Segmentation threshold')
    parser.add_argument('--save_format', type=str, default='png',
                        choices=['png', 'jpg'], help='Output format')
    args = parser.parse_args()

    segment_images_from_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        model_dir=args.model_dir,
        encoder_file=args.encoder_file,
        decoder_file=args.decoder_file,
        device=args.device,
        threshold=args.threshold,
        save_format=args.save_format
    )