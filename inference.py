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
import json
import matplotlib.pyplot as plt


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


def load_model(encoder_path, decoder_path, device):
    """Load the trained model from separate encoder and decoder files"""

    # Initialize model
    model = UNetResNet18(num_classes=1, pretrained=False)

    # Load encoder weights
    if os.path.exists(encoder_path):
        encoder_state = torch.load(encoder_path, map_location=device)
        model.encoder.load_state_dict(encoder_state)
        print(f"Loaded encoder from: {encoder_path}")
    else:
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    # Load decoder weights
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
    """Preprocess image for inference"""

    # Load and convert image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)

    # Resize image
    image = image.resize(target_size, Image.BILINEAR)

    # Apply transforms
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
    original_size = (frame.shape[1], frame.shape[0])  # (width, height)

    # Resize image
    image = image.resize(target_size, Image.BILINEAR)

    # Apply transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor, original_size


def extract_ellipse_from_mask(mask_tensor, original_size, threshold=0.5):
    """Extract mask edge and fit an ellipse, return ellipse parameters"""

    # Remove batch dimension and convert to numpy
    mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Resize to original size
    resized_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Find contours in the mask
    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Fit ellipse to the contour
        if len(largest_contour) >= 5:  # Need at least 5 points to fit ellipse
            ellipse = cv2.fitEllipse(largest_contour)
            center, axes, angle = ellipse

            # 转换为可序列化的格式
            ellipse_params = {
                'type': 'ellipse',
                'center': (float(center[0]), float(center[1])),
                'axes': (float(axes[0]), float(axes[1])),
                'angle': float(angle)
            }

            return ellipse_params, resized_mask
        else:
            return None, resized_mask
    else:
        return None, resized_mask


def create_mask_overlay(frame, mask_tensor, original_size, threshold=0.5, alpha=0.5):
    """Create an overlay of the original frame and segmentation mask"""

    # Remove batch dimension and convert to numpy
    mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # Apply threshold
    binary_mask = (mask > threshold).astype(np.uint8) * 255

    # Resize mask to original size
    resized_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Convert mask to 3-channel for overlay
    mask_colored = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:, :, 0] = 0  # Set blue channel to 0
    mask_colored[:, :, 1] = 0  # Set green channel to 0
    # Red channel is already 255 where mask is present

    # Create overlay
    overlay = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)

    return overlay


def draw_ellipse_on_frame(frame, ellipse_params, color=(0, 255, 0), thickness=2):
    """Draw ellipse on the frame"""

    if ellipse_params is not None:
        frame_with_ellipse = frame.copy()
        center = (int(ellipse_params['center'][0]), int(ellipse_params['center'][1]))
        axes = (int(ellipse_params['axes'][0] / 2), int(ellipse_params['axes'][1] / 2))
        angle = ellipse_params['angle']

        cv2.ellipse(frame_with_ellipse, center, axes, angle, 0, 360, color, thickness)
        # 绘制中心点
        cv2.circle(frame_with_ellipse, center, 5, (255, 0, 0), -1)

        return frame_with_ellipse
    else:
        return frame


def create_comparison_image(frame, model_output, original_size,
                            threshold=0.5, ellipse_params=None):
    """Create comparison image with original, segmentation, and ellipse"""

    # 将OpenCV BGR图像转换为RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame

    # 获取原始分割掩膜
    mask = model_output.squeeze(0).squeeze(0).cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8) * 255
    binary_mask_image = Image.fromarray(binary_mask, mode='L')
    binary_mask_image = binary_mask_image.resize(original_size, Image.NEAREST)
    binary_mask_array = np.array(binary_mask_image)

    # 创建分割掩膜彩色图（红色）
    seg_colored = np.zeros((*original_size[::-1], 3), dtype=np.uint8)
    seg_colored[binary_mask_array > 0] = [255, 0, 0]  # 红色

    # 创建掩膜叠加图
    overlay_seg = cv2.addWeighted(frame_rgb, 0.7, seg_colored, 0.3, 0)
    y_offset = 30
    # 创建椭圆叠加图
    overlay_ellipse = frame_rgb.copy()
    if ellipse_params is not None:
        center = (int(ellipse_params['center'][0]), int(ellipse_params['center'][1]))
        axes = (int(ellipse_params['axes'][0] / 2), int(ellipse_params['axes'][1] / 2))
        angle = ellipse_params['angle']

        # 绘制椭圆
        cv2.ellipse(overlay_ellipse, center, axes, angle, 0, 360, (0, 255, 0), 2)  # 绿色椭圆
        cv2.circle(overlay_ellipse, center, 5, (255, 0, 255), -1)  # 洋红色中心点

        # 添加参数文本
        param_text = f"Center: ({center[0]}, {center[1]})"
        axes_text = f"Axes: ({axes[0] * 2:.1f}, {axes[1] * 2:.1f})"
        angle_text = f"Angle: {angle:.1f}°"


        for text in [param_text, axes_text, angle_text]:
            cv2.putText(overlay_ellipse, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

    # 创建3个子图的对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原图
    axes[0].imshow(frame_rgb)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    # 分割掩膜叠加
    axes[1].imshow(overlay_seg)
    axes[1].set_title('Segmentation Mask (Red)', fontsize=12)
    axes[1].axis('off')

    # 椭圆叠加
    axes[2].imshow(overlay_ellipse)
    if ellipse_params is not None:
        axes[2].set_title('Fitted Ellipse (Green)', fontsize=12)
    else:
        axes[2].set_title('No Ellipse Detected', fontsize=12)
    axes[2].axis('off')

    # 调整布局
    plt.tight_layout()

    # 将matplotlib图形转换为PIL图像
    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    # 转换为RGB（去掉alpha通道）
    comparison_img = Image.fromarray(img_array).convert('RGB')

    return comparison_img


def inference_on_folder(input_folder, output_folder, model, device, threshold=0.5):
    """Process images in folder and create comparison images"""

    # Create output directories
    os.makedirs(output_folder, exist_ok=True)

    # 创建inf文件夹用于保存对比图
    inf_folder = os.path.join(output_folder, 'inf')
    os.makedirs(inf_folder, exist_ok=True)

    # 创建参数文件夹
    params_folder = os.path.join(output_folder, 'parameters')
    os.makedirs(params_folder, exist_ok=True)

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
    print(f"Comparison images will be saved to: {inf_folder}")

    # Store parameters
    params_dict = {}
    processed_count = 0
    failed_count = 0

    # Process each image
    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                image_path = os.path.join(input_folder, image_file)
                base_name = os.path.splitext(image_file)[0]

                # Read image
                frame = cv2.imread(image_path)
                if frame is None:
                    print(f"Error reading image: {image_file}")
                    failed_count += 1
                    continue

                # Preprocess
                image_tensor, original_size = preprocess_image(image_path)
                image_tensor = image_tensor.to(device)

                # Inference
                output = model(image_tensor)

                # Extract ellipse
                ellipse_params, original_mask = extract_ellipse_from_mask(
                    output, original_size, threshold
                )

                # Create comparison image
                comparison_img = create_comparison_image(
                    frame, output, original_size,
                    threshold=threshold, ellipse_params=ellipse_params
                )

                # Save comparison image
                comparison_path = os.path.join(inf_folder, f"{base_name}_comparison.png")
                comparison_img.save(comparison_path)

                # Store parameters
                if ellipse_params is not None:
                    params_dict[image_file] = ellipse_params
                else:
                    params_dict[image_file] = {
                        'type': None,
                        'center': None,
                        'axes': None,
                        'angle': None
                    }

                processed_count += 1

            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_count += 1

    # Save parameters to JSON
    if params_dict:
        params_path = os.path.join(params_folder, 'ellipse_parameters.json')
        serializable_params = {}
        for key, data in params_dict.items():
            if data['type'] == 'ellipse':
                serializable_params[key] = {
                    'type': 'ellipse',
                    'center': [float(data['center'][0]), float(data['center'][1])],
                    'axes': [float(data['axes'][0]), float(data['axes'][1])],
                    'angle': float(data['angle'])
                }
            else:
                serializable_params[key] = {
                    'type': None,
                    'center': None,
                    'axes': None,
                    'angle': None
                }

        with open(params_path, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        print(f"Parameters saved to: {params_path}")

    print(f"\nImage processing completed!")
    print(f"Processed: {processed_count} images")
    print(f"Failed: {failed_count} images")
    print(f"Comparison images saved to: {inf_folder}")


def inference_on_video(video_path, output_folder, model, device, threshold=0.5, fps=None):
    """Run inference on video frame by frame and create output video with comparison images"""

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # 创建comparison_images文件夹用于保存对比图（每隔2帧）
    comparison_folder = os.path.join(output_folder, 'comparison_images')
    os.makedirs(comparison_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
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
    print(f"  Comparison frames will be saved every 2 frames to: {comparison_folder}")

    # Create output video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_folder, f"{video_name}_ellipse_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (frame_width, frame_height))

    # Store parameters
    params_list = []
    frame_count = 0

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

            # Extract ellipse
            ellipse_params, original_mask = extract_ellipse_from_mask(
                output, original_size, threshold
            )

            # 每隔2帧保存对比图
            if frame_count % 2 == 0:
                comparison_img = create_comparison_image(
                    frame, output, original_size,
                    threshold=threshold, ellipse_params=ellipse_params
                )

                # 保存对比图
                comparison_path = os.path.join(
                    comparison_folder,
                    f"{video_name}_frame_{frame_count:06d}_comparison.png"
                )
                comparison_img.save(comparison_path)

                print(f"Saved comparison image for frame {frame_count}: {comparison_path}")

            # Draw ellipse on frame for output video
            ellipse_frame = draw_ellipse_on_frame(
                frame, ellipse_params, color=(0, 255, 0), thickness=2
            )

            # Add text with ellipse parameters if available
            if ellipse_params is not None:
                center_text = f"Center: ({ellipse_params['center'][0]:.1f}, {ellipse_params['center'][1]:.1f})"
                axes_text = f"Axes: ({ellipse_params['axes'][0]:.1f}, {ellipse_params['axes'][1]:.1f})"
                angle_text = f"Angle: {ellipse_params['angle']:.1f}°"

                cv2.putText(ellipse_frame, center_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(ellipse_frame, axes_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(ellipse_frame, angle_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(ellipse_frame, "No ellipse detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Write frame to output video
            out.write(ellipse_frame)

            # Store parameters
            if ellipse_params is not None:
                params_list.append({
                    'frame': frame_count,
                    'type': 'ellipse',
                    'center': [float(ellipse_params['center'][0]), float(ellipse_params['center'][1])],
                    'axes': [float(ellipse_params['axes'][0]), float(ellipse_params['axes'][1])],
                    'angle': float(ellipse_params['angle'])
                })
            else:
                params_list.append({
                    'frame': frame_count,
                    'type': None,
                    'center': None,
                    'axes': None,
                    'angle': None
                })

            frame_count += 1
            pbar.update(1)

        pbar.close()

    # Cleanup
    cap.release()
    out.release()

    # Save parameters
    if params_list:
        params_path = os.path.join(output_folder, 'ellipse_parameters.json')
        with open(params_path, 'w') as f:
            json.dump(params_list, f, indent=2)
        print(f"Parameters saved to: {params_path}")

    print(f"Video processing completed!")
    print(f"Processed {frame_count} frames")
    print(f"Output video saved to: {output_video_path}")
    print(f"Comparison images saved to: {comparison_folder} (every 2 frames)")


def main():
    parser = argparse.ArgumentParser(description='U-Net ResNet18 Region Segmentation with Ellipse Fitting')

    # 主要输入输出参数
    parser.add_argument('--input_path', type=str, default='data/112',
                        help='Path to folder containing images OR path to video file')
    parser.add_argument('--output_folder', type=str, default='F:/result/result_seg/Result3',
                        help='Path to folder where results will be saved')

    # 模型参数
    parser.add_argument('--model_dir', type=str, default='./checkpoints_unet_resnet18/Result3',
                        help='Path to directory containing model weights')
    parser.add_argument('--encoder_file', type=str, default='resnet18_encoder.pth',
                        help='Filename of encoder weights')
    parser.add_argument('--decoder_file', type=str, default='unet_decoder.pth',
                        help='Filename of decoder weights')

    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')

    # 处理模式参数
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'video'],
                        help='Processing mode: image (process image folder) or video (process video file)')

    # 处理参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation (default: 0.5)')
    parser.add_argument('--fps', type=int, default=None,
                        help='Output video FPS (video mode only, default: use input video FPS)')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Model paths
    encoder_path = os.path.join(args.model_dir, args.encoder_file)
    decoder_path = os.path.join(args.model_dir, args.decoder_file)

    # Load model
    print("Loading model...")
    model = load_model(encoder_path, decoder_path, device)
    print("Model loaded successfully!")

    print(f"Processing mode: {args.mode}")

    if args.mode == 'video':
        # 处理视频
        if not os.path.isfile(args.input_path):
            print(f"Error: Video file not found: {args.input_path}")
            return

        inference_on_video(
            video_path=args.input_path,
            output_folder=args.output_folder,
            model=model,
            device=device,
            threshold=args.threshold,
            fps=args.fps
        )
    else:
        # 处理图片
        if not os.path.isdir(args.input_path):
            print(f"Error: Image folder not found: {args.input_path}")
            return

        inference_on_folder(
            input_folder=args.input_path,
            output_folder=args.output_folder,
            model=model,
            device=device,
            threshold=args.threshold
        )


if __name__ == "__main__":
    main()