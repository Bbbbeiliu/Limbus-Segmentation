import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from skimage import measure
from scipy.spatial import KDTree

# 导入原有脚本中的模型加载和预处理函数
from inference_contour_modified_espllipse import (
    load_contour_model, preprocess_image, postprocess_contour_mask_fast
)


# -------------------- 工具函数 --------------------
def imread_chinese(path, as_bgr=True):
    """使用PIL读取图片（支持中文路径），返回RGB或BGR数组"""
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_rgb = np.array(img)
    if as_bgr:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_rgb


def load_mask_pil(mask_path, target_size=None):
    """用PIL加载mask，并resize到目标尺寸，返回二值图（0/1）"""
    mask = Image.open(mask_path).convert('L')
    if target_size is not None:
        mask = mask.resize(target_size, Image.NEAREST)
    mask = np.array(mask)
    return (mask > 0).astype(np.uint8)


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
    # 提取轮廓点坐标
    pred_points = np.column_stack(np.where(pred_bin > 0))
    true_points = np.column_stack(np.where(true_bin > 0))

    if len(pred_points) == 0 or len(true_points) == 0:
        return np.nan  # 缺少轮廓

    # 构建KD树
    tree = KDTree(true_points)
    distances, _ = tree.query(pred_points)
    return np.mean(distances)


def count_components(bin_img, min_area=5):
    """统计二值图中的连通组件个数（忽略面积小于min_area的噪声）"""
    labeled = measure.label(bin_img, connectivity=2)
    props = measure.regionprops(labeled)
    # 过滤小面积
    comps = [p for p in props if p.area >= min_area]
    return len(comps)


def dilate(bin_img, kernel_size=3, iterations=1):
    """对二值图进行膨胀"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(bin_img, kernel, iterations=iterations)


# -------------------- 主处理函数 --------------------
def evaluate_validation_set(
        model, device,
        img_root, mask_root,
        output_dir,
        threshold=0.5,
        dilate_iter=1,  # 膨胀次数，用于计算容忍Dice
        image_exts=['.jpg', '.png', '.jpeg'],
        mask_ext='.png'
):
    os.makedirs(output_dir, exist_ok=True)
    low_dice_dir = os.path.join(output_dir, 'High_comps_samples')
    os.makedirs(low_dice_dir, exist_ok=True)

    all_results = []

    # 遍历图片根目录
    for root, dirs, files in os.walk(img_root):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_exts):
                img_path = os.path.join(root, file)
                # 构造mask相对路径
                rel_path = os.path.relpath(img_path, img_root)
                mask_path = os.path.join(mask_root, rel_path)
                mask_path = os.path.splitext(mask_path)[0] + mask_ext

                if not os.path.exists(mask_path):
                    print(f"Warning: mask not found for {img_path}, skipping.")
                    continue

                # ---- 推理 ----
                img_tensor, original_size = preprocess_image(img_path, target_size=(256, 256))
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    output = model(img_tensor)  # (1,1,256,256)

                # 后处理：获取骨架化的预测轮廓（0/255）
                _, _, _, _, pred_contour = postprocess_contour_mask_fast(
                    output.cpu(), original_size, threshold=threshold,
                    skeletonize=True, fit_shape=False
                )  # pred_contour: 原始尺寸，0/255
                pred_contour = (pred_contour > 0).astype(np.uint8)

                # 加载真实轮廓（0/1）
                true_contour = load_mask_pil(mask_path, target_size=(original_size[1], original_size[0]))

                # ---- 计算各项指标 ----
                # 1. 膨胀后Dice（容忍小偏移）
                pred_dilated = dilate(pred_contour, kernel_size=3, iterations=dilate_iter)
                true_dilated = dilate(true_contour, kernel_size=3, iterations=dilate_iter)
                dice_tol = dice_coefficient(pred_dilated, true_dilated)

                # 2. 平均轮廓距离
                avg_dist = average_contour_distance(pred_contour, true_contour)

                # 3. 连通组件数
                pred_comps = count_components(pred_contour)
                true_comps = count_components(true_contour)

                # 4. 面积比
                area_pred = pred_contour.sum()
                area_true = true_contour.sum()
                area_ratio = area_pred / area_true if area_true > 0 else np.nan

                # 记录
                record = {
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'dice_tol': dice_tol,
                    'avg_dist': avg_dist,
                    'pred_components': pred_comps,
                    'true_components': true_comps,
                    'area_pred': area_pred,
                    'area_true': area_true,
                    'area_ratio': area_ratio
                }
                all_results.append(record)

                # 如果满足任一异常条件，保存可视化对比图
                # if dice_tol < 0.2 or avg_dist > 5 or area_ratio < 1.3:
                if pred_comps > 2:
                    # 读取原图（用于可视化）
                    img_bgr = imread_chinese(img_path)  # BGR
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # 生成彩色叠加图
                    pred_color = np.zeros((*pred_contour.shape, 3), dtype=np.uint8)
                    pred_color[pred_contour > 0] = [255, 0, 0]  # 红色
                    true_color = np.zeros((*true_contour.shape, 3), dtype=np.uint8)
                    true_color[true_contour > 0] = [0, 255, 0]  # 绿色

                    overlay_pred = cv2.addWeighted(img_rgb, 0.7, pred_color, 0.3, 0)
                    overlay_true = cv2.addWeighted(img_rgb, 0.7, true_color, 0.3, 0)

                    # 拼接：原图 | 预测 | 真实
                    h, w = img_rgb.shape[:2]
                    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
                    canvas[:, :w] = img_rgb
                    canvas[:, w:2 * w] = overlay_pred
                    canvas[:, 2 * w:3 * w] = overlay_true

                    # 添加指标文本（canvas 仍为 RGB）
                    cv2.putText(canvas, f"Dice_tol:{dice_tol:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                    cv2.putText(canvas, f"Dist:{avg_dist:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                    cv2.putText(canvas, f"Comps:{pred_comps}/{true_comps}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                    cv2.putText(canvas, f"Area ratio:{area_ratio:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)

                    # 保存图像（避免 OpenCV 中文路径问题）
                    safe_name = rel_path.replace('/', '_').replace('\\', '_')
                    save_path = os.path.join(low_dice_dir, f"{safe_name}_flag.png")

                    # 方法：将 canvas (RGB) 转换为 BGR，然后用 imencode 编码为 PNG 字节流，再用普通文件写入
                    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                    success, encoded_img = cv2.imencode('.png', canvas_bgr)
                    if success:
                        with open(save_path, 'wb') as f:
                            f.write(encoded_img.tobytes())
                    else:
                        print(f"Failed to encode image for {img_path}")

                print(f"Processed: {img_path} -> dice_tol={dice_tol:.4f}, avg_dist={avg_dist:.2f}")

    # 保存所有结果到CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'contour_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")


# -------------------- 主函数 --------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate contour prediction with multiple metrics')
    parser.add_argument('--model_dir', type=str, default='./checkpoints_contour_unet/result11',
                        help='Directory containing model weights')
    parser.add_argument('--encoder_file', type=str, default='resnet18_encoder_contour.pth')
    parser.add_argument('--decoder_file', type=str, default='unet_decoder_contour.pth')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', type=str, default='F:/evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--dilate_iter', type=int, default=1,
                        help='Number of dilations for tolerance Dice')
    args = parser.parse_args()

    # 设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # 加载模型
    encoder_path = os.path.join(args.model_dir, args.encoder_file)
    decoder_path = os.path.join(args.model_dir, args.decoder_file)
    model = load_contour_model(encoder_path, decoder_path, device)
    model.eval()

    # 硬编码验证集路径
    img_root = r"F:/validation_dataset"
    mask_root = r"F:/validation_mask"

    evaluate_validation_set(
        model, device,
        img_root, mask_root,
        args.output_dir,
        threshold=args.threshold,
        dilate_iter=args.dilate_iter
    )


if __name__ == '__main__':
    main()