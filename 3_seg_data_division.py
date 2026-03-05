import os
import random
from pathlib import Path
from collections import defaultdict
from pypinyin import lazy_pinyin  # 用于中文姓名拼音转换

# ===================== 全局配置（所有参数都在这里设置）=====================
# D盘数据集配置（按数字文件夹编号划分）
D_DATA_ROOT = r"D:\project\segmentation\data"
D_MASK_ROOT = r"D:\project\segmentation\mask_seg"
D_TRAIN_START = 51       # D盘训练集起始文件夹编号
D_TRAIN_END = 105         # D盘训练集结束文件夹编号
D_VAL_START = 106         # D盘验证集起始文件夹编号
D_VAL_END = 109           # D盘验证集结束文件夹编号

# F盘数据集配置（按拼音首字母排序划分）
F_DATA_ROOT = r"F:\data_resize_2"
F_MASK_ROOT = r"F:\mask_seg_2"
F_TRAIN_NUM = 60         # F盘按拼音排序后，前40个文件夹作为训练集（可根据实际调整）

# 通用配置
OUTPUT_DIR = r"dataset_combined"  # 合并后文件输出目录
SEED = 42                              # 随机种子
SHUFFLE = True                         # 是否打乱数据
# ===========================================================================

def get_pinyin_first_letter(text):
    """获取文本的拼音首字母（用于文件夹排序）"""
    if not isinstance(text, str):
        text = str(text)
    if text.isdigit():
        return text
    pinyin_list = lazy_pinyin(text, style=0)
    first_letters = ''.join([p[0].upper() for p in pinyin_list])
    return first_letters

def collect_data_pairs(data_root, mask_root, target_subdirs):
    """
    收集指定子目录下的所有数据对（图像 .jpg，掩膜 .png）
    增强图像命名（以 agument_ / augmented_ / aug_ 开头）会保留相同基本名
    Args:
        data_root: 数据根目录
        mask_root: 掩膜根目录
        target_subdirs: 需要处理的子目录列表（如['001', '002', '张三']）
    Returns:
        data_pairs: (image_path, mask_path) 列表
        valid_subdirs: 实际存在且有数据的子目录列表
    """
    data_root = Path(data_root)
    mask_root = Path(mask_root)
    data_pairs = []
    valid_subdirs = []

    for subdir in target_subdirs:
        data_subdir = data_root / subdir
        mask_subdir = mask_root / subdir

        if not data_subdir.is_dir() or not mask_subdir.is_dir():
            print(f"跳过目录 {subdir}: 数据或掩膜目录不存在")
            continue

        print(f"处理目录: {subdir}")
        valid_subdirs.append(subdir)

        # 收集该目录下的所有 .jpg 图像文件
        image_files = list(data_subdir.glob("*.jpg"))
        if not image_files:
            print(f"  警告: 文件夹 {subdir} 中没有找到 .jpg 文件")
            continue

        for image_path in image_files:
            base_name = image_path.stem   # 不含扩展名

            # 处理增强图像命名
            if base_name.startswith(('agument_', 'augmented_', 'aug_')):
                mask_base_name = base_name
            else:
                mask_base_name = base_name

            # 掩膜必须为 .png
            mask_path = mask_subdir / f"{mask_base_name}.png"
            if mask_path.is_file():
                data_pairs.append((str(image_path.absolute()), str(mask_path.absolute())))
            else:
                print(f"  警告: 未找到掩膜文件 {mask_base_name}.png")

    return data_pairs, valid_subdirs

def process_d_drive():
    """处理D盘数据集（按数字文件夹编号划分）"""
    print("===== 开始处理D盘数据集 =====")
    train_subdirs_d = [f"{i:03d}" for i in range(D_TRAIN_START, D_TRAIN_END + 1)]
    val_subdirs_d   = [f"{i:03d}" for i in range(D_VAL_START, D_VAL_END + 1)]

    print("收集D盘训练集数据...")
    train_pairs_d, train_valid = collect_data_pairs(D_DATA_ROOT, D_MASK_ROOT, train_subdirs_d)
    print("\n收集D盘验证集数据...")
    val_pairs_d, val_valid = collect_data_pairs(D_DATA_ROOT, D_MASK_ROOT, val_subdirs_d)

    print(f"\nD盘训练集: {len(train_pairs_d)} 个样本 (来自文件夹 {D_TRAIN_START:03d}-{D_TRAIN_END:03d})")
    print(f"D盘验证集: {len(val_pairs_d)} 个样本 (来自文件夹 {D_VAL_START:03d}-{D_VAL_END:03d})")
    return train_pairs_d, val_pairs_d

def process_f_drive():
    """处理F盘数据集（按拼音首字母排序划分）"""
    print("\n===== 开始处理F盘数据集 =====")
    f_data_root = Path(F_DATA_ROOT)
    f_mask_root = Path(F_MASK_ROOT)

    # 获取F盘所有子文件夹（同时存在于数据和掩膜目录中）
    all_subdirs = [
        d.name for d in f_data_root.iterdir()
        if d.is_dir() and (f_mask_root / d.name).is_dir()
    ]

    if not all_subdirs:
        print("错误: F盘未找到任何有效子文件夹")
        return [], []

    # 按拼音首字母排序
    sorted_subdirs = sorted(all_subdirs, key=lambda x: get_pinyin_first_letter(x))
    print(f"F盘所有有效子文件夹（按拼音排序）: {sorted_subdirs}")
    print(f"F盘总共有 {len(sorted_subdirs)} 个子文件夹")

    # 划分训练集和验证集
    if F_TRAIN_NUM >= len(sorted_subdirs):
        print(f"警告: F_TRAIN_NUM({F_TRAIN_NUM}) >= F盘总文件夹数({len(sorted_subdirs)})，所有F盘数据都作为训练集")
        train_subdirs_f = sorted_subdirs
        val_subdirs_f = []
    elif F_TRAIN_NUM <= 0:
        print("警告: F_TRAIN_NUM <= 0，所有F盘数据都作为验证集")
        train_subdirs_f = []
        val_subdirs_f = sorted_subdirs
    else:
        train_subdirs_f = sorted_subdirs[:F_TRAIN_NUM]
        val_subdirs_f = sorted_subdirs[F_TRAIN_NUM:]

    print(f"\n收集F盘训练集数据（前{len(train_subdirs_f)}个文件夹）...")
    train_pairs_f, train_valid = collect_data_pairs(F_DATA_ROOT, F_MASK_ROOT, train_subdirs_f)
    print(f"\n收集F盘验证集数据（剩余{len(val_subdirs_f)}个文件夹）...")
    val_pairs_f, val_valid = collect_data_pairs(F_DATA_ROOT, F_MASK_ROOT, val_subdirs_f)

    print(f"\nF盘训练集: {len(train_pairs_f)} 个样本 (来自 {len(train_valid)} 个文件夹)")
    print(f"F盘验证集: {len(val_pairs_f)} 个样本 (来自 {len(val_valid)} 个文件夹)")
    return train_pairs_f, val_pairs_f

def merge_and_save_data(train_pairs_d, val_pairs_d, train_pairs_f, val_pairs_f):
    """合并D盘和F盘数据，打乱并保存为 train_seg.txt / val_seg.txt"""
    total_train_pairs = train_pairs_d + train_pairs_f
    total_val_pairs   = val_pairs_d + val_pairs_f

    if len(total_train_pairs) == 0:
        print("\n错误: 合并后训练集无有效样本")
        return
    if len(total_val_pairs) == 0:
        print("\n错误: 合并后验证集无有效样本")
        return

    if SHUFFLE:
        random.shuffle(total_train_pairs)
        random.shuffle(total_val_pairs)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练集
    train_txt = output_dir / 'train_seg.txt'
    with open(train_txt, 'w', encoding='utf-8') as f:
        for img, msk in total_train_pairs:
            f.write(f"{img} {msk}\n")

    # 保存验证集
    val_txt = output_dir / 'val_seg.txt'
    with open(val_txt, 'w', encoding='utf-8') as f:
        for img, msk in total_val_pairs:
            f.write(f"{img} {msk}\n")

    print(f"\n===== 合并后最终统计 =====")
    print(f"训练集总样本数: {len(total_train_pairs)} (D盘: {len(train_pairs_d)} + F盘: {len(train_pairs_f)})")
    print(f"验证集总样本数: {len(total_val_pairs)} (D盘: {len(val_pairs_d)} + F盘: {len(val_pairs_f)})")
    print(f"训练集文件已保存: {train_txt}")
    print(f"验证集文件已保存: {val_txt}")

    # 统计原始/增强图像分布
    def count_image_type(pairs):
        orig = aug = 0
        for img, _ in pairs:
            name = Path(img).stem.lower()
            if name.startswith(('agument_', 'augmented_', 'aug_')):
                aug += 1
            else:
                orig += 1
        return orig, aug

    train_orig, train_aug = count_image_type(total_train_pairs)
    val_orig, val_aug = count_image_type(total_val_pairs)
    print(f"\n训练集样本类型分布:  原始图像 {train_orig}，增强图像 {train_aug}")
    print(f"验证集样本类型分布:  原始图像 {val_orig}，增强图像 {val_aug}")

    total = len(total_train_pairs) + len(total_val_pairs)
    print(f"\n总体统计:")
    print(f"  训练集比例: {len(total_train_pairs)/total*100:.1f}% ({len(total_train_pairs)}/{total})")
    print(f"  验证集比例: {len(total_val_pairs)/total*100:.1f}% ({len(total_val_pairs)}/{total})")
    print(f"  总样本数: {total}")

def main():
    random.seed(SEED)
    train_d, val_d = process_d_drive()
    train_f, val_f = process_f_drive()
    merge_and_save_data(train_d, val_d, train_f, val_f)

if __name__ == "__main__":
    print("语义分割数据集分割工具（支持 D 盘 + F 盘合并）")
    print("此脚本将数据分割为 train_seg.txt 和 val_seg.txt\n")
    main()