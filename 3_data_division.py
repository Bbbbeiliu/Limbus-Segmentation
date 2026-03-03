import os
import random
from collections import defaultdict
import glob
from pypinyin import lazy_pinyin  # 用于中文姓名拼音转换

# ===================== 全局配置（所有参数都在这里设置）=====================
# D盘数据集配置（按数字文件夹编号划分）
D_DATA_ROOT = r"D:\project\segmentation\data"
D_MASK_ROOT = r"D:\project\segmentation\mask_contour"
D_TRAIN_START = 10       # D盘训练集起始文件夹编号
D_TRAIN_END = 90        # D盘训练集结束文件夹编号
D_VAL_START = 91        # D盘验证集起始文件夹编号
D_VAL_END = 110          # D盘验证集结束文件夹编号

# F盘数据集配置（按拼音首字母排序划分）
F_DATA_ROOT = r"F:\data_resize_2"
F_MASK_ROOT = r"F:\mask_contour_2"
F_TRAIN_NUM = 40        # F盘按拼音排序后，前80个文件夹作为训练集（可根据实际调整）

# 通用配置
OUTPUT_DIR = r"dataset_combined"  # 合并后文件输出目录
SEED = 42                         # 随机种子
SHUFFLE = True                    # 是否打乱数据

# ===========================================================================

def get_pinyin_first_letter(text):
    """获取文本的拼音首字母（用于文件夹排序）"""
    if not isinstance(text, str):
        text = str(text)
    # 处理数字/特殊字符，返回其本身（确保001这类文件夹排序稳定）
    if text.isdigit():
        return text
    # 处理中文姓名，返回拼音首字母大写
    pinyin_list = lazy_pinyin(text, style=0)
    first_letters = ''.join([p[0].upper() for p in pinyin_list])
    return first_letters

def collect_data_pairs(data_root, mask_root, target_subdirs):
    """
    收集指定子目录下的所有数据对（图像+掩膜）
    Args:
        data_root: 数据根目录
        mask_root: 掩膜根目录
        target_subdirs: 需要处理的子目录列表（如['001', '002', '张三']）
    Returns:
        data_pairs: (image_path, mask_path) 列表
        valid_subdirs: 实际存在且有数据的子目录列表
    """
    data_pairs = []
    valid_subdirs = []

    for subdir in target_subdirs:
        data_subdir = os.path.join(data_root, subdir)
        mask_subdir = os.path.join(mask_root, subdir)

        # 跳过不存在的目录
        if not (os.path.isdir(data_subdir) and os.path.isdir(mask_subdir)):
            print(f"跳过目录 {subdir}: 数据或掩膜目录不存在")
            continue

        print(f"处理目录: {subdir}")
        valid_subdirs.append(subdir)

        # 收集该目录下的所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            pattern = os.path.join(data_subdir, f'*{ext}')
            image_files.extend(glob.glob(pattern))

        # 为每个图像匹配对应的掩膜
        for image_path in image_files:
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # 处理增强图像命名
            if base_name.startswith(('agument_', 'augmented_', 'aug_')):
                mask_base_name = base_name
            else:
                mask_base_name = base_name

            # 查找掩膜文件
            mask_found = False
            for mask_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                mask_path = os.path.join(mask_subdir, f"{mask_base_name}{mask_ext}")
                if os.path.exists(mask_path):
                    data_pairs.append((image_path, mask_path))
                    mask_found = True
                    break

            if not mask_found:
                print(f"警告: 未找到掩膜文件 {mask_base_name} 对应的任何掩膜文件")

    return data_pairs, valid_subdirs

def process_d_drive():
    """处理D盘数据集（按数字文件夹编号划分）"""
    print("===== 开始处理D盘数据集 =====")
    # 生成D盘训练集文件夹列表（三位数格式）
    train_subdirs_d = [f"{i:03d}" for i in range(D_TRAIN_START, D_TRAIN_END + 1)]
    # 生成D盘验证集文件夹列表
    val_subdirs_d = [f"{i:03d}" for i in range(D_VAL_START, D_VAL_END + 1)]

    # 收集D盘训练集数据
    print("收集D盘训练集数据...")
    train_pairs_d, train_subdirs_d_valid = collect_data_pairs(D_DATA_ROOT, D_MASK_ROOT, train_subdirs_d)
    # 收集D盘验证集数据
    print("\n收集D盘验证集数据...")
    val_pairs_d, val_subdirs_d_valid = collect_data_pairs(D_DATA_ROOT, D_MASK_ROOT, val_subdirs_d)

    print(f"\nD盘训练集: {len(train_pairs_d)} 个样本 (来自文件夹 {D_TRAIN_START:03d}-{D_TRAIN_END:03d})")
    print(f"D盘验证集: {len(val_pairs_d)} 个样本 (来自文件夹 {D_VAL_START:03d}-{D_VAL_END:03d})")

    return train_pairs_d, val_pairs_d

def process_f_drive():
    """处理F盘数据集（按拼音首字母排序划分）"""
    print("\n===== 开始处理F盘数据集 =====")
    # 获取F盘所有子文件夹
    all_subdirs = [
        name for name in os.listdir(F_DATA_ROOT)
        if os.path.isdir(os.path.join(F_DATA_ROOT, name)) and
           os.path.isdir(os.path.join(F_MASK_ROOT, name))
    ]

    if not all_subdirs:
        print("错误: F盘未找到任何有效子文件夹")
        return [], []

    # 按拼音首字母排序（中文姓名按拼音，数字按字符序）
    sorted_subdirs = sorted(
        all_subdirs,
        key=lambda x: get_pinyin_first_letter(x)
    )

    print(f"F盘所有有效子文件夹（按拼音排序）: {sorted_subdirs}")
    print(f"F盘总共有 {len(sorted_subdirs)} 个子文件夹")

    # 划分训练集和验证集（前F_TRAIN_NUM个为训练，其余为验证）
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

    # 收集F盘训练集数据
    print(f"\n收集F盘训练集数据（前{F_TRAIN_NUM}个文件夹）...")
    train_pairs_f, train_subdirs_f_valid = collect_data_pairs(F_DATA_ROOT, F_MASK_ROOT, train_subdirs_f)
    # 收集F盘验证集数据
    print(f"\n收集F盘验证集数据（剩余 {len(val_subdirs_f)} 个文件夹）...")
    val_pairs_f, val_subdirs_f_valid = collect_data_pairs(F_DATA_ROOT, F_MASK_ROOT, val_subdirs_f)

    print(f"\nF盘训练集: {len(train_pairs_f)} 个样本 (来自前 {len(train_subdirs_f_valid)} 个文件夹)")
    print(f"F盘验证集: {len(val_pairs_f)} 个样本 (来自剩余 {len(val_subdirs_f_valid)} 个文件夹)")

    return train_pairs_f, val_pairs_f

def merge_and_save_data(train_pairs_d, val_pairs_d, train_pairs_f, val_pairs_f):
    """合并D盘和F盘数据，打乱并保存为最终的train.txt/val.txt"""
    # 合并数据
    total_train_pairs = train_pairs_d + train_pairs_f
    total_val_pairs = val_pairs_d + val_pairs_f

    # 验证数据有效性
    if len(total_train_pairs) == 0:
        print("\n错误: 合并后训练集无有效样本")
        return
    if len(total_val_pairs) == 0:
        print("\n错误: 合并后验证集无有效样本")
        return

    # 打乱数据
    if SHUFFLE:
        random.shuffle(total_train_pairs)
        random.shuffle(total_val_pairs)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存训练集
    train_txt_path = os.path.join(OUTPUT_DIR, 'train.txt')
    with open(train_txt_path, 'w', encoding='utf-8') as f:
        for image_path, mask_path in total_train_pairs:
            f.write(f"{image_path} {mask_path}\n")

    # 保存验证集
    val_txt_path = os.path.join(OUTPUT_DIR, 'val.txt')
    with open(val_txt_path, 'w', encoding='utf-8') as f:
        for image_path, mask_path in total_val_pairs:
            f.write(f"{image_path} {mask_path}\n")

    print(f"\n===== 合并后最终统计 =====")
    print(f"训练集总样本数: {len(total_train_pairs)} (D盘: {len(train_pairs_d)} + F盘: {len(train_pairs_f)})")
    print(f"验证集总样本数: {len(total_val_pairs)} (D盘: {len(val_pairs_d)} + F盘: {len(val_pairs_f)})")
    print(f"训练集文件已保存: {train_txt_path}")
    print(f"验证集文件已保存: {val_txt_path}")

    # 详细统计 - 原始/增强图像分布
    def count_image_type(pairs):
        original = 0
        augmented = 0
        for image_path, _ in pairs:
            img_name = os.path.basename(image_path).lower()
            if img_name.startswith(('agument_', 'augmented_', 'aug_')):
                augmented += 1
            else:
                original += 1
        return original, augmented

    # 训练集类型统计
    train_ori, train_aug = count_image_type(total_train_pairs)
    print(f"\n训练集样本类型分布:")
    print(f"  原始图像: {train_ori}")
    print(f"  增强图像: {train_aug}")

    # 验证集类型统计
    val_ori, val_aug = count_image_type(total_val_pairs)
    print(f"\n验证集样本类型分布:")
    print(f"  原始图像: {val_ori}")
    print(f"  增强图像: {val_aug}")

    # 总体统计
    total_samples = len(total_train_pairs) + len(total_val_pairs)
    print(f"\n总体统计:")
    print(f"  训练集比例: {len(total_train_pairs)/total_samples*100:.1f}% ({len(total_train_pairs)}/{total_samples})")
    print(f"  验证集比例: {len(total_val_pairs)/total_samples*100:.1f}% ({len(total_val_pairs)}/{total_samples})")
    print(f"  总样本数: {total_samples}")

def main():
    """主函数：执行完整的数据划分流程"""
    # 设置随机种子
    random.seed(SEED)

    # 处理D盘数据
    train_pairs_d, val_pairs_d = process_d_drive()

    # 处理F盘数据
    train_pairs_f, val_pairs_f = process_f_drive()

    # 合并并保存最终数据
    merge_and_save_data(train_pairs_d, val_pairs_d, train_pairs_f, val_pairs_f)

if __name__ == "__main__":
    main()