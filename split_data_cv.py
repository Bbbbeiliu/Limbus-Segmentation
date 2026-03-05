import os
import random
from glob import glob

# 硬编码路径
data_root = "F:/data_resize_2/"
mask_root = "F:/mask_seg_2/"
output_dir = "./cv_folds"          # 划分文件输出目录
high_freq_dirs = ["001", "002", "003", "004"]
random.seed(42)                     # 固定随机种子

os.makedirs(output_dir, exist_ok=True)

# 获取所有案例文件夹
all_case_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
print(f"总案例数: {len(all_case_dirs)}")

high_cases = [d for d in all_case_dirs if d in high_freq_dirs]
low_cases  = [d for d in all_case_dirs if d not in high_freq_dirs]
print(f"高频案例: {high_cases}")
print(f"低频案例数: {len(low_cases)}")
assert len(low_cases) == 59, f"低频案例数应为59，实际为{len(low_cases)}"

random.shuffle(low_cases)

K = 5
folds = [[] for _ in range(K)]

# 分配高频案例：前4折各一个
for i, case in enumerate(high_cases):
    folds[i].append(case)

# 分配低频案例
low_per_fold = len(low_cases) // K          # 11
remainder = len(low_cases) % K               # 4
start = 0
for i in range(K):
    num = low_per_fold + (1 if i < remainder else 0)
    folds[i].extend(low_cases[start:start+num])
    start += num

# 打印各折案例分布
for i, fold_cases in enumerate(folds):
    print(f"Fold {i+1}: {len(fold_cases)} 案例，高频: {[c for c in fold_cases if c in high_cases]}")

# 收集每个案例的所有图像-掩码对
case_pairs = {}
for case in all_case_dirs:
    img_dir = os.path.join(data_root, case)
    mask_dir = os.path.join(mask_root, case)
    if not os.path.exists(mask_dir):
        print(f"警告: 掩码目录 {mask_dir} 不存在，跳过")
        continue
    # 支持常见图像格式
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        img_files.extend(glob(os.path.join(img_dir, ext)))
    pairs = []
    for img_path in img_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, base + '.png')
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            print(f"警告: 掩码不存在 {mask_path}，跳过")
    case_pairs[case] = pairs
    print(f"案例 {case}: {len(pairs)} 对")

# 写入每个折的 train.txt / val.txt
for i in range(K):
    val_cases = folds[i]
    train_cases = []
    for j in range(K):
        if j != i:
            train_cases.extend(folds[j])

    # 验证集
    with open(os.path.join(output_dir, f"fold{i+1}_val.txt"), 'w', encoding='utf-8') as f:
        for case in val_cases:
            for img, msk in case_pairs.get(case, []):
                f.write(f"{img} {msk}\n")

    # 训练集
    with open(os.path.join(output_dir, f"fold{i+1}_train.txt"), 'w', encoding='utf-8') as f:
        for case in train_cases:
            for img, msk in case_pairs.get(case, []):
                f.write(f"{img} {msk}\n")

    print(f"Fold {i+1}: 训练集 {len(train_cases)} 案例，{sum(len(case_pairs.get(c,[])) for c in train_cases)} 图像；"
          f"验证集 {len(val_cases)} 案例，{sum(len(case_pairs.get(c,[])) for c in val_cases)} 图像")

print("划分完成！")