import os
import sys
import subprocess


def quick_test():
    """快速测试训练流程"""
    cmd = [
        "python", "train_contour_network.py",
        "--batch_size", "4",  # 减小batch_size
        "--num_epochs", "3",  # 只训练3个epoch
        "--save_freq", "1",  # 每个epoch都保存检查点
        "--num_workers", "2",  # 减少数据加载进程
        "--learning_rate", "1e-4",  # 保持学习率
    ]

    print("开始快速测试训练流程...")
    print("命令:", " ".join(cmd))

    # 设置环境变量（如果需要）
    env = os.environ.copy()

    try:
        # 运行训练脚本
        result = subprocess.run(cmd, env=env, check=True)
        print("快速测试完成!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"训练过程出错: {e}")
        return e.returncode
    except FileNotFoundError:
        print("错误: 找不到 train_contour_network.py 文件")
        return 1


if __name__ == "__main__":
    # 添加命令行参数解析，以便可以灵活调整参数
    import argparse

    parser = argparse.ArgumentParser(description='快速训练测试')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--num_epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--train_txt', type=str, default='dataset/train.txt', help='训练集文件')
    parser.add_argument('--val_txt', type=str, default='dataset/val.txt', help='验证集文件')

    args = parser.parse_args()

    # 构建命令
    cmd = [
        "python", "train_contour_network.py",
        "--batch_size", str(args.batch_size),
        "--num_epochs", str(args.num_epochs),
        "--save_freq", "1",
        "--num_workers", "4",
        "--learning_rate", str(args.learning_rate),
        "--train_txt", args.train_txt,
        "--val_txt", args.val_txt,
    ]

    print("开始快速测试训练流程...")
    print("命令:", " ".join(cmd))

    env = os.environ.copy()

    try:
        result = subprocess.run(cmd, env=env, check=True)
        print("快速测试完成!")
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"训练过程出错: {e}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("错误: 找不到 train_contour_network.py 文件")
        sys.exit(1)