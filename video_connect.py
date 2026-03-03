import cv2
import numpy as np
import time
from tqdm import tqdm  # 进度条库

# 定义4个视频路径
video1 = r"F:\result\result_test\A2_Video08_crop_shape_overlay.mp4"
video2 = r"F:\result\result_test\B2_Video08_crop_shape_overlay.mp4"
video3 = r"F:\result\result11\Result11_Video08_crop_shape_overlay.mp4"
video4 = r"F:\result\result_seg\Result2\Video08_crop_ellipse_overlay.mp4"

# 对应每个视频的标题
titles = [
    "test_A2",
    "test_B2",
    "Result11_contour",
    "Result2_seg"
]

# 打开4个视频捕获对象
cap1 = cv2.VideoCapture(video1)
cap2 = cv2.VideoCapture(video2)
cap3 = cv2.VideoCapture(video3)
cap4 = cv2.VideoCapture(video4)

# 检查视频是否成功打开
def check_video_cap(cap, path):
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{path}")

check_video_cap(cap1, video1)
check_video_cap(cap2, video2)
check_video_cap(cap3, video3)
check_video_cap(cap4, video4)

# 获取视频基础参数
fps = int(cap1.get(cv2.CAP_PROP_FPS))
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 获取每个视频的总帧数，找到最大帧数（总处理帧数）
total_frames_1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_3 = int(cap3.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames_4 = int(cap4.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames = max(total_frames_1, total_frames_2, total_frames_3, total_frames_4)

# 定义输出视频参数
output_path = 'F:/combined_video_2x2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height * 2))

# 定义字体样式（标题显示用）
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)  # 白色
font_thickness = 2
text_offset = (10, 30)  # 标题在帧左上角的偏移量

# 初始化进度条（tqdm），显示预估剩余时间
pbar = tqdm(total=total_frames, desc="视频拼接进度", unit="帧",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} 帧 [{elapsed}<{remaining}, {rate_fmt}]")

# 逐帧处理
start_time = time.time()
for frame_idx in range(total_frames):
    # 读取4个视频的当前帧
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    # 若某视频帧读取失败，用黑屏填充
    if not ret1:
        frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    if not ret2:
        frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    if not ret3:
        frame3 = np.zeros((height, width, 3), dtype=np.uint8)
    if not ret4:
        frame4 = np.zeros((height, width, 3), dtype=np.uint8)

    # 给每个帧添加标题
    cv2.putText(frame1, titles[0], text_offset, font, font_scale, font_color, font_thickness)
    cv2.putText(frame2, titles[1], text_offset, font, font_scale, font_color, font_thickness)
    cv2.putText(frame3, titles[2], text_offset, font, font_scale, font_color, font_thickness)
    cv2.putText(frame4, titles[3], text_offset, font, font_scale, font_color, font_thickness)

    # 2×2布局拼接
    row1 = np.hstack([frame1, frame2])  # 第一行：视频1 + 视频2
    row2 = np.hstack([frame3, frame4])  # 第二行：视频3 + 视频4
    combined = np.vstack([row1, row2])  # 整体拼接

    # 写入输出视频
    output.write(combined)

    # 更新进度条
    pbar.update(1)

# 关闭进度条
pbar.close()

# 释放所有资源
cap1.release()
cap2.release()
cap3.release()
cap4.release()
output.release()
cv2.destroyAllWindows()

# 输出最终统计信息
total_time = time.time() - start_time
print(f"\n视频合成完成！输出路径：{output_path}")
print(f"总处理帧数：{total_frames} 帧")
print(f"总耗时：{total_time:.2f} 秒（约 {total_time/60:.1f} 分钟）")
print(f"平均处理速度：{total_frames/total_time:.2f} 帧/秒")