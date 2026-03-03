import cv2
import numpy as np
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import glob


class MaskEraserTool:
    def __init__(self, root):
        self.root = root
        self.root.title("掩膜擦除工具")
        self.root.geometry("1400x900")

        # 初始化变量
        self.data_dir = "data"
        self.mask_seg_dir = "mask_seg"
        self.mask_contour_dir = "mask_contour"
        self.current_index = 0
        self.current_mask_type = "seg"  # seg 或 contour
        self.show_binary = False
        self.eraser_size = 20
        self.erase_mode = "erase"  # erase或add
        self.history = []  # 历史记录
        self.max_history = 10  # 最大历史记录数
        self.auto_save = False  # 自动保存开关

        # 图片列表
        self.image_paths = []
        self.current_image = None
        self.current_mask = None

        # 橡皮擦状态
        self.dragging = False
        self.last_point = None

        # 橡皮擦可视化
        self.eraser_visible = False
        self.eraser_x = 0
        self.eraser_y = 0

        # 创建UI
        self.create_widgets()

        # 绑定快捷键
        self.bind_shortcuts()

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 顶部控制组
        top_control_frame = ttk.Frame(control_frame)
        top_control_frame.pack(fill=tk.X, pady=(0, 15))

        # 文件夹选择
        ttk.Label(top_control_frame, text="数据目录:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.data_dir_entry = ttk.Entry(top_control_frame, width=30)
        self.data_dir_entry.insert(0, self.data_dir)
        self.data_dir_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        ttk.Button(top_control_frame, text="选择文件夹", command=self.select_data_dir).grid(row=0, column=3, padx=5,
                                                                                            pady=2)

        # 图片选择
        ttk.Label(top_control_frame, text="指定图片:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.image_entry = ttk.Entry(top_control_frame, width=30)
        self.image_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=2, sticky=tk.W)
        ttk.Button(top_control_frame, text="选择图片", command=self.select_image).grid(row=1, column=3, padx=5, pady=2)

        # 加载/保存按钮
        button_frame = ttk.Frame(top_control_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=5, sticky=tk.W)
        ttk.Button(button_frame, text="加载", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="保存", command=self.save_mask).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="撤销(Ctrl+Z)", command=self.undo).pack(side=tk.LEFT, padx=2)

        # 自动保存复选框
        self.auto_save_var = tk.BooleanVar(value=self.auto_save)
        auto_save_check = ttk.Checkbutton(button_frame, text="自动保存",
                                          variable=self.auto_save_var,
                                          command=self.toggle_auto_save)
        auto_save_check.pack(side=tk.LEFT, padx=2)

        # 分隔线
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 导航控制组
        nav_frame = ttk.LabelFrame(control_frame, text="导航", padding="10")
        nav_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Button(nav_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(nav_frame, text="下一张", command=self.next_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Label(nav_frame, text="").pack(side=tk.LEFT, expand=True)  # 占位符

        # 当前索引显示
        self.index_label = ttk.Label(nav_frame, text="0/0")
        self.index_label.pack(side=tk.RIGHT, padx=5)

        # 分隔线
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 掩膜控制组
        mask_frame = ttk.LabelFrame(control_frame, text="掩膜操作", padding="10")
        mask_frame.pack(fill=tk.X, pady=(0, 15))

        # 掩膜类型选择
        mask_type_frame = ttk.Frame(mask_frame)
        mask_type_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(mask_type_frame, text="掩膜类型:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(mask_type_frame, text="Seg掩膜", command=lambda: self.set_mask_type("seg")).pack(side=tk.LEFT,
                                                                                                    padx=2)
        ttk.Button(mask_type_frame, text="Contour掩膜", command=lambda: self.set_mask_type("contour")).pack(
            side=tk.LEFT, padx=2)

        # 擦除模式选择
        erase_mode_frame = ttk.Frame(mask_frame)
        erase_mode_frame.pack(fill=tk.X, pady=5)

        ttk.Label(erase_mode_frame, text="擦除模式:").pack(side=tk.LEFT, padx=(0, 5))
        self.erase_mode_var = tk.StringVar(value="erase")
        ttk.Radiobutton(erase_mode_frame, text="擦除(白→黑)", variable=self.erase_mode_var,
                        value="erase", command=self.update_erase_mode).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(erase_mode_frame, text="添加(黑→白)", variable=self.erase_mode_var,
                        value="add", command=self.update_erase_mode).pack(side=tk.LEFT, padx=2)

        # 显示选项
        view_frame = ttk.Frame(mask_frame)
        view_frame.pack(fill=tk.X, pady=5)

        self.binary_var = tk.BooleanVar()
        ttk.Checkbutton(view_frame, text="显示二值图", variable=self.binary_var,
                        command=self.toggle_binary_view).pack(side=tk.LEFT)

        # 分隔线
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 橡皮擦控制组
        eraser_frame = ttk.LabelFrame(control_frame, text="橡皮擦设置", padding="10")
        eraser_frame.pack(fill=tk.X)

        ttk.Label(eraser_frame, text="橡皮擦大小:").pack(anchor=tk.W)

        # 先创建标签，再创建滑块
        self.size_label = ttk.Label(eraser_frame, text=str(self.eraser_size))
        self.size_label.pack(anchor=tk.W)

        self.size_scale = ttk.Scale(eraser_frame, from_=5, to=100, orient=tk.HORIZONTAL)
        self.size_scale.set(self.eraser_size)
        self.size_scale.pack(fill=tk.X, pady=(0, 10))

        # 设置滑块事件（在标签创建后）
        self.size_scale.configure(command=self.update_eraser_size)

        # 状态信息
        self.status_label = ttk.Label(control_frame, text="就绪", relief=tk.SUNKEN, padding=5)
        self.status_label.pack(fill=tk.X, pady=(15, 0))

        # 右侧图像显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 图像信息显示
        info_frame = ttk.Frame(display_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.info_label = ttk.Label(info_frame, text="未加载图像", font=('Arial', 10))
        self.info_label.pack(anchor=tk.W)

        # 图像显示画布
        self.canvas = tk.Canvas(display_frame, bg="gray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 创建橡皮擦可视化矩形
        self.eraser_rect = self.canvas.create_rectangle(0, 0, 0, 0,
                                                        outline="white",
                                                        width=1,
                                                        fill="",  # 透明填充
                                                        state="hidden")  # 初始隐藏

        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_drag)
        self.canvas.bind("<Motion>", self.update_eraser_position)
        self.canvas.bind("<Leave>", self.hide_eraser)

        # 初始加载
        self.scan_images()

    def bind_shortcuts(self):
        """绑定快捷键"""
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-Z>", lambda e: self.undo())
        self.root.bind("<Control-s>", lambda e: self.save_mask())
        self.root.bind("<Control-S>", lambda e: self.save_mask())
        self.root.bind("<KeyPress-a>", lambda e: self.prev_image())
        self.root.bind("<KeyPress-A>", lambda e: self.prev_image())
        self.root.bind("<KeyPress-d>", lambda e: self.next_image())
        self.root.bind("<KeyPress-D>", lambda e: self.next_image())

    def scan_images(self):
        """扫描data目录下的所有图片"""
        self.image_paths = []
        data_path = Path(self.data_dir_entry.get())

        if data_path.exists():
            # 查找所有子文件夹中的图片
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
                self.image_paths.extend(glob.glob(str(data_path / "**" / ext), recursive=True))

            # 按路径排序
            self.image_paths.sort()
            self.update_status(f"找到 {len(self.image_paths)} 张图片")

    def select_data_dir(self):
        """选择数据目录"""
        dir_path = filedialog.askdirectory(initialdir=self.data_dir)
        if dir_path:
            self.data_dir_entry.delete(0, tk.END)
            self.data_dir_entry.insert(0, dir_path)
            self.scan_images()

    def select_image(self):
        """选择单张图片"""
        file_path = filedialog.askopenfilename(
            initialdir=self.data_dir,
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        if file_path:
            self.image_entry.delete(0, tk.END)
            self.image_entry.insert(0, file_path)

    def load_image(self):
        """加载图片和对应的掩膜"""
        image_path = self.image_entry.get()

        if not image_path:
            if not self.image_paths:
                messagebox.showerror("错误", "没有找到图片")
                return
            # 使用当前索引的图片
            if 0 <= self.current_index < len(self.image_paths):
                image_path = self.image_paths[self.current_index]
            else:
                return

        # 检查图片是否存在
        if not os.path.exists(image_path):
            messagebox.showerror("错误", f"图片不存在: {image_path}")
            return

        # 加载原始图片
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                raise ValueError("无法读取图片")
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
            return

        # 获取图片相对路径和名称
        try:
            rel_path = Path(image_path).relative_to(self.data_dir_entry.get())
            subfolder = rel_path.parent
            image_name = Path(image_path).stem

            # 处理子文件夹为"."的情况
            if str(subfolder) == ".":
                # 尝试获取上一级文件夹
                parent_dir = Path(image_path).parent.name
                if parent_dir.isdigit():
                    subfolder = parent_dir
                else:
                    subfolder = "."
        except ValueError:
            # 如果图片不在data目录下，使用文件名作为标识
            subfolder = "."
            image_name = Path(image_path).stem

        # 加载对应的掩膜
        self.load_mask(subfolder, image_name)

        # 更新状态
        self.update_status(f"正在处理: {subfolder}/{image_name}")
        self.index_label.config(text=f"{self.current_index + 1}/{len(self.image_paths)}")

        # 显示图片
        self.display_image()

    def load_mask(self, subfolder, image_name):
        """加载对应的掩膜"""
        mask_dir = self.mask_seg_dir if self.current_mask_type == "seg" else self.mask_contour_dir

        # 构建掩膜路径
        if str(subfolder) == ".":
            mask_path = os.path.join(mask_dir, f"{image_name}.png")
        else:
            mask_path = os.path.join(mask_dir, str(subfolder), f"{image_name}.png")

        print(f"尝试加载掩膜: {mask_path}")

        if os.path.exists(mask_path):
            if self.current_mask_type == "seg":
                # 对于seg掩膜，使用PIL读取并转换为二值图像
                pil_img = Image.open(mask_path)

                # 如果是调色板图像，转换为灰度
                if pil_img.mode == 'P':
                    # 获取调色板数据
                    palette = pil_img.getpalette()
                    # 转换为灰度图像
                    pil_img = pil_img.convert('L')

                # 转换为numpy数组
                mask_array = np.array(pil_img)

                # 确保是二值图像（将任何非零值设为255）
                _, self.current_mask = cv2.threshold(mask_array, 1, 255, cv2.THRESH_BINARY)
            else:
                # 对于contour掩膜，直接读取灰度图
                self.current_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # 确保是二值图像
                _, self.current_mask = cv2.threshold(self.current_mask, 127, 255, cv2.THRESH_BINARY)

            print(f"成功加载掩膜，尺寸: {self.current_mask.shape}")
        else:
            # 如果没有找到掩膜，创建空掩膜
            self.current_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
            print(f"警告: 未找到掩膜文件，创建空白掩膜，尺寸: {self.current_mask.shape}")

        # 保存掩膜路径信息
        self.current_mask_path = mask_path
        self.current_subfolder = subfolder
        self.current_image_name = image_name

        # 清空历史记录
        self.history = []
        self.save_to_history()

    def display_image(self):
        """显示图片（原图叠加掩膜或二值图）"""
        if self.current_image is None or self.current_mask is None:
            return

        display_img = self.current_image.copy()

        if self.show_binary:
            # 显示二值图
            binary_rgb = cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2BGR)
            display_img = binary_rgb

            # 更新信息显示
            mask_pixels = np.sum(self.current_mask > 0)
            total_pixels = self.current_mask.size
            mask_percentage = (mask_pixels / total_pixels) * 100
            self.info_label.config(text=f"二值模式 | 掩膜像素: {mask_pixels}/{total_pixels} ({mask_percentage:.1f}%)")
        else:
            # 创建彩色掩膜（红色半透明）
            mask_color = np.zeros_like(display_img)
            mask_color[:, :, 2] = self.current_mask  # 红色通道

            # 将掩膜叠加到原图
            alpha = 0.3
            mask_area = self.current_mask > 0
            # 确保掩膜区域有数据
            if np.any(mask_area):
                display_img[mask_area] = cv2.addWeighted(
                    display_img[mask_area], 1 - alpha,
                    mask_color[mask_area], alpha, 0
                )

            # 更新信息显示
            mask_type = "Seg掩膜" if self.current_mask_type == "seg" else "Contour掩膜"
            erase_mode = "擦除模式" if self.erase_mode == "erase" else "添加模式"
            self.info_label.config(text=f"{mask_type} | {erase_mode} | 橡皮擦大小: {self.eraser_size}")

        # 转换颜色空间 BGR -> RGB
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        # 调整图像大小以适应画布
        h, w = display_img.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w > 1 and canvas_h > 1:  # 确保画布已创建
            scale = min(canvas_w / w, canvas_h / h) * 0.95  # 留出边距
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 转换为PIL图像并显示
        self.display_img_pil = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(self.display_img_pil)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            image=self.photo,
            anchor=tk.CENTER
        )

        # 重新创建橡皮擦矩形并更新位置
        self.eraser_rect = self.canvas.create_rectangle(0, 0, 0, 0,
                                                        outline="white",
                                                        width=1,
                                                        fill="",  # 透明填充
                                                        state="hidden")  # 初始隐藏

        # 如果鼠标在画布内，更新橡皮擦位置
        if self.eraser_visible:
            self.update_eraser_position()

    def start_drag(self, event):
        """开始拖动橡皮擦"""
        self.dragging = True
        self.last_point = (event.x, event.y)

        # 保存当前状态到历史记录
        self.save_to_history()

        self.erase_at_point(event.x, event.y)
        self.display_image()

    def drag(self, event):
        """拖动橡皮擦"""
        if self.dragging and self.last_point:
            self.erase_between_points(self.last_point, (event.x, event.y))
            self.last_point = (event.x, event.y)
            self.display_image()

    def end_drag(self, event):
        """结束拖动"""
        self.dragging = False
        self.last_point = None

    def erase_at_point(self, x, y):
        """在指定点擦除"""
        if self.current_mask is None:
            return

        # 计算图像在画布上的位置
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # 确保有PIL图像
        if not hasattr(self, 'display_img_pil') or self.display_img_pil is None:
            return

        img_w, img_h = self.display_img_pil.size

        # 计算图像在画布中的偏移
        offset_x = (canvas_w - img_w) // 2
        offset_y = (canvas_h - img_h) // 2

        img_x = x - offset_x
        img_y = y - offset_y

        # 检查是否在图像范围内
        if img_x < 0 or img_x >= img_w or img_y < 0 or img_y >= img_h:
            return

        # 转换到原始图像坐标
        scale_x = self.current_image.shape[1] / img_w
        scale_y = self.current_image.shape[0] / img_h

        orig_x = int(img_x * scale_x)
        orig_y = int(img_y * scale_y)

        # 确保坐标在图像范围内
        orig_x = max(0, min(orig_x, self.current_image.shape[1] - 1))
        orig_y = max(0, min(orig_y, self.current_image.shape[0] - 1))

        # 应用橡皮擦
        radius = self.eraser_size // 2
        x1 = max(0, orig_x - radius)
        y1 = max(0, orig_y - radius)
        x2 = min(self.current_image.shape[1], orig_x + radius)
        y2 = min(self.current_image.shape[0], orig_y + radius)

        # 根据模式执行擦除或添加
        if self.erase_mode == "erase":
            # 擦除（将白色改为黑色）
            self.current_mask[y1:y2, x1:x2] = 0
        else:
            # 添加（将黑色改为白色）
            self.current_mask[y1:y2, x1:x2] = 255

    def erase_between_points(self, point1, point2):
        """在两点之间连续擦除"""
        x1, y1 = point1
        x2, y2 = point2

        # 简单的线性插值
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return

        for i in range(steps + 1):
            t = i / steps
            x = int(x1 * (1 - t) + x2 * t)
            y = int(y1 * (1 - t) + y2 * t)
            self.erase_at_point(x, y)

    def update_eraser_size(self, value):
        """更新橡皮擦大小"""
        self.eraser_size = int(float(value))
        self.size_label.config(text=str(self.eraser_size))

    def update_erase_mode(self):
        """更新擦除模式"""
        self.erase_mode = self.erase_mode_var.get()
        self.display_image()

    def set_mask_type(self, mask_type):
        """设置当前掩膜类型"""
        self.current_mask_type = mask_type
        if hasattr(self, 'current_subfolder') and hasattr(self, 'current_image_name'):
            self.load_mask(self.current_subfolder, self.current_image_name)
            self.display_image()

    def toggle_binary_view(self):
        """切换二值图显示"""
        self.show_binary = self.binary_var.get()
        self.display_image()

    def toggle_auto_save(self):
        """切换自动保存状态"""
        self.auto_save = self.auto_save_var.get()
        status = "开启" if self.auto_save else "关闭"
        self.update_status(f"自动保存已{status}")

    def save_to_history(self):
        """保存当前掩膜状态到历史记录"""
        if self.current_mask is not None:
            # 深拷贝当前掩膜
            mask_copy = self.current_mask.copy()
            self.history.append(mask_copy)

            # 限制历史记录数量
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

    def undo(self):
        """撤销上一次操作"""
        if len(self.history) > 1:
            # 移除当前状态
            self.history.pop()
            # 恢复上一个状态
            prev_mask = self.history[-1]
            self.current_mask = prev_mask.copy()
            self.display_image()
            self.update_status("已撤销上一次操作")
        else:
            self.update_status("没有可撤销的操作")

    def save_mask(self):
        """保存当前掩膜"""
        if self.current_mask is None:
            messagebox.showwarning("警告", "没有掩膜可保存")
            return

        # 确保掩膜目录存在
        os.makedirs(os.path.dirname(self.current_mask_path), exist_ok=True)

        # 保存掩膜
        cv2.imwrite(self.current_mask_path, self.current_mask)
        self.update_status(f"掩膜已保存: {self.current_mask_path}")

        # 保存后清空历史记录
        self.history = [self.current_mask.copy()]

    def auto_save_mask(self):
        """自动保存当前掩膜（不显示消息框）"""
        if self.current_mask is not None and hasattr(self, 'current_mask_path'):
            # 确保掩膜目录存在
            os.makedirs(os.path.dirname(self.current_mask_path), exist_ok=True)

            # 保存掩膜
            cv2.imwrite(self.current_mask_path, self.current_mask)
            print(f"自动保存: {self.current_mask_path}")
            return True
        return False

    def prev_image(self):
        """上一张图片"""
        if len(self.image_paths) == 0:
            return

        # 如果开启了自动保存，先保存当前掩膜
        if self.auto_save and self.current_mask is not None:
            if self.auto_save_mask():
                self.update_status("已自动保存当前掩膜")

        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.image_entry.delete(0, tk.END)
        self.image_entry.insert(0, self.image_paths[self.current_index])
        self.load_image()

    def next_image(self):
        """下一张图片"""
        if len(self.image_paths) == 0:
            return

        # 如果开启了自动保存，先保存当前掩膜
        if self.auto_save and self.current_mask is not None:
            if self.auto_save_mask():
                self.update_status("已自动保存当前掩膜")

        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.image_entry.delete(0, tk.END)
        self.image_entry.insert(0, self.image_paths[self.current_index])
        self.load_image()

    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
        print(f"状态: {message}")

    def update_eraser_position(self, event=None):
        """更新橡皮擦可视化矩形的位置"""
        if event:
            self.eraser_x = event.x
            self.eraser_y = event.y
            self.eraser_visible = True

        # 计算矩形的大小和位置
        size = int(self.eraser_size * 3)  # 放大3倍
        x1 = self.eraser_x - size // 2
        y1 = self.eraser_y - size // 2
        x2 = self.eraser_x + size // 2
        y2 = self.eraser_y + size // 2

        # 更新矩形的位置和大小
        self.canvas.coords(self.eraser_rect, x1, y1, x2, y2)
        self.canvas.itemconfig(self.eraser_rect, state="normal")

    def hide_eraser(self, event=None):
        """隐藏橡皮擦可视化矩形"""
        self.eraser_visible = False
        self.canvas.itemconfig(self.eraser_rect, state="hidden")


def main():
    root = tk.Tk()
    app = MaskEraserTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()