import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
from ultralytics import SAM
import os


class SAMSegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM 图像分割工具")
        self.root.geometry("1000x700")

        # 模型相关变量
        self.model_dir = "model"  # 模型存放目录
        self.default_model = "sam2.1_b.pt"  # 默认模型
        self.model = None  # SAM模型实例
        self.available_models = []  # 可用模型列表
        self.selected_model = tk.StringVar(value=self.default_model)  # 选中的模型

        # 图像相关变量
        self.img_path = None
        self.original_img = None  # 原始图像(RGB格式)
        self.display_img = None  # 显示用图像
        self.resized_img = None  # 调整大小后的图像
        self.mask = None  # 分割掩码
        self.imgh, self.imgw = 0, 0  # 原始图像尺寸
        self.new_imgh, self.new_imgw = 0, 0  # 调整后尺寸

        # 标注点相关变量
        self.points = []
        self.labels = []  # 1: 前景, 0: 背景
        self.restrict_shape = (600, 900)

        # 初始化模型列表并加载默认模型
        self.load_model_list()
        # 创建UI
        self.create_widgets()

        self.load_selected_model()


    def create_widgets(self):
        # 创建顶部按钮栏
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        # 模型选择下拉菜单
        model_label = tk.Label(btn_frame, text="选择模型:")
        model_label.pack(side=tk.LEFT, padx=5)

        self.model_combobox = ttk.Combobox(
            btn_frame,
            textvariable=self.selected_model,
            values=self.available_models,
            state="readonly",
            width=20
        )
        self.model_combobox.pack(side=tk.LEFT, padx=5)
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_selected)

        self.select_btn = tk.Button(btn_frame, text="选择图片", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.segment_btn = tk.Button(btn_frame, text="执行分割", command=self.perform_segmentation, state=tk.DISABLED)
        self.segment_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(btn_frame, text="保存抠图", command=self.save_segmentation, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # 添加保存掩码按钮
        self.save_mask_btn = tk.Button(btn_frame, text="保存掩码", command=self.save_mask, state=tk.DISABLED)
        self.save_mask_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(btn_frame, text="清除标注", command=self.clear_annotations, state=tk.DISABLED)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # 提示标签
        self.info_label = tk.Label(btn_frame, text="操作提示: 左键-前景 右键-背景 中键-分割")
        self.info_label.pack(side=tk.RIGHT, padx=5)

        # 创建图像显示区域
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 状态栏（用于显示通知信息）
        self.status_bar = tk.Label(self.root, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.on_left_click)  # 左键点击 - 前景点
        self.canvas.bind("<Button-3>", self.on_right_click)  # 右键点击 - 背景点
        self.canvas.bind("<Button-2>", self.on_middle_click)  # 中键点击 - 执行分割

    def show_status(self, message, duration=3000):
        """在状态栏显示临时通知信息"""
        self.status_bar.config(text=message)
        # 3秒后自动恢复为就绪状态
        self.root.after(duration, lambda: self.status_bar.config(text="就绪"))

    def load_model_list(self):
        """加载model目录下的所有可用模型文件"""
        try:
            # 检查模型目录是否存在，不存在则创建
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
                self.show_status(f"模型目录不存在，已自动创建：{self.model_dir}")
                return

            # 遍历目录下的文件，筛选出.pt文件
            for filename in os.listdir(self.model_dir):
                if filename.endswith(".pt") and os.path.isfile(os.path.join(self.model_dir, filename)):
                    self.available_models.append(filename)


            # 如果没有找到模型文件
            if not self.available_models:
                messagebox.showwarning("警告", f"在{self.model_dir}目录下未找到模型文件(.pt)")
            # 如果默认模型不在列表中，选中第一个模型
            elif self.default_model not in self.available_models:
                self.selected_model.set(self.available_models[0])

        except Exception as e:
            messagebox.showerror("错误", f"加载模型列表失败：{str(e)}")

    def load_selected_model(self):
        """加载选中的模型"""
        if not self.available_models:
            self.model = None
            return

        try:
            model_path = os.path.join(self.model_dir, self.selected_model.get())
            self.model = SAM(model_path)
            # 改为状态栏通知（3秒后自动消失）
            self.show_status(f"模型加载成功：{self.selected_model.get()}")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败：{str(e)}")
            self.model = None

    def on_model_selected(self, event):
        """当选择不同模型时触发"""
        # 清除当前的标注和分割结果
        self.clear_annotations()
        # 加载选中的模型
        self.load_selected_model()

    def select_image(self):
        """选择并加载图像"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if file_path:
            self.img_path = file_path
            # 读取图像并转换为RGB格式
            self.original_img = cv2.imread(file_path)
            self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            self.imgh, self.imgw = self.original_img.shape[:2]

            # 计算调整后的图像尺寸
            self.new_imgh, self.new_imgw = self.calc_new_hw(self.imgh, self.imgw)
            self.resized_img = cv2.resize(self.original_img, (self.new_imgw, self.new_imgh))

            # 重置标注点和掩码
            self.points = []
            self.labels = []
            self.mask = None

            # 更新显示
            self.update_display()
            self.show_status(f"已加载图像：{os.path.basename(file_path)}")

            # 启用按钮
            self.segment_btn.config(state=tk.NORMAL if self.model is not None else tk.DISABLED)
            self.clear_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)
            self.save_mask_btn.config(state=tk.DISABLED)

    def calc_new_hw(self, imgh, imgw, restrict_shape=None):
        """计算调整后的图像尺寸，保持比例"""
        if restrict_shape is None:
            restrict_shape = self.restrict_shape
        max_h, max_w = restrict_shape

        # 计算宽高比例
        ratio = imgw / imgh

        # 根据限制尺寸调整
        if imgh > max_h or imgw > max_w:
            if ratio > 1:  # 宽图
                new_w = min(imgw, max_w)
                new_h = int(new_w / ratio)
            else:  # 高图
                new_h = min(imgh, max_h)
                new_w = int(new_h * ratio)
            return new_h, new_w
        return imgh, imgw  # 不需要调整

    def calc_old_xy(self, xy):
        """将显示坐标转换为原始图像坐标"""
        x, y = xy

        # 计算缩放比例
        scale_x = self.imgw / self.new_imgw
        scale_y = self.imgh / self.new_imgh

        old_x = x * scale_x
        old_y = y * scale_y
        return int(old_x), int(old_y)

    def calc_display_xy(self, xy):
        """将原始图像坐标转换为显示图像坐标"""
        x, y = xy

        # 计算缩放比例
        scale_x = self.new_imgw / self.imgw
        scale_y = self.new_imgh / self.imgh

        display_x = x * scale_x
        display_y = y * scale_y
        return int(display_x), int(display_y)

    def draw_points(self, img):
        """在图像上绘制所有标注点"""
        if not self.points:
            return img

        # 创建可编辑的图像副本
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        for i, (x, y) in enumerate(self.points):
            # 转换为显示坐标
            display_x, display_y = self.calc_display_xy((x, y))

            # 前景点(1)用绿色，背景点(0)用红色
            color = (0, 255, 0) if self.labels[i] == 1 else (255, 0, 0)

            # 绘制点和边框
            draw.ellipse(
                [display_x - 5, display_y - 5, display_x + 5, display_y + 5],
                fill=color
            )
            draw.ellipse(
                [display_x - 7, display_y - 7, display_x + 7, display_y + 7],
                outline=(255, 255, 255),
                width=1
            )

        return draw_img

    def update_display(self):
        """更新画布上的图像显示"""
        if self.resized_img is None:
            return

        # 转换为PIL图像
        pil_img = Image.fromarray(self.resized_img)

        # 绘制标注点
        pil_img = self.draw_points(pil_img)

        # 如果有分割结果，添加分割掩码
        if self.mask is not None:
            # 调整掩码大小以匹配显示图像
            mask_display = cv2.resize(self.mask.astype(np.uint8),
                                      (self.new_imgw, self.new_imgh),
                                      interpolation=cv2.INTER_NEAREST)

            # 创建彩色掩码
            mask_img = Image.new('RGBA', (self.new_imgw, self.new_imgh), (0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_img)

            # 在掩码上绘制蓝色半透明区域
            for y in range(self.new_imgh):
                for x in range(self.new_imgw):
                    if mask_display[y, x] > 0:
                        mask_draw.point((x, y), fill=(255, 200, 100, 128))  # 半透明浅蓝色

            # 合并图像
            pil_img = Image.alpha_composite(pil_img.convert('RGBA'), mask_img)

        # 转换为Tkinter可用的图像格式
        self.display_img = ImageTk.PhotoImage(image=pil_img)

        # 更新画布
        self.canvas.delete("all")
        self.canvas.config(width=self.display_img.width(), height=self.display_img.height())
        self.canvas.create_image(0, 0, image=self.display_img, anchor=tk.NW)

    def on_left_click(self, event):
        """左键点击处理 - 添加/删除前景点"""
        if self.original_img is None:
            return

        # 获取点击坐标
        x, y = event.x, event.y
        original_xy = self.calc_old_xy((x, y))
        x_orig, y_orig = original_xy

        is_del = False
        # 检查是否点击了已有的前景点（用于删除）
        for idx, (px, py) in enumerate(self.points):
            if (px - x_orig) ** 2 + (py - y_orig) ** 2 < 250 and self.labels[idx] == 1:
                self.points.pop(idx)
                self.labels.pop(idx)
                is_del = True
                break

        if not is_del:
            self.points.append([x_orig, y_orig])
            self.labels.append(1)
            self.show_status("已添加前景点标注")

        # 更新显示
        self.update_display()

    def on_right_click(self, event):
        """右键点击处理 - 添加/删除背景点"""
        if self.original_img is None:
            return

        # 获取点击坐标
        x, y = event.x, event.y
        original_xy = self.calc_old_xy((x, y))
        x_orig, y_orig = original_xy

        is_del = False
        # 检查是否点击了已有的背景点（用于删除）
        for idx, (px, py) in enumerate(self.points):
            if (px - x_orig) ** 2 + (py - y_orig) ** 2 < 250 and self.labels[idx] == 0:
                self.points.pop(idx)
                self.labels.pop(idx)
                is_del = True
                break

        if not is_del:
            self.points.append([x_orig, y_orig])
            self.labels.append(0)
            self.show_status("已添加背景点标注")

        # 更新显示
        self.update_display()

    def on_middle_click(self, event):
        """中键点击处理 - 执行分割"""
        self.perform_segmentation()

    def perform_segmentation(self):
        """执行图像分割"""
        if not self.points:
            messagebox.showinfo("提示", "请至少选择一个点后再执行分割")
            return

        if self.img_path is None or self.original_img is None:
            return

        if self.model is None:
            messagebox.showwarning("警告", "未加载模型，请先选择并加载模型")
            return

        try:
            self.show_status("正在执行分割...")
            # 准备输入数据
            tpoints = [self.points]
            tlabels = [self.labels]

            # 执行分割
            result = self.model(self.img_path, points=tpoints, labels=tlabels)
            res = result[0]
            masks = res.masks.data.cpu().numpy()

            # 取第一个掩码作为结果
            if len(masks) > 0:
                # 确保掩码尺寸与原图一致
                if masks[0].shape != (self.imgh, self.imgw):
                    self.mask = cv2.resize(masks[0].astype(np.uint8),
                                           (self.imgw, self.imgh),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    self.mask = masks[0]
                self.update_display()
                self.save_btn.config(state=tk.NORMAL)
                self.save_mask_btn.config(state=tk.NORMAL)
                self.show_status("分割完成！")
            else:
                self.show_status("未找到分割结果")

        except Exception as e:
            messagebox.showerror("错误", f"分割过程中发生错误：{str(e)}")
            self.show_status("分割失败")

    def save_segmentation(self):
        """保存分割结果"""
        if self.mask is None or self.original_img is None:
            return

        try:
            # 确保掩码与原图尺寸完全一致
            if self.mask.shape != (self.imgh, self.imgw):
                mask_resized = cv2.resize(
                    self.mask.astype(np.uint8),
                    (self.imgw, self.imgh),  # 注意是(width, height)
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask_resized = self.mask

            # 创建Alpha通道 (0-255)
            alpha_channel = np.zeros((self.imgh, self.imgw), dtype=np.uint8)
            alpha_channel[mask_resized] = 255  # 前景不透明，背景透明

            # 合并RGB通道和Alpha通道
            rgba_image = np.dstack((self.original_img, alpha_channel))

            # 转换为PIL图像并保存
            pil_img = Image.fromarray(rgba_image, mode='RGBA')

            # 询问保存路径
            default_filename = os.path.splitext(os.path.basename(self.img_path))[0] + "_segmented.png"
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG图像", "*.png"), ("所有文件", "*.*")],
                initialfile=default_filename
            )

            if save_path:
                pil_img.save(save_path)
                self.show_status(f"抠图已保存至：{os.path.basename(save_path)}")

        except Exception as e:
            messagebox.showerror("错误", f"保存过程中发生错误：{str(e)}")
            self.show_status("保存抠图失败")

    def save_mask(self):
        """保存掩码图像（前景为白色，背景为黑色）"""
        if self.mask is None:
            return

        try:
            # 确保掩码与原图尺寸完全一致
            if self.mask.shape != (self.imgh, self.imgw):
                mask_resized = cv2.resize(
                    self.mask.astype(np.uint8),
                    (self.imgw, self.imgh),  # 注意是(width, height)
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask_resized = self.mask

            # 创建掩码图像（0为黑色背景，255为白色前景）
            mask_image = np.zeros((self.imgh, self.imgw), dtype=np.uint8)
            mask_image[mask_resized] = 255  # 前景区域设为白色

            # 转换为PIL图像
            pil_mask = Image.fromarray(mask_image, mode='L')  # 'L'表示灰度图像

            # 询问保存路径
            default_filename = os.path.splitext(os.path.basename(self.img_path))[0] + "_mask.png"
            save_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG图像", "*.png"), ("所有文件", "*.*")],
                initialfile=default_filename
            )

            if save_path:
                pil_mask.save(save_path)
                self.show_status(f"掩码已保存至：{os.path.basename(save_path)}")

        except Exception as e:
            messagebox.showerror("错误", f"保存掩码过程中发生错误：{str(e)}")
            self.show_status("保存掩码失败")

    def clear_annotations(self):
        """清除所有标注点和分割结果"""
        self.points = []
        self.labels = []
        self.mask = None
        self.update_display()
        self.save_btn.config(state=tk.DISABLED)
        self.save_mask_btn.config(state=tk.DISABLED)
        self.show_status("已清除所有标注和分割结果")


if __name__ == "__main__":
    root = tk.Tk()
    app = SAMSegmentationTool(root)
    root.mainloop()