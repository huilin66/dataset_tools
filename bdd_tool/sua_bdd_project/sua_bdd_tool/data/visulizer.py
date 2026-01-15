from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import random

from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tqdm import tqdm

import config


class YoloDetVisualizer:
    def __init__(self, class_file, output_dir=None, crop_dir=None, 
                 vis_method='cv2', font_path=None, font_size=20, color_palette=config.COLOR_PALETTE, num_workers=1):
        self.cats = self._load_classes(class_file)
        # 如果没有传入颜色，使用默认的
        self.color_palette = color_palette
        self.output_dir = Path(output_dir) if output_dir else None
        self.crop_dir = Path(crop_dir) if crop_dir else None
        self.vis_method = vis_method
        self.num_workers = num_workers

        # 初始化字体
        self._font_init(font_path, font_size)
        # 初始化文件夹
        self._folder_init()

    @staticmethod
    def _load_classes(class_file):
        cats = {}
        if not os.path.exists(class_file):
            return cats
        with open(class_file, 'r', encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
            for idx, line in enumerate(lines):
                # 兼容 "id,name" 或 纯name 格式
                name = line.split(',')[0] if ',' in line else line
                cats[idx] = name.strip()
        return cats

    def _font_init(self, font_path, font_size):
        self.font = None
        self.font_size = font_size
        if self.vis_method == 'pil':
            try:
                if font_path and os.path.exists(font_path):
                    self.font = ImageFont.truetype(font_path, size=font_size)
                else:
                    # 尝试自动寻找系统字体
                    sys_font = "arial.ttf"
                    if platform.system() == "Linux":
                        # 常见的 Linux 字体路径
                        candidates = [
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
                        ]
                        for c in candidates:
                            if os.path.exists(c):
                                sys_font = c
                                break
                    self.font = ImageFont.truetype(sys_font, size=font_size)
            except IOError:
                print("⚠️ Warning: Could not load custom/system font. Using default (small/bitmap).")
                self.font = ImageFont.load_default()

    def _folder_init(self):
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.crop_dir:
            for cat in self.cats.values():
                (self.crop_dir / cat).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_light_color(rgb_color, threshold=0.5):
        r, g, b = rgb_color
        luminance = 0.2126 * (r/255.0) + 0.7152 * (g/255.0) + 0.0722 * (b/255.0)
        return luminance > threshold

    def _get_coords(self, xywh, w_img, h_img):
        x_c, y_c, bw, bh = xywh
        x1 = int((x_c - bw / 2) * w_img)
        y1 = int((y_c - bh / 2) * h_img)
        x2 = int((x_c + bw / 2) * w_img)
        y2 = int((y_c + bh / 2) * h_img)
        return max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)

    def draw_box_cv2(self, img, cls_id, coords, label_name):
        x1, y1, x2, y2 = coords
        color_bgr = self.color_palette[cls_id % len(self.color_palette)]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
        
        text_size = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(img, (x1, y1 + 10), (x1 + text_size[0], y1 + 10 - text_size[1] - 5), color_bgr, -1)
        
        txt_color = (0, 0, 0) if self.is_light_color(color_bgr[::-1]) else (255, 255, 255)
        cv2.putText(img, label_name, (x1, y1 + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

    def draw_box_pil(self, draw, cls_id, coords, label_name):
        x1, y1, x2, y2 = coords
        color_bgr = self.color_palette[cls_id % len(self.color_palette)]
        color_rgb = color_bgr[::-1] # BGR转RGB
        
        draw.rectangle([x1, y1, x2, y2], outline=tuple(color_rgb), width=3)
        
        # 获取文字包围盒
        if hasattr(self.font, 'getbbox'):
            bbox = self.font.getbbox(label_name)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            # 兼容旧版 Pillow
            text_w, text_h = self.font.getsize(label_name)

        txt_x, txt_y = x1, y1
        if txt_y < text_h + 5: 
            txt_y = y2 
        
        draw.rectangle([txt_x, txt_y, txt_x + text_w + 4, txt_y + text_h + 4], fill=tuple(color_rgb))
        
        txt_color = (0, 0, 0) if self.is_light_color(color_rgb) else (255, 255, 255)
        draw.text((txt_x + 2, txt_y), label_name, fill=txt_color, font=self.font)

    def process_file(self, img_path, label_path):
        """
        [核心修改]：将单张文件的读取、解析、绘图、保存封装在这里
        """
        if not os.path.exists(label_path):
            return

        # 1. 读取图像 (统一用CV2读取，效率高)
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            return
        h, w = img_raw.shape[:2]

        # 2. 准备绘图环境
        img_vis_cv2 = None
        pil_draw = None
        pil_img = None

        if self.output_dir:
            if self.vis_method == 'cv2':
                img_vis_cv2 = img_raw.copy()
            else:
                # 懒转换：只在需要PIL时才转
                pil_img = Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
                pil_draw = ImageDraw.Draw(pil_img)

        # 3. 解析 Label 并处理
        with open(label_path, 'r') as f:
            lines = f.read().strip().splitlines()

        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5: continue
            
            cls_id = int(float(parts[0]))
            xywh = [float(x) for x in parts[1:5]]
            
            # 计算坐标
            coords = self._get_coords(xywh, w, h)
            label_name = self.cats.get(cls_id, str(cls_id))

            # A. 处理裁剪 (Numpy操作)
            if self.crop_dir:
                cx1, cy1, cx2, cy2 = coords
                # 简单的越界保护
                cx1, cy1 = max(0, cx1), max(0, cy1)
                cx2, cy2 = min(w, cx2), min(h, cy2)
                
                if cx2 > cx1 and cy2 > cy1:
                    crop = img_raw[cy1:cy2, cx1:cx2]
                    save_name = f"{Path(img_path).stem}_{idx}.jpg"
                    save_path = self.crop_dir / label_name / save_name
                    cv2.imwrite(str(save_path), crop)

            # B. 处理绘图
            if self.output_dir:
                if self.vis_method == 'cv2':
                    self._draw_box_cv2(img_vis_cv2, cls_id, coords, label_name)
                else:
                    self._draw_box_pil(pil_draw, cls_id, coords, label_name)

        # 4. 保存结果图
        if self.output_dir:
            save_path = self.output_dir / Path(img_path).name
            if self.vis_method == 'cv2':
                cv2.imwrite(str(save_path), img_vis_cv2)
            else:
                pil_img.save(str(save_path), quality=95)

class DedupVisualizer(YoloDetVisualizer):
    def __init__(self, class_names=None, output_dir=None, vis_method='pil', font_size=20, num_workers=1):
        # 1. 调用基类初始化
        super().__init__(class_file="", color_palette=config.COLOR_PALETTE, 
                         output_dir=output_dir, vis_method=vis_method, 
                         font_size=font_size, num_workers=num_workers)

        # 2. 覆盖类别字典
        if class_names and isinstance(class_names, list):
            self.cats = {i: name for i, name in enumerate(class_names)}

    def _process_single_img(self, img_name, dets, img_dir, save_by_id_dir):
        """
        [新增] 处理单张图片的具体逻辑 (从原来的循环体中抽离)
        """
        # A. 寻找图片
        img_path = img_dir / img_name

        # B. 读取与绘图准备
        try:
            # 统一读取逻辑 (OpenCV读取快)
            img_raw = cv2.imread(str(img_path))
            if img_raw is None: return # 图片无法读取则跳过
            h, w = img_raw.shape[:2]

            # 准备绘图对象
            pil_img = None
            pil_draw = None
            img_vis_cv2 = None

            if self.vis_method == 'pil':
                pil_img = Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
                pil_draw = ImageDraw.Draw(pil_img)
            else:
                img_vis_cv2 = img_raw.copy()

            ids_in_this_img = []

            for d in dets:
                cls_idx = int(d['cls'])
                obj_id = d['id']
                # 绝对坐标: px, py, bw, bh
                px, py, bw, bh = d['pxpywh']

                # 转换坐标: 中心点wh -> 左上右下xy
                x1 = int(px - bw / 2)
                y1 = int(py - bh / 2)
                x2 = int(px + bw / 2)
                y2 = int(py + bh / 2)
                # 边界保护
                coords = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                
                cat_name = self.cats.get(cls_idx, str(cls_idx))
                label_text = f"{cat_name}|ID:{obj_id}"

                # C. 调用基类绘图方法
                if self.vis_method == 'pil':
                    self.draw_box_pil(pil_draw, cls_idx, coords, label_text)
                else:
                    self.draw_box_cv2(img_vis_cv2, cls_idx, coords, label_text)
                
                ids_in_this_img.append(obj_id)

            # D. 保存总览图
            final_name = img_path.name
            if self.output_dir:
                save_p = self.output_dir / final_name
                if self.vis_method == 'pil':
                    pil_img.save(str(save_p), quality=95)
                else:
                    cv2.imwrite(str(save_p), img_vis_cv2)

            # E. 按 ID 分发 (直接利用内存中的图像对象)
            if save_by_id_dir and ids_in_this_img:
                unique_ids = set(ids_in_this_img)
                for uid in unique_ids:
                    id_folder = save_by_id_dir / f"id_{uid:03d}"
                    id_folder.mkdir(parents=True, exist_ok=True)
                    target_p = id_folder / final_name
                    
                    if self.vis_method == 'pil':
                        pil_img.save(str(target_p), quality=95)
                    else:
                        cv2.imwrite(str(target_p), img_vis_cv2)

        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")

    def process_dedup_dict(self, dets_by_img, img_dir, save_by_id_dir=None):
        """
        [修改] 支持多进程处理
        :param workers: 进程数，1为单进程，>1为多进程
        """
        img_dir = Path(img_dir)
        if save_by_id_dir:
            save_by_id_dir = Path(save_by_id_dir)
            
        total_imgs = len(dets_by_img)
        print(f"🖼️ [DedupVis] Processing {total_imgs} images with {self.num_workers} workers...")

        # 模式 1: 单进程 (调试用，或者量少时用)
        if self.num_workers <= 1:
            for img_name, dets in tqdm(dets_by_img.items(), desc="Visualizing (Single)"):
                self._process_single_img(img_name, dets, img_dir, save_by_id_dir)
        
        # 模式 2: 多进程 (速度快)
        else:
            # 准备参数列表: (self, img_name, dets, img_dir, save_by_id_dir)
            # 注意：self 会被 pickle 序列化传递给子进程，所以确保 self 里没有打开的文件句柄
            tasks = []
            for img_name, dets in dets_by_img.items():
                task_args = (self, img_name, dets, img_dir, save_by_id_dir)
                tasks.append(task_args)
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # 使用 tqdm 包装 executor.map 或迭代 results
                # 2. 提交所有任务 (Submit)
                # submit 会立即返回一个 future 对象，不会阻塞
                futures = [executor.submit(_dedup_worker, t) for t in tasks]
                
                # 3. 按完成顺序更新进度条 (As Completed)
                # 只要有一个任务完成，进度条就走一格，不需要等前面的任务
                for _ in tqdm(as_completed(futures), total=total_imgs, desc=f"Multi-{self.num_workers}"):
                    pass


class FacadeVisualizer:
    def __init__(self, floor_manager=None, font_size=20, color_palette=config.COLOR_PALETTE):
        """
        初始化可视化器
        :param output_dir: 图片保存目录
        :param view_name: 视图名称（用于生成文件名）
        """
        # 预设颜色盘 (避免颜色太乱)
        self.palette = color_palette
        self.color_map = {}
        self.floor_manager = floor_manager

    def _get_color(self, uid):
        """获取颜色，并兼容 (0-255) 格式"""
        color = None
        
        # 1. 获取原始颜色
        if uid == -1:
            color = '#BBBBBB'
        elif uid in self.color_map:
            color = self.color_map[uid]
        else:
            # 分配新颜色
            if len(self.color_map) < len(self.palette):
                color = self.palette[len(self.color_map)]
            else:
                color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
            self.color_map[uid] = color

        # 2. [关键修复] 检查并转换 (0-255) 的 RGB 元组为 (0-1)
        if isinstance(color, (list, tuple)):
            # 如果发现有数值大于 1.0，说明是 255 格式，需要归一化
            if any(c > 1.0 for c in color):
                color = [c / 255.0 for c in color]
                # 确保转换后是 tuple 或 list，且数值在 0-1 之间
                # 如果有 alpha 通道 (RGBA)，通常 alpha 是 0-1 或 0-255，这里简单统一除以 255
                # 但要注意 matplotlib 的 alpha 参数通常是单独控制的
        
        return color

    def load_and_plot(self, json_path, save_path, view_name):
        """
        主函数：读取 JSON 并绘图
        :param json_path: projection_details.json 的路径
        :param floor_manager: (可选) 传入 floor_manager 对象以绘制楼层线
        :param save_suffix: 保存文件名的后缀
        """
        print(f"🎨 [FacadeVisualizer] Loading {json_path} ...")
        
        if not Path(json_path).exists():
            print("❌ JSON file not found.")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data_by_img = json.load(f)

        # 1. 数据扁平化处理
        all_points = []
        unique_ids = set()
        
        for img_name, items in data_by_img.items():
            for item in items:
                proj = item['projection_world']
                raw = item['raw_yolo']
                
                # 计算物理宽度: h_real * (w_pixel / h_pixel)
                # 注意：假设像素长宽比为 1:1
                aspect_ratio = raw['w'] / raw['h'] 
                h_real = proj['h (obj_height_m)']
                w_real = h_real * aspect_ratio

                all_points.append({
                    'x': proj['x (horizontal_m)'],
                    'z': proj['z (height_m)'], # 这是中心点 Z
                    'w': w_real,
                    'h': h_real,
                    'id': item['id'],
                    'floor': item.get('floor', 'N/A')
                })
                unique_ids.add(item['id'])
        min_boundary_x = min([p['x'] - p['w']/2 for p in all_points])

        offset_x = -min_boundary_x
        print(f"📐 Applying X-offset: {offset_x:.2f}m (Shifting negative values to positive)")
        
        for p in all_points:
            p['x_plot'] = p['x'] + offset_x

        if not all_points:
            print("⚠️ No detection data to visualize.")
            return

        # 2. 开始绘图
        self._plot_canvas(save_path, view_name, all_points, unique_ids)

    def _plot_canvas(self, save_path, view_name, points, unique_ids):
        # 动态计算画布大小
        xs = [p['x_plot'] for p in points]
        zs = [p['z'] for p in points]
        x_span = max(xs) - min(xs)
        z_span = max(zs) - min(zs)
        
        # 保持比例，但限制最小尺寸
        fig_w = max(15, x_span / 2) 
        fig_h = max(10, z_span / 2)
        
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_title(f"Facade Map: {view_name} (IDs: {len(unique_ids)})", fontsize=16)
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Absolute Altitude (m)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal') # 关键：保持 1:1 物理比例

        # A. 绘制楼层线 (如果提供了 floor_manager)
        if self.floor_manager:
            # 假设 floor_manager 有一个方法或者属性获取所有楼层高度
            # 这里模拟一下，你需要根据你的 floor_manager 实际结构调整
            # 例如: floor_manager.floors = {'F1': 30.0, 'F2': 33.5}
            if hasattr(self.floor_manager, 'floors_heights'): # 假设是个字典 {name: height}
                 for fname, fheight in self.floor_manager.floors_heights.items():
                     ax.axhline(y=fheight, color='gray', linestyle='-', alpha=0.3, linewidth=1)
                     ax.text(min(xs)-2, fheight, fname, color='gray', va='center', fontsize=8)

        # B. 绘制物体
        groups = {}
        for p in points:
            groups.setdefault(p['id'], []).append(p)

        for uid, group_points in groups.items():
            color = self._get_color(uid)
            
            # 计算该组的几何中心，用于放置标签
            center_x_sum = 0
            center_z_sum = 0
            
            for p in group_points:
                # Matplotlib Rectangle 接受左下角坐标 (x, y)
                x0 = p['x_plot'] - p['w'] / 2
                z0 = p['z'] - p['h'] / 2
                
                # 画框
                rect = patches.Rectangle(
                    (x0, z0), p['w'], p['h'],
                    linewidth=1, edgecolor=color, facecolor=color, alpha=0.5
                )
                ax.add_patch(rect)
                
                # 画中心点
                ax.plot(p['x_plot'], p['z'], marker='.', color='black', markersize=1, alpha=0.5)
                
                center_x_sum += p['x_plot']
                center_z_sum += p['z']

            # C. 绘制 ID 标签 (只在聚类中心画一次)
            if uid != -1:
                avg_x = center_x_sum / len(group_points)
                avg_z = center_z_sum / len(group_points)
                
                # 标签带白底，防遮挡
                ax.text(avg_x, avg_z, str(uid), fontsize=10, color='black', weight='bold',
                        ha='center', va='center', 
                        bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=color, alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig) # 释放内存
        print(f"✅ Visualization saved: {save_path}")


def _dedup_worker(args):
    """
    [新增] 多进程的 Worker 函数
    必须定义在类外部，以便 pickle 序列化
    """
    vis_instance, img_name, dets, img_dir, save_by_id_dir = args
    vis_instance._process_single_img(img_name, dets, img_dir, save_by_id_dir)

def dedup_vis_colored(dets_by_img, img_dir, save_dir, class_names=None, font_size=20, vis=True, num_workers=1):
    """
    优化后的 dedup_vis_colored
    保留了接口签名，内部调用 DedupVisualizer
    """
    if not vis:
        return

    # 实例化派生类
    # 强制使用 PIL 以获得更好的文字效果，也可以改成 'cv2'
    visualizer = DedupVisualizer(
        class_names=class_names, 
        output_dir=save_dir, 
        vis_method='pil',  
        font_size=font_size,
        num_workers=num_workers
    )
    
    # 调用专门的处理逻辑
    visualizer.process_dedup_dict(dets_by_img, img_dir)


def dedup_vis(by_img, img_dir, vis_all_dir, vis_by_id_dir, class_names=None, font_size=20, vis=True, num_workers=1):
    """
    优化后的 dedup_vis (增强版)
    """
    if not vis:
        return
    # 实例化派生类
    visualizer = DedupVisualizer(
        class_names=class_names,
        output_dir=vis_all_dir, # 总览图保存位置
        vis_method='pil',
        font_size=font_size,
        num_workers=num_workers
    )

    # 调用处理逻辑，并传入 ID 分发目录
    visualizer.process_dedup_dict(by_img, img_dir, save_by_id_dir=vis_by_id_dir)


