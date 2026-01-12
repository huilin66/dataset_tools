from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import platform

from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm

from sua_bdd_tool.utils import load_class_names

COLOR_PALETTE = [
    (255, 42, 4), (183, 223, 0), (104, 31, 17), (221, 111, 255),
    (79, 68, 255), (0, 237, 204), (68, 243, 0), (255, 0, 189),
    (255, 180, 0), (186, 0, 221), (255, 255, 0), (0, 192, 38)
]

class YoloDetVisualizer:
    def __init__(self, class_file, output_dir=None, crop_dir=None, 
                 vis_method='cv2', font_path=None, font_size=20, color_palette=COLOR_PALETTE):
        self.cats = self._load_classes(class_file)
        # 如果没有传入颜色，使用默认的
        self.color_palette = color_palette
        self.output_dir = Path(output_dir) if output_dir else None
        self.crop_dir = Path(crop_dir) if crop_dir else None
        self.vis_method = vis_method
        
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
    def __init__(self, class_names=None, output_dir=None, vis_method='pil', font_size=20):
        # 1. 调用基类初始化
        # 基类会自动调用 self.font_init() 设置好 self.font
        # 基类会自动调用 self.folder_init() 创建好目录
        # 我们传一个假的 class_file (None)，因为下面我们会手动覆盖 self.cats
        super().__init__(class_file="", color_palette=COLOR_PALETTE, # 假设 COLOR_PALETTE 在外部定义了
                         output_dir=output_dir, vis_method=vis_method, 
                         font_size=font_size)

        # 2. 覆盖类别字典 (因为 dedup 通常直接传 list 进来)
        if class_names and isinstance(class_names, list):
            self.cats = {i: name for i, name in enumerate(class_names)}
        
        # 定义支持的后缀
        self.valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.JPG', '.PNG'}

    def process_dedup_dict(self, dets_by_img, img_dir, save_by_id_dir=None):
        """
        处理去重后的字典数据
        """
        img_dir = Path(img_dir)
        if save_by_id_dir:
            save_by_id_dir = Path(save_by_id_dir)
            
        print(f"🖼️ [DedupVis] Processing {len(dets_by_img)} images...")

        for img_name, dets in tqdm(dets_by_img.items(), desc="Visualizing"):
            # A. 寻找图片
            img_path = img_dir / img_name

            # B. 读取与绘图准备
            try:
                # 统一读取逻辑 (OpenCV读取快)
                img_raw = cv2.imread(str(img_path))
                if img_raw is None: continue
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
                    # 注意：这里不需要调用基类的 get_coords，因为那个是针对归一化坐标的
                    # 如果你的数据是绝对像素值，直接算：
                    x1 = int(px - bw / 2)
                    y1 = int(py - bh / 2)
                    x2 = int(px + bw / 2)
                    y2 = int(py + bh / 2)
                    # 边界保护
                    coords = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                    
                    cat_name = self.cats.get(cls_idx, str(cls_idx))
                    label_text = f"{cat_name}|ID:{obj_id}"

                    # C. 调用基类绘图方法 (复用 draw_box_pil / draw_box_cv2)
                    if self.vis_method == 'pil':
                        # 基类已经有 self.font 和 self.color_pattern
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


def dedup_vis_colored(dets_by_img, img_dir, save_dir, class_names=None, font_size=20, vis=True):
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
        font_size=font_size
    )
    
    # 调用专门的处理逻辑
    visualizer.process_dedup_dict(dets_by_img, img_dir)


def dedup_vis(by_img, img_dir, vis_all_dir, vis_by_id_dir, class_names=None, font_size=20, vis=True):
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
        font_size=font_size
    )

    # 调用处理逻辑，并传入 ID 分发目录
    visualizer.process_dedup_dict(by_img, img_dir, save_by_id_dir=vis_by_id_dir)


def single_vis(args):
    """
    多进程 Worker
    解包参数并调用 Visualizer 的 process_file 方法
    """
    img_path, label_path, vis_instance = args
    vis_instance.process_file(img_path, label_path)

def batch_vis(root, use_pil, font, workers):
    root = Path(root)

    # 1. 实例化 Visualizer (配置和全局资源)
    vis = YoloDetVisualizer(
        class_file=root / 'class.txt',
        output_dir=root / 'vis_output',
        crop_dir=root / 'crop_output',
        vis_method='pil' if use_pil else 'cv2',
        font_path=font,
        font_size=24
    )

    # 2. 扫描文件 (外部控制)
    img_dir = root / 'images'
    lbl_dir = root / 'labels'
    valid_exts = {'.jpg', '.png', '.jpeg', '.bmp', '.tif'}
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts])
    
    print(f"Processing {len(img_files)} images...")

    # 3. 组装任务 (传递 vis 实例)
    # 注意：vis 实例会被 pickle 序列化传给子进程，所以 vis 内部不能持有 打开的文件句柄
    tasks = []
    for img_p in img_files:
        lbl_p = lbl_dir / (img_p.stem + '.txt')
        tasks.append((str(img_p), str(lbl_p), vis))

    # 4. 多进程执行
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(single_vis, tasks), total=len(tasks)))