# loaders/yolo_loader.py
import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
from sua_bdd_tool.utils.file_opt import find_all_images

class YoloLoader:
    def __init__(self, img_dir, txt_dir, class_path, target_cls_ids=None, exif_db=None):
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.exif_db = exif_db

        # 1. 读取完整类别列表
        self.full_classes = self._read_classes(class_path)
        self.target_cls_ids = target_cls_ids if target_cls_ids else list(range(len(self.full_classes)))
        
        # 2. 计算实际关注的类别名称列表 (用于报告显示)
        self.target_class_names = [
            self.full_classes[i] for i in self.target_cls_ids 
            if 0 <= i < len(self.full_classes)
        ]

    def _read_classes(self, path):
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        return []

    def _yolo_norm_to_pixel(self, yolo_line, img_w, img_h):
        """解析单行 YOLO 格式"""
        parts = yolo_line.strip().split()
        cls_id = int(parts[0])
        
        if len(parts) >= 6:
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5])
            uid = int(parts[6])
        else:
            xc, yc, w, h = map(float, parts[1:5])
            conf = 1.0
            uid = None

        x_center = xc * img_w
        y_center = yc * img_h
        width = w * img_w
        height = h * img_h
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return [cls_id, conf, x1, y1, x2, y2, uid]

    def load(self):
        """
        加载数据并返回标准格式列表
        Returns: List of dict {'image_path': str, 'detections': np.array}
        """

        img_paths = find_all_images(self.img_dir)
        print(f"[Loader] Found {len(img_paths)} images in {self.img_dir}")

        # ✅ 1. 初始化统计器
        raw_counter = Counter()    # 统计 txt 文件里实际存在的 ID
        final_counter = Counter()  # 统计通过筛选后保留的 ID
        total_boxes_raw = 0

        data_list = []
        for img_path in img_paths:
            stem = Path(img_path).stem
            txt_path = os.path.join(self.txt_dir, stem + '.txt')
            detections = []
            
            # (读取图片尺寸部分省略，保持原样)
            with Image.open(img_path) as img:
                w, h = img.size
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            det = self._yolo_norm_to_pixel(line, w, h)
                            
                            # det[0] 是 cls_id
                            cls_id = int(det[0])

                            # ✅ 2. 在筛选前统计（这是最真实的 txt 数据）
                            raw_counter[cls_id] += 1
                            total_boxes_raw += 1

                            # === 类别筛选逻辑 ===
                            if self.target_cls_ids is not None:
                                # 🔍 重点怀疑对象：如果 v32 对应的 ID 不在这里，就被 continue 扔掉了
                                if cls_id not in self.target_cls_ids:
                                    continue 
                            
                            # ✅ 3. 在筛选后统计
                            final_counter[cls_id] += 1
                            detections.append(det)
            
            data_list.append({
                'image_path': img_path,
                'detections': np.array(detections) if detections else np.array([])
            })
            
        # ✅ 4. 打印诊断报告 (这里会告诉你 v32 去哪了)
        print("\n" + "="*50)
        print(f"📊 [Loader Statistic Report]")
        print(f"   - 原始检测框总数 (Raw): {total_boxes_raw}")
        print(f"   - 筛选后保留总数 (Final): {sum(final_counter.values())}")
        print(f"   - 目标 ID 列表 (target_cls_ids): {self.target_cls_ids}")
        print("-" * 50)
        print(f"{'Class ID':<10} | {'原始数量':<10} | {'最终数量':<10} | {'状态'}")
        print("-" * 50)
        
        # 遍历所有出现过的 ID
        all_ids = sorted(raw_counter.keys())
        for cid in all_ids:
            raw_count = raw_counter[cid]
            final_count = final_counter[cid]
            
            status = "✅ 正常"
            if raw_count > 0 and final_count == 0:
                status = "❌ 被过滤 (不在 target_ids 中)"
            elif raw_count != final_count:
                status = "⚠️ 部分过滤"
                
            print(f"{cid:<10} | {raw_count:<10} | {final_count:<10} | {status}")
            
        print("="*50 + "\n")

        return data_list


class DedupLoader(YoloLoader):
    """
    专门加载 yolo_dedup.py 生成的 labels_dedup_fuse 数据
    格式: class cx cy w h conf id
    """
    def _yolo_norm_to_pixel(self, yolo_line, img_w, img_h):
        parts = yolo_line.strip().split()
        
        # 解析逻辑
        if len(parts) >= 7:
            # 标准 7 列格式
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5])
            track_id = int(parts[6])
        elif len(parts) == 6:
            # 缺失 conf 的 6 列格式
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
            conf = 1.0 # 默认置信度
            track_id = int(parts[5])
        else:
            # 异常回退
            return super()._yolo_norm_to_pixel(yolo_line, img_w, img_h) + [-1]

        # 坐标转换 (Norm -> Pixel)
        x_center = xc * img_w
        y_center = yc * img_h
        width = w * img_w
        height = h * img_h
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # 统一输出标准 7 维格式: [cls, conf, x1, y1, x2, y2, id]
        return [cls_id, conf, x1, y1, x2, y2, track_id]

    def load(self):
        # 复用父类逻辑，但因为 _yolo_norm_to_pixel 变了，
        # 返回的 detections numpy array 会多一列
        return super().load()


class DedupLoader_AuxImage(DedupLoader):
    def __init__(self, img_dir, img_aux_dir, txt_dir, class_path, target_cls_ids=None, exif_db=None, img_aux_vis_dir=None):
        super().__init__(img_dir, txt_dir, class_path, target_cls_ids, exif_db)
        self.img_aux_dir = img_aux_dir
        self.img_aux_vis_dir = img_aux_vis_dir

    def load(self):
        """
        加载数据并返回标准格式列表
        Returns: List of dict {'image_path': str, 'detections': np.array}
        """

        img_paths = find_all_images(self.img_dir)
        img_aux_paths = find_all_images(self.img_aux_dir)
        print(f"[Loader] Found {len(img_paths)} images in {self.img_dir}")
        print(f"[Loader] Found {len(img_aux_paths)} aux images in {self.img_aux_dir}") if img_aux_paths else print(f"[Loader] No aux images found in {self.img_aux_dir}")
        
        # ✅ 1. 初始化统计器
        raw_counter = Counter()    # 统计 txt 文件里实际存在的 ID
        final_counter = Counter()  # 统计通过筛选后保留的 ID
        total_boxes_raw = 0

        data_list = []
        for idx, img_path in enumerate(img_paths):
            img_aux_path = img_aux_paths[idx] if img_aux_paths else None
            stem = Path(img_path).stem
            txt_path = os.path.join(self.txt_dir, stem + '.txt')
            detections = []
            
            # (读取图片尺寸部分省略，保持原样)
            with Image.open(img_path) as img:
                w, h = img.size
            
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip():
                            det = self._yolo_norm_to_pixel(line, w, h)
                            
                            # det[0] 是 cls_id
                            cls_id = int(det[0])

                            # ✅ 2. 在筛选前统计（这是最真实的 txt 数据）
                            raw_counter[cls_id] += 1
                            total_boxes_raw += 1

                            # === 类别筛选逻辑 ===
                            if self.target_cls_ids is not None:
                                # 🔍 重点怀疑对象：如果 v32 对应的 ID 不在这里，就被 continue 扔掉了
                                if cls_id not in self.target_cls_ids:
                                    continue 
                            
                            # ✅ 3. 在筛选后统计
                            final_counter[cls_id] += 1
                            detections.append(det)
            
            data_list.append({
                'image_path': img_path,
                'image_aux_path': img_aux_path,
                'detections': np.array(detections) if detections else np.array([])
            })
            
        # ✅ 4. 打印诊断报告 (这里会告诉你 v32 去哪了)
        print("\n" + "="*50)
        print(f"📊 [Loader Statistic Report]")
        print(f"   - 原始检测框总数 (Raw): {total_boxes_raw}")
        print(f"   - 筛选后保留总数 (Final): {sum(final_counter.values())}")
        print(f"   - 目标 ID 列表 (target_cls_ids): {self.target_cls_ids}")
        print("-" * 50)
        print(f"{'Class ID':<10} | {'原始数量':<10} | {'最终数量':<10} | {'状态'}")
        print("-" * 50)
        
        # 遍历所有出现过的 ID
        all_ids = sorted(raw_counter.keys())
        for cid in all_ids:
            raw_count = raw_counter[cid]
            final_count = final_counter[cid]
            
            status = "✅ 正常"
            if raw_count > 0 and final_count == 0:
                status = "❌ 被过滤 (不在 target_ids 中)"
            elif raw_count != final_count:
                status = "⚠️ 部分过滤"
                
            print(f"{cid:<10} | {raw_count:<10} | {final_count:<10} | {status}")
            
        print("="*50 + "\n")

        return data_list