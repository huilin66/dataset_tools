from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import os
import cv2
import numpy as np
from tqdm import tqdm


class DroneImageAligner:
    def __init__(self, vis_shape, therm_shape, params=None):
        """
        初始化对齐器 - 只计算一次 ROI
        :param vis_shape: 可见光图像的 shape (h, w)
        :param therm_shape: 红外图像的 shape (h, w)
        :param params: 字典，包含 scale_bias, offset_x, offset_y 等微调参数
        """
        self.h_v, self.w_v = vis_shape[:2]
        self.h_t, self.w_t = therm_shape[:2]

        # 临时存储用于可视化的图片
        self.vis_img = None
        self.therm_img = None

        # --- 默认参数 (M4T 经验值) ---
        self.params = {
            'base_fov_scale': 38.2 / 73.7, # 基础 FOV 比例
            'scale_bias': 0.88,            # 缩放微调系数
            'offset_x': -10,               # 像素偏移 X
            'offset_y': -5                 # 像素偏移 Y
        }
        if params:
            self.params.update(params)

        # 计算并缓存裁剪区域信息 (只运行一次)
        self._calculate_crop_roi()

    def _calculate_crop_roi(self):
        """计算可见光图像上的裁剪区域 (Region of Interest)"""
        final_scale = self.params['base_fov_scale'] * self.params['scale_bias']
        
        # 目标尺寸 (在可见光图像中的像素大小)
        self.roi_w = int(self.w_v * final_scale)
        # 强制保持红外图像的长宽比
        self.roi_h = int(self.roi_w * (self.h_t / self.w_t))

        # 中心点计算
        center_x, center_y = self.w_v / 2, self.h_v / 2
        adj_center_x = center_x + self.params['offset_x']
        adj_center_y = center_y + self.params['offset_y']

        # 裁剪框左上角坐标 (float)
        self.roi_x1 = adj_center_x - self.roi_w / 2
        self.roi_y1 = adj_center_y - self.roi_h / 2
        
        # 整数坐标用于实际裁剪 (预计算好，供后续反复调用)
        self.crop_x1 = int(max(0, self.roi_x1))
        self.crop_y1 = int(max(0, self.roi_y1))
        self.crop_x2 = int(min(self.w_v, self.roi_x1 + self.roi_w))
        self.crop_y2 = int(min(self.h_v, self.roi_y1 + self.roi_h))

        print(f"Aligner Initialized: Crop Box [{self.crop_x1}:{self.crop_x2}, {self.crop_y1}:{self.crop_y2}]")

    def align_image_data(self, img_data):
        """
        核心处理逻辑：接收图片数据(numpy array)，返回对齐后的图片数据
        """
        cropped = img_data[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
        aligned = cv2.resize(cropped, (self.w_t, self.h_t), interpolation=cv2.INTER_LANCZOS4)
        return aligned

    def process_and_save(self, rgb_path, save_path):
        """
        批量处理专用函数：读取 -> 裁剪 -> 保存
        无需加载红外图片，无需存储 self.vis_img，线程安全
        """
        # 1. 读取
        vis_img = cv2.imread(str(rgb_path))
        if vis_img is None:
            raise FileNotFoundError(f"Read Error: {rgb_path}")
            
        # 2. 对齐
        # 简单检查尺寸是否匹配，防止图片损坏或尺寸突变导致 crash
        if vis_img.shape[:2] != (self.h_v, self.w_v):
            # 可以在这里做 resize 或者抛出警告，这里简单跳过严格检查，直接尝试裁剪
            pass 

        aligned = self.align_image_data(vis_img)
        
        # 3. 保存
        cv2.imwrite(str(save_path), aligned)
        return vis_img, aligned

    # ==========================================
    # 可视化与调试功能 (需要加载图片)
    # ==========================================
    
    def load_images_for_debug(self, vis_path, therm_path):
        """调试或可视化前调用，加载具体图片到内存"""
        self.vis_img = cv2.imread(str(vis_path))
        self.therm_img = cv2.imread(str(therm_path))

    def visualize(self, therm_view, aligned_vis, vis_img=None, mode='overlay', save_path=None):
        """
        多种可视化模式
        :param mode: 
            'overlay': 透明叠加
            'checkerboard': 棋盘格交替
            'side_by_side': 左RGB(Crop)，右Thermal
            'edges': 边缘检测对比 (红外红线，可见光绿线)
            'context': 左原始RGB，右原始RGB+红外框覆盖
        """
        result_img = None

        if mode == 'overlay':
            # 叠加显示
            alpha = 0.6
            result_img = cv2.addWeighted(aligned_vis, alpha, therm_view, 1 - alpha, 0)
            cv2.putText(result_img, "Overlay Mode", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        elif mode == 'checkerboard':
            # 棋盘格显示
            result_img = aligned_vis.copy()
            block_size = 128
            rows = self.h_t // block_size
            cols = self.w_t // block_size
            for r in range(rows + 1):
                for c in range(cols + 1):
                    if (r + c) % 2 == 1:
                        r_s, c_s = r*block_size, c*block_size
                        r_e, c_e = min((r+1)*block_size, self.h_t), min((c+1)*block_size, self.w_t)
                        result_img[r_s:r_e, c_s:c_e] = therm_view[r_s:r_e, c_s:c_e]
            cv2.putText(result_img, "Checkerboard Mode", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif mode == 'side_by_side':
            # 并排显示：左侧裁剪后RGB，右侧红外
            result_img = np.hstack((aligned_vis, therm_view))
            # 画一条中线
            cv2.line(result_img, (self.w_t, 0), (self.w_t, self.h_t), (0, 255, 255), 2)

        elif mode == 'edges':
            # 边缘对比模式 (非常适合检查对齐精度)
            # 1. 提取边缘
            edges_vis = cv2.Canny(aligned_vis, 100, 200)
            edges_therm = cv2.Canny(therm_view, 100, 200)
            
            # 2. 创建彩色遮罩
            h, w = self.h_t, self.w_t
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 可见光边缘 -> 绿色
            canvas[edges_vis > 0] = [0, 255, 0]
            # 红外边缘 -> 红色 (叠加在绿色之上)
            # 使用逻辑或操作或简单的覆盖
            mask_t = edges_therm > 0
            canvas[mask_t] = [0, 0, 255] # BGR: Red
            
            # 混合重叠部分变成黄色
            overlap = (edges_vis > 0) & (edges_therm > 0)
            canvas[overlap] = [0, 255, 255]

            result_img = canvas
            cv2.putText(result_img, "Edge: Green=RGB, Red=Thermal", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        elif mode == 'context_box':
            # 上下文模式：显示原始大图，并在其上框出红外的位置, 原始 RGB + box
            
            # 1. 在原始RGB上画框
            vis_with_box = vis_img.copy()
            pt1 = (self.crop_x1, self.crop_y1)
            pt2 = (self.crop_x2, self.crop_y2)
            cv2.rectangle(vis_with_box, pt1, pt2, (0, 0, 255), 15) # 粗红框

            result_img = vis_with_box

        elif mode == 'context_overlay':
            # 上下文模式：显示原始大图，并在其上框出红外的位置, 原始 RGB + 半透明红外覆盖
            
            # 2. 制作覆盖图
            vis_overlay = vis_img.copy()
            pt1 = (self.crop_x1, self.crop_y1)
            pt2 = (self.crop_x2, self.crop_y2)

            # 将红外图像resize回裁剪区域的大小
            roi_h, roi_w = self.crop_y2 - self.crop_y1, self.crop_x2 - self.crop_x1
            therm_resized = cv2.resize(therm_view, (roi_w, roi_h))
            
            # 叠加
            roi_section = vis_overlay[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
            blended_roi = cv2.addWeighted(roi_section, 0.5, therm_resized, 0.5, 0)
            vis_overlay[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2] = blended_roi
            cv2.rectangle(vis_overlay, pt1, pt2, (0, 255, 0), 10) # 绿框

            # 缩小以便显示 (因为原始图太大，比如 4000px 宽)
            disp_scale = 0.25
            img2_s = cv2.resize(vis_overlay, (0,0), fx=disp_scale, fy=disp_scale)
            
            result_img = vis_overlay

        elif mode == 'context_compare':
            # 上下文模式：显示原始大图，并在其上框出红外的位置
            # 左侧：原始 RGB
            # 右侧：原始 RGB + 半透明红外覆盖
            
            # 1. 在原始RGB上画框
            vis_with_box = vis_img.copy()
            pt1 = (self.crop_x1, self.crop_y1)
            pt2 = (self.crop_x2, self.crop_y2)
            cv2.rectangle(vis_with_box, pt1, pt2, (0, 0, 255), 15) # 粗红框
            
            # 2. 制作覆盖图
            vis_overlay = vis_img.copy()
            # 将红外图像resize回裁剪区域的大小
            roi_h, roi_w = self.crop_y2 - self.crop_y1, self.crop_x2 - self.crop_x1
            therm_resized = cv2.resize(therm_view, (roi_w, roi_h))
            
            # 叠加
            roi_section = vis_overlay[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
            blended_roi = cv2.addWeighted(roi_section, 0.5, therm_resized, 0.5, 0)
            vis_overlay[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2] = blended_roi
            cv2.rectangle(vis_overlay, pt1, pt2, (0, 255, 0), 10) # 绿框

            # 缩小以便显示 (因为原始图太大，比如 4000px 宽)
            disp_scale = 0.25
            img1_s = cv2.resize(vis_with_box, (0,0), fx=disp_scale, fy=disp_scale)
            img2_s = cv2.resize(vis_overlay, (0,0), fx=disp_scale, fy=disp_scale)
            
            result_img = np.hstack((img1_s, img2_s))
        else:
            print("未知模式")
            return

        # 显示或保存
        if save_path is not None:
            cv2.imwrite(str(save_path), result_img)
            # print(f"结果已保存至: {save_path}")
        else:
            # 自动缩放显示以适应屏幕
            display_h = 800
            scale = display_h / result_img.shape[0]
            if scale < 1:
                disp_img = cv2.resize(result_img, (0, 0), fx=scale, fy=scale)
            else:
                disp_img = result_img
                
            cv2.imshow(f"Visualization - {mode}", disp_img)
            print(f"按任意键关闭 {mode} 窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# --- 2. 辅助函数保持不变 ---
def build_global_lookup(json_path):
    # (保持原代码不变)
    print(f"Loading Index: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    lookup = {}
    items = []
    if isinstance(data, list): items = data
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list): items.extend(v)
            elif isinstance(v, dict): items.append(v)
    for item in items:
        r_name, t_name = item.get('rgb_name'), item.get('t_name')
        if r_name and t_name:
            lookup[os.path.basename(r_name)] = os.path.basename(t_name)
    return lookup

# --- 3. 修正后的单任务处理函数 ---
def process_single_pair(aligner_instance, rgb_full_path, t_full_path, save_full_path, align_compare_path=None, align_vis_path=None):
    """
    现在接收预初始化好的 aligner 实例
    """
    try:
        # 1. 核心任务：仅处理 RGB 并保存
        rgb_img, align_vis = aligner_instance.process_and_save(rgb_full_path, save_full_path)
        t_img = cv2.imread(str(t_full_path))
        # 2. (可选) 如果需要生成对比图，需要额外读取红外图
        # 注意：这会显著降低速度，建议仅在调试时开启
        if align_compare_path is not None:
            # 这是一个比较重的操作，需要加锁或者在 aligner 内部小心处理
            # 为了线程安全，这里我们手动读取并调用静态处理逻辑，而不使用 aligner.vis_img 成员变量
            # 简易实现：只做简单的叠加保存，不复用 visualize 里的复杂逻辑以避免状态冲突
            aligner_instance.visualize(t_img, align_vis, rgb_img, mode='context_box', save_path=align_vis_path)
            aligner_instance.visualize(t_img, align_vis, rgb_img, mode='context_compare', save_path=align_compare_path)
            
        return True, "OK"
    except Exception as e:
        return False, f"Error: {str(e)}"

# --- 4. 主流程 ---
def batch_align(json_path, rgb_root, t_root, output_root, align_compare_root=None, align_vis_root=None, num_workers=4):
    rgb_root, t_root = Path(rgb_root), Path(t_root)
    output_root = Path(output_root)
    if align_compare_root: align_compare_root = Path(align_compare_root)
    if align_vis_root: align_vis_root = Path(align_vis_root)

    # Step 1: 索引
    global_lookup = build_global_lookup(json_path)

    # Step 2: 扫描 View
    rgb_views = sorted([d for d in rgb_root.iterdir() if d.is_dir()], key=lambda x: x.name)
    
    # --- 关键修改：预初始化 Aligner ---
    print("正在初始化对齐器参数...")
    # 寻找第一个有效的 RGB 和 T 文件来获取尺寸
    sample_rgb_path = None
    sample_t_path = None
    
    # 简单的两层遍历找到第一对存在的图片
    found_sample = False
    for view in rgb_views:
        for f in view.glob("*.[jJ][pP][gG]"):
            t_name = global_lookup.get(f.name)
            if t_name:
                t_path = t_root / view.name / t_name
                if t_path.exists():
                    sample_rgb_path = f
                    sample_t_path = t_path
                    found_sample = True
                    break
        if found_sample: break
    
    if not found_sample:
        print("未找到任何匹配的 RGB-T 图片对，无法初始化对齐器。")
        return

    # 读取一次尺寸
    img_v_temp = cv2.imread(str(sample_rgb_path))
    img_t_temp = cv2.imread(str(sample_t_path))
    
    # 初始化单例 Aligner (参数可在此处修改)
    aligner_params = {
        'scale_bias': 0.88, 
        'offset_x': -10, 
        'offset_y': -5
    }
    # 传入 Shape 而不是 Path
    global_aligner = DroneImageAligner(img_v_temp.shape, img_t_temp.shape, params=aligner_params)
    print(f"对齐器初始化完成。基准参考: {sample_rgb_path.name}")
    del img_v_temp, img_t_temp # 释放内存

    # Step 3: 遍历处理
    global_stats = {'success': 0, 'fail': 0, 'skip': 0}

    for i, view_dir in enumerate(rgb_views):
        os.makedirs(output_root/view_dir.name, exist_ok=True)
        if align_compare_root: os.makedirs(align_compare_root/view_dir.name, exist_ok=True)
        if align_vis_root: os.makedirs(align_vis_root/view_dir.name, exist_ok=True)

        rgb_files = list(view_dir.glob("*.[jJ][pP][gG]"))
        if not rgb_files: continue
            
        print(f"\n[{i+1}/{len(rgb_views)}] Processing View: {view_dir.name}")

        tasks = []
        for rgb_file in rgb_files:
            t_filename = global_lookup.get(rgb_file.name)
            if not t_filename:
                global_stats['skip'] += 1
                continue
            
            t_file_path = t_root / view_dir.name / t_filename
            save_path = output_root / view_dir.name / rgb_file.name
            cmp_path = align_compare_root / view_dir.name / rgb_file.name if align_compare_root else None
            vis_path = align_vis_root / view_dir.name / rgb_file.name if align_vis_root else None
            
            # 仅检查 T 是否存在，不需要在这里读取
            if not t_file_path.exists():
                global_stats['skip'] += 1
                continue

            # 传入 global_aligner 实例
            tasks.append((global_aligner, rgb_file, t_file_path, save_path, cmp_path, vis_path))

        # 执行任务
        view_success = 0
        
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 注意：这里 process_single_pair 的第一个参数变成了 global_aligner
                futures = {executor.submit(process_single_pair, *args): args[1].name for args in tasks}
                for future in tqdm(as_completed(futures), total=len(tasks), desc="Aligning", leave=False):
                    res, msg = future.result()
                    if res: view_success += 1
                    else: print(f"Fail: {msg}")
        else:
            for args in tqdm(tasks, desc="Aligning", leave=False):
                res, msg = process_single_pair(*args)
                if res: view_success += 1
        
        global_stats['success'] += view_success
        print(f"  -> View Done: {view_success} processed")

    print("\nDone.")

# ================= 使用示例 =================
if __name__ == "__main__":
    # 单元测试模式
    rgb_path = "path/to/test_rgb.JPG" 
    nir_path = "path/to/test_nir.JPG"
    
    if os.path.exists(rgb_path):
        # 1. 模拟获取 shape
        v_shape = cv2.imread(rgb_path).shape
        t_shape = cv2.imread(nir_path).shape
        
        # 2. 初始化
        aligner = DroneImageAligner(v_shape, t_shape)
        
        # 3. 调试可视化 (需要手动加载图片)
        aligner.load_images_for_debug(rgb_path, nir_path)
        aligner.visualize(mode='overlay')
    else:
        print("请设置正确的测试路径或运行 batch_align")