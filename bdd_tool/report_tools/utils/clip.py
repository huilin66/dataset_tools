import cv2
import numpy as np

def crop_visible_to_match_thermal(vis_img_path, save_path):
    # 读取广角图像
    vis_img = cv2.imread(vis_img_path)
    h_v, w_v = vis_img.shape[:2] # 3024, 4032

    # 1. 计算缩放比例 (基于等效焦距: 24mm / 52mm)
    # 或者基于 FOV 比例计算更精确
    fov_scale_w = 38.2 / 73.7  # 约 0.518
    
    # 2. 目标尺寸 (广角图中对应的红外区域大小)
    target_w = int(w_v * fov_scale_w)
    # 红外是 1280:1024 (5:4)，保持该比例截取
    target_h = int(target_w * (1024 / 1280))

    # 3. 计算中心偏移 (Offset)
    # 理论中心
    center_x, center_y = w_v / 2, h_v / 2
    
    # 偏移修正：红外在广角左上方，则在广角图中，目标区域中心会向左上偏移
    # 以下为 M4T 的经验修正值（单位：像素），具体取决于物距，建议根据结果微调
    # 物距越近，偏移越大；物距无穷远时趋近于 0
    offset_x = -20  # 向左偏移
    offset_y = -15  # 向上偏移
    
    adj_center_x = center_x + offset_x
    adj_center_y = center_y + offset_y

    # 4. 计算裁剪坐标 [x1, y1, x2, y2]
    x1 = int(adj_center_x - target_w / 2)
    y1 = int(adj_center_y - target_h / 2)
    x2 = x1 + target_w
    y2 = y1 + target_h

    # 边界检查
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_v, x2), min(h_v, y2)

    # 5. 执行裁剪并缩放到红外尺寸 (可选)
    cropped_vis = vis_img[y1:y2, x1:x2]
    # 如果需要强制对齐到 1280x1024
    aligned_vis = cv2.resize(cropped_vis, (1280, 1024), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(save_path, aligned_vis)
    print(f"对齐完成。裁剪坐标: ({x1},{y1}) 到 ({x2},{y2})")

def refined_crop_visible(vis_img_path, save_path):
    # 读取图像
    vis_img = cv2.imread(vis_img_path)
    h_v, w_v = vis_img.shape[:2]

    # --- 核心调优参数区 ---
    # 1. 缩放偏置：如果RGB框比红外大，减小这个值（如 0.518 -> 0.48）
    # 理论值是 38.2 / 73.7 ≈ 0.518
    # 如果上下左右都大了一圈，尝试降到 0.49 或 0.50
    base_fov_scale = 38.2 / 73.7 
    scale_bias = 0.88
    offset_x = -10
    offset_y = -5
    
    # scale_bias = 0.88  # 这是一个系数，1.0代表原始比例，<1.0 代表缩小裁剪范围
    # # 2. 位置偏置（单位：像素）
    # # 如果RGB图像相对于红外偏右了，减小 offset_x
    # # 如果RGB图像相对于红外偏下了，减小 offset_y
    # offset_x = -15  # 左右微调
    # offset_y = -10  # 上下微调
    final_scale = base_fov_scale * scale_bias
    

    # -----------------------

    # 计算目标尺寸
    target_w = int(w_v * final_scale)
    # 保持红外的 5:4 比例 (1280/1024 = 1.25)
    target_h = int(target_w / 1.25)

    # 计算中心点坐标
    center_x, center_y = w_v / 2, h_v / 2
    adj_center_x = center_x + offset_x
    adj_center_y = center_y + offset_y

    # 计算裁剪矩形坐标
    x1 = int(adj_center_x - target_w / 2)
    y1 = int(adj_center_y - target_h / 2)
    x2 = x1 + target_w
    y2 = y1 + target_h

    # 边界限制
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_v, x2), min(h_v, y2)

    # 裁剪并缩放到红外标准分辨率
    cropped = vis_img[y1:y2, x1:x2]
    aligned_vis = cv2.resize(cropped, (1280, 1024), interpolation=cv2.INTER_LANCZOS4)

    cv2.imwrite(save_path, aligned_vis)
    print(f"调整完成：Scale={final_scale:.4f}, Offset=({offset_x},{offset_y})")

# 使用示例

def visualize_alignment(thermal_path, aligned_vis_path):
    """
    可视化红外图像与对齐后的可见光图像。
    提供了三种视图：透明混合、假彩色边缘对比、棋盘格对比。
    """
    # 1. 读取图像
    # 注意：DJI的红外JPG通常已经是伪彩色（如IronRed），读取为彩色模式
    thermal_img = cv2.imread(thermal_path, cv2.IMREAD_COLOR)
    aligned_vis_img = cv2.imread(aligned_vis_path, cv2.IMREAD_COLOR)

    if thermal_img is None or aligned_vis_img is None:
        print("错误：无法读取图像，请检查路径。")
        return

    # 2. 确保尺寸严格一致 (以红外图像尺寸为准)
    # 虽然上一步代码做了缩放，这里再加一层保险
    h_t, w_t = thermal_img.shape[:2]
    if aligned_vis_img.shape[:2] != (h_t, w_t):
        print(f"正在微调可见光图像尺寸以匹配红外图像: {w_t}x{h_t}")
        aligned_vis_img = cv2.resize(aligned_vis_img, (w_t, h_t), interpolation=cv2.INTER_LINEAR)

    # ==========================================
    # 模式 A: 透明度混合 (Alpha Blending)
    # ==========================================
    # alpha 值决定了可见光的透明度，0.5 表示两者各占 50%
    alpha = 0.6 
    blended_img = cv2.addWeighted(aligned_vis_img, alpha, thermal_img, 1 - alpha, 0)
    cv2.putText(blended_img, "Mode A: Alpha Blend (50%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ==========================================
    # 模式 B: 假彩色边缘对比 (False Color Composite) - 强烈推荐用于检查对齐
    # ==========================================
    # 原理：将可见光作为背景（灰度），将红外的高亮区域以红色叠加显示。
    # 或者，将可见光放入蓝/绿通道，红外放入红通道。完全对齐区域为灰白，错位区域显色。
    
    # 转换为灰度以便合成
    vis_gray = cv2.cvtColor(aligned_vis_img, cv2.COLOR_BGR2GRAY)
    thermal_gray = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
    
    # 增加对比度以便观察
    vis_gray = cv2.normalize(vis_gray, None, 0, 255, cv2.NORM_MINMAX)
    thermal_gray = cv2.normalize(thermal_gray, None, 0, 255, cv2.NORM_MINMAX)

    # 创建假彩色图像 (BGR空间)
    false_color_img = np.zeros_like(aligned_vis_img)
    # 蓝色通道放可见光
    false_color_img[:, :, 0] = vis_gray 
    # 绿色通道放可见光
    false_color_img[:, :, 1] = vis_gray
    # 红色通道放红外光
    false_color_img[:, :, 2] = thermal_gray 
    
    # 说明：图像整体偏青色（蓝+绿）。
    # 如果物体边缘出现明显的红色或青色重影，说明未对齐。
    # 如果边缘清晰且主要呈灰白色调，说明对齐良好。
    cv2.putText(false_color_img, "Mode B: False Color (Cyan=Vis, Red=Thermal)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # ==========================================
    # 模式 C: 棋盘格对比 (Checkerboard)
    # ==========================================
    # 将图像分割成棋盘格，交替显示
    checkerboard_img = aligned_vis_img.copy()
    block_size = 128 # 棋盘格块的大小
    rows = h_t // block_size
    cols = w_t // block_size
    
    for r in range(rows + 1):
        for c in range(cols + 1):
            # 每一行交替起始块
            if (r + c) % 2 == 1: 
                r_start = r * block_size
                c_start = c * block_size
                r_end = min((r + 1) * block_size, h_t)
                c_end = min((c + 1) * block_size, w_t)
                checkerboard_img[r_start:r_end, c_start:c_end] = thermal_img[r_start:r_end, c_start:c_end]

    cv2.putText(checkerboard_img, "Mode C: Checkerboard View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # ==========================================
    # 显示结果
    # ==========================================
    # 为了方便在屏幕上查看，缩小一点显示 (如果屏幕够大可以注释掉resize)
    display_scale = 0.8
    def resize_for_display(img, scale):
        return cv2.resize(img, (0, 0), fx=scale, fy=scale)

    cv2.imwrite(aligned_vis_path.replace('.JPG', '_blended.png'), resize_for_display(blended_img, display_scale))

    # cv2.imshow("Alignment Check - Alpha Blend", resize_for_display(blended_img, display_scale))
    # cv2.imshow("Alignment Check - False Color (Best for precision)", resize_for_display(false_color_img, display_scale))
    # cv2.imshow("Alignment Check - Checkerboard", resize_for_display(checkerboard_img, display_scale))

    # print("已显示可视化窗口。请按任意键退出。")
    # print("观察重点：在'False Color'视图中，寻找物体边缘是否有明显的红色或青色分离。")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    rgb_path = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\visible\DJI_20251216155617_0520_V.JPG"  # <-- 改成你的路径
    nir_path  = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\thermal\DJI_20251216155618_0520_T.JPG"  # <-- 改成你的路径
    crop_path = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\data\demo.JPG'
    refined_crop_visible(rgb_path, crop_path)
    visualize_alignment(nir_path, crop_path)