import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# --- 1. 核心计算逻辑 ---

def calculate_slice_coords(img_h, img_w, slice_h, slice_w, overlap_h, overlap_w):
    coords = []
    y_overlap, x_overlap = int(overlap_h * slice_h), int(overlap_w * slice_w)
    y_min = 0
    while y_min < img_h:
        y_max = min(y_min + slice_h, img_h)
        y_start = max(0, y_max - slice_h)
        x_min = 0
        while x_min < img_w:
            x_max = min(x_min + slice_w, img_w)
            x_start = max(0, x_max - slice_w)
            coords.append([x_start, y_start, x_max, y_max])
            if x_max >= img_w: break
            x_min = x_max - x_overlap
        if y_max >= img_h: break
        y_min = y_max - y_overlap
    return coords

# --- 2. 单张图片处理（增加绝对路径强制转换和写入检查） ---

def process_single_image(img_name, input_img_dir, input_txt_dir, output_img_dir, output_txt_dir, 
                         slice_w, slice_h, overlap_w, overlap_h):
    save_count, skip_count = 0, 0
    stem = Path(img_name).stem
    ext = Path(img_name).suffix
    
    # 确保使用绝对路径防止多进程上下文丢失
    img_path = os.path.join(input_img_dir, img_name)
    txt_path = os.path.join(input_txt_dir, stem + ".txt")

    try:
        img_pil = Image.open(img_path)
        img_w, img_h = img_pil.size
    except:
        return 0, 0

    slice_coords = calculate_slice_coords(img_h, img_w, slice_h, slice_w, overlap_h, overlap_w)

    # 加载原图标注
    labels = None
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        try:
            data = np.loadtxt(txt_path)
            labels = data.reshape(-1, 5) if data.ndim == 1 else data
            abs_labels = np.zeros_like(labels)
            abs_labels[:, 0] = labels[:, 0]
            abs_labels[:, 1] = (labels[:, 1] - labels[:, 3] / 2) * img_w
            abs_labels[:, 2] = (labels[:, 2] - labels[:, 4] / 2) * img_h
            abs_labels[:, 3] = (labels[:, 1] + labels[:, 3] / 2) * img_w
            abs_labels[:, 4] = (labels[:, 2] + labels[:, 4] / 2) * img_h
            labels = abs_labels
        except: labels = None

    for sc in slice_coords:
        xmin, ymin, xmax, ymax = sc
        current_labels = []
        if labels is not None:
            for lb in labels:
                ixmin, iymin = max(lb[1], xmin), max(lb[2], ymin)
                ixmax, iymax = min(lb[3], xmax), min(lb[4], ymax)
                if ixmin < ixmax and iymin < iymax: 
                    w, h = ixmax - ixmin, iymax - iymin
                    cx, cy = (ixmin - xmin) + w / 2, (iymin - ymin) + h / 2
                    current_labels.append(f"{int(lb[0])} {cx/slice_w:.6f} {cy/slice_h:.6f} {w/slice_w:.6f} {h/slice_h:.6f}")

        # 如果有目标则直接保存到最终路径
        if current_labels:
            slice_name = f"{stem}_{xmin}_{ymin}_{xmax}_{ymax}"
            img_save_path = os.path.join(output_img_dir, slice_name + ext)
            txt_save_path = os.path.join(output_txt_dir, slice_name + ".txt")

            img_pil.crop((xmin, ymin, xmax, ymax)).save(img_save_path)
            with open(txt_save_path, "w") as f:
                f.write("\n".join(current_labels))
            save_count += 1
        else:
            skip_count += 1
            
    return save_count, skip_count

# --- 3. 封装后的主函数 ---

def yolo_slice(input_img_dir, input_txt_dir, output_img_dir, output_txt_dir, 
               slice_w=1920, slice_h=1920, overlap_w=0.5, overlap_h=0.5, workers=4):
    """
    适配你原始需求的函数：直接将切片保存到指定的图片和标注文件夹
    """
    # 关键：将路径全部转为绝对路径，解决 UNC 路径无法识别或为空的问题
    input_img_dir = os.path.abspath(input_img_dir)
    input_txt_dir = os.path.abspath(input_txt_dir)
    output_img_dir = os.path.abspath(output_img_dir)
    output_txt_dir = os.path.abspath(output_txt_dir)

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    img_list = [f for f in os.listdir(input_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n正在处理目录: {input_img_dir}")
    print(f"输出目标: {output_img_dir}")

    process_func = partial(
        process_single_image,
        input_img_dir=input_img_dir, input_txt_dir=input_txt_dir,
        output_img_dir=output_img_dir, output_txt_dir=output_txt_dir,
        slice_w=slice_w, slice_h=slice_h, overlap_w=overlap_w, overlap_h=overlap_h
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_func, img_list), total=len(img_list)))
    
    total_saved = sum(r[0] for r in results)
    total_skipped = sum(r[1] for r in results)
    print(f"完成: 保存 {total_saved} 张, 跳过 {total_skipped} 张。")

# --- 4. 执行部分 ---

def generate_slice_split_txt(original_split_file, sliced_img_dir, output_split_file):
    """
    根据原始数据集的划分文件，生成对应的切片数据集划分文件
    
    Args:
        original_split_file: 原始 train.txt 或 val.txt 的路径
        sliced_img_dir: 切片图片存放的绝对路径文件夹
        output_split_file: 生成的 train_slice.txt 或 val_slice.txt 路径
    """
    # 1. 读取原始文件中所有图片的 stem (不含后缀的文件名)
    with open(original_split_file, 'r') as f:
        # 假设原始文件里每一行是一个路径
        original_stems = set(Path(line.strip()).stem for line in f if line.strip())

    # 2. 获取输出目录下所有的切片文件 (支持多种后缀)
    all_sliced_images = [f for f in os.listdir(sliced_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 3. 匹配并写入新文件
    count = 0
    with open(output_split_file, 'w') as f_out:
        for slice_name in all_sliced_images:
            # 解析切片文件名：找到最后一个坐标下划线之前的部分
            # 格式：{original_stem}_{xmin}_{ymin}_{xmax}_{ymax}.jpg
            parts = slice_name.rsplit('_', 4)
            parent_stem = parts[0]
            
            if parent_stem in original_stems:
                # 写入切片的绝对路径
                full_path = os.path.join(os.path.abspath(sliced_img_dir), slice_name)
                f_out.write(full_path + '\n')
                count += 1
    
    print(f"已生成 {output_split_file}, 包含 {count} 个切片路径。")

if __name__ == "__main__":
    # 修改这里为你真正的输入路径
    root_dir = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\rgb_selected_3_p12'
    
    # 修改这里为你真正的输出路径（如果你想要保存在本地，就写本地路径如 C:\output）
    # 如果你保持原样，它就会保存在那个网络地址下
    base_output_dir = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\rgb_selected_3_p12_slice'

    slice_nums = [640, 960, 1280]
    
    orig_train_txt = os.path.join(root_dir, 'train.txt')
    orig_val_txt = os.path.join(root_dir, 'val.txt')

    for slice_num in slice_nums:
        # 在这里精确定义每一轮的输出位置
        slice_dir = f'{base_output_dir}_{slice_num}'
        out_img = os.path.join(slice_dir, 'images')
        out_txt = os.path.join(slice_dir, 'labels')
        
        yolo_slice(
            input_img_dir=os.path.join(root_dir, 'images'),
            input_txt_dir=os.path.join(root_dir, 'labels'),
            output_img_dir=out_img,
            output_txt_dir=out_txt,
            slice_w=slice_num,
            slice_h=slice_num,
            overlap_w=0.1,
            overlap_h=0.1,
            workers=4  # 访问网络路径时，进程数不宜过高，建议 4-8
        )
        generate_slice_split_txt(
            original_split_file=orig_train_txt,
            sliced_img_dir=out_img,
            output_split_file=os.path.join(slice_dir, f'train.txt')
        )
        generate_slice_split_txt(
            original_split_file=orig_val_txt,
            sliced_img_dir=out_img,
            output_split_file=os.path.join(slice_dir, f'val.txt')
        )