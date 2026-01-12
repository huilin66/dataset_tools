import os
from pathlib import Path
from PIL import Image


def update_labels(
    classes_file_path,
    corrected_root_dir,
    original_label_dir,
    output_label_dir
):
    # 1. 读取 classes.txt，建立 {类别名: 类别ID} 的映射
    # 注意：YOLO ID 是基于行号的，从 0 开始
    class_map = {}
    try:
        with open(classes_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                class_name = line.strip()
                if class_name:
                    class_map[class_name] = idx
        print(f"✅ 已加载类别表，共 {len(class_map)} 个类别: {class_map}")
    except Exception as e:
        print(f"❌ 读取 classes.txt 失败: {e}")
        return

    # 2. 扫描矫正后的数据，建立修改清单
    # 结构: corrections[文件名_不带后缀] = { 目标框索引ID: 新的类别ID }
    corrections = {}
    
    # 遍历 corrected_root_dir 下的每个文件夹（文件夹名即为类别名）
    print("\n🔍 正在扫描矫正文件夹...")
    corrected_root = Path(corrected_root_dir)
    
    for class_dir in corrected_root.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        
        # 检查文件夹名是否在 classes.txt 中
        if class_name not in class_map:
            print(f"⚠️ 跳过未知类别文件夹: {class_name} (不在 classes.txt 中)")
            continue
            
        new_class_id = class_map[class_name]
        
        # 遍历该类别下的所有图片
        for img_file in class_dir.glob("*.JPG"): # 根据需要修改后缀，如 *.png
            # 解析文件名: 原文件名_ID.JPG
            # 使用 rsplit 从右边切分一次，确保原文件名里有下划线也不受影响
            stem = img_file.stem # 去掉后缀的文件名，如 "image_01_3"
            
            try:
                original_name, obj_id_str = stem.rsplit('_', 1)
                obj_id = int(obj_id_str)
                
                if original_name not in corrections:
                    corrections[original_name] = {}
                
                # 记录：这张图的这个ID，应该改成这个新类别
                corrections[original_name][obj_id] = new_class_id
                
            except ValueError:
                print(f"⚠️ 文件名格式无法解析，跳过: {img_file.name} (预期格式: filename_id.JPG)")

    print(f"✅ 扫描完成，共涉及 {len(corrections)} 个原文件的修改。")

    # 3. 处理标签文件
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    print("\n📝 正在更新标签文件...")
    original_root = Path(original_label_dir)
    count_processed = 0
    count_modified = 0

    # 遍历原标签目录中的所有 txt
    for txt_file in original_root.glob("*.txt"):
        if txt_file.name == "classes.txt": continue # 跳过自身

        file_stem = txt_file.stem # 不带后缀的文件名
        
        # 读取原内容
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        is_file_modified = False
        
        # 逐行处理（每一行是一个目标框）
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if not parts: 
                continue
                
            # 检查是否有针对 (文件名, 行号/ID) 的修正记录
            if file_stem in corrections and idx in corrections[file_stem]:
                new_class_id = corrections[file_stem][idx]
                old_class_id = int(parts[0])
                
                if old_class_id != new_class_id:
                    # 修改类别 ID (parts[0])
                    parts[0] = str(new_class_id)
                    line = " ".join(parts) + "\n"
                    is_file_modified = True
            
            new_lines.append(line.strip() + "\n") # 保持格式

        # 写入新文件
        output_path = os.path.join(output_label_dir, txt_file.name)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        count_processed += 1
        if is_file_modified:
            count_modified += 1

    print("-" * 30)
    print(f"🎉 处理完成！")
    print(f"原标签目录: {original_label_dir}")
    print(f"新标签目录: {output_label_dir}")
    print(f"共扫描文件: {count_processed}")
    print(f"产生修改的文件: {count_modified}")




def remap_yolo_labels(
    old_classes_path,
    new_classes_path,
    src_label_dir,
    dst_label_dir
):
    # 1. 加载 旧类别表 (ID -> Name)
    old_id_to_name = {}
    try:
        with open(old_classes_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, name in enumerate(lines):
                old_id_to_name[idx] = name.strip()
        print(f"📖 旧类别表加载完成: 共 {len(old_id_to_name)} 类")
    except Exception as e:
        print(f"❌ 无法读取旧类别文件: {e}")
        return

    # 2. 加载 新类别表 (Name -> ID)
    name_to_new_id = {}
    try:
        with open(new_classes_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, name in enumerate(lines):
                name = name.strip()
                if name: # 忽略空行
                    name_to_new_id[name] = idx
        print(f"📖 新类别表加载完成: 共 {len(name_to_new_id)} 类")
    except Exception as e:
        print(f"❌ 无法读取新类别文件: {e}")
        return

    # 3. 生成 映射字典 (Old ID -> New ID)
    # 只有在新表中存在的类别才会被迁移，否则会被丢弃
    id_map = {}
    dropped_classes = []
    
    for old_id, name in old_id_to_name.items():
        if name in name_to_new_id:
            id_map[old_id] = name_to_new_id[name]
        else:
            dropped_classes.append(name)
            
    print("\n🔄 ID 映射关系:")
    for old_i, new_i in id_map.items():
        print(f"  Old ID {old_i} ({old_id_to_name[old_i]}) -> New ID {new_i}")
    
    if dropped_classes:
        print(f"⚠️ 注意: 以下类别在新表中不存在，将被从标注中删除: {dropped_classes}")

    # 4. 批量处理 txt 文件
    if not os.path.exists(dst_label_dir):
        os.makedirs(dst_label_dir)
        
    src_path = Path(src_label_dir)
    txt_files = list(src_path.glob("*.txt"))
    print(f"\n🚀 开始转换 {len(txt_files)} 个标注文件...")
    
    processed_count = 0
    
    for txt_file in txt_files:
        if txt_file.name == "classes.txt": continue 

        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            try:
                old_id = int(parts[0])
                
                # 检查这个旧 ID 是否需要保留
                if old_id in id_map:
                    new_id = id_map[old_id]
                    # 替换 ID
                    parts[0] = str(new_id)
                    # 重组字符串 (保持原坐标不变)
                    new_lines.append(" ".join(parts) + "\n")
                else:
                    # 如果该类别被删除了，这行数据直接丢弃
                    pass
                    
            except ValueError:
                print(f"⚠️ 跳过格式错误行: {txt_file.name} -> {line.strip()}")

        # 写入新文件
        # 如果转换后文件为空（例如只包含被删除的类），通常也可以创建一个空文件以免报错
        dst_file_path = os.path.join(dst_label_dir, txt_file.name)
        with open(dst_file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        processed_count += 1

    print(f"\n✅ 转换完成！新文件已保存在: {dst_label_dir}")


def delete_specific_classes(
    classes_file,
    src_dir,
    dst_dir,
    classes_to_delete
):
    # 1. 读取 classes.txt 获取 ID 映射
    # 结果: {'person': 0, 'car': 1, ...}
    name_to_id = {}
    try:
        with open(classes_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]
            for idx, name in enumerate(lines):
                if name:
                    name_to_id[name] = idx
    except Exception as e:
        print(f"❌ 无法读取 classes.txt: {e}")
        return

    # 2. 确定要删除的 ID 列表
    ids_to_remove = []
    print(f"准备删除以下类别: {classes_to_delete}")
    
    for name in classes_to_delete:
        if name in name_to_id:
            target_id = name_to_id[name]
            ids_to_remove.append(target_id)
            print(f"  - 找到类别 '{name}' -> 对应 ID: {target_id}")
        else:
            print(f"  ⚠️ 警告: 类别 '{name}' 不在 classes.txt 中，忽略。")

    if not ids_to_remove:
        print("没有有效的类别需要删除，程序退出。")
        return

    # 3. 处理标注文件
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_path = Path(src_dir)
    txt_files = list(src_path.glob("*.txt"))
    
    count_deleted_boxes = 0
    
    print(f"\n🚀 开始处理 {len(txt_files)} 个文件...")

    for txt_file in txt_files:
        if txt_file.name == "classes.txt": continue

        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            try:
                class_id = int(parts[0])
                
                # 核心判断：如果 ID 在删除列表中，就跳过
                if class_id in ids_to_remove:
                    count_deleted_boxes += 1
                    continue 
                
                # 否则保留
                new_lines.append(line)
                
            except ValueError:
                continue

        # 写入新文件 (即使为空也写入，保持数据集完整性)
        dst_file_path = os.path.join(dst_dir, txt_file.name)
        with open(dst_file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    print("-" * 30)
    print(f"✅ 完成！")
    print(f"共删除了 {count_deleted_boxes} 个目标框。")
    print(f"清理后的标签已保存至: {dst_dir}")


def merge_yolo_classes(
    classes_file,
    src_dir,
    dst_dir,
    merge_config
):
    """
    merge_config 格式:
    {
        '要被合并的旧类别名': '保留的目标类别名',
        'truck': 'car',      # 把 truck 变成 car
        'van': 'car'         # 把 van 也变成 car
    }
    """
    
    # 1. 读取 classes.txt 获取 ID 映射
    name_to_id = {}
    try:
        with open(classes_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]
            for idx, name in enumerate(lines):
                if name:
                    name_to_id[name] = idx
    except Exception as e:
        print(f"❌ 无法读取 classes.txt: {e}")
        return

    # 2. 建立 ID 转换表 (Old ID -> Target ID)
    id_map = {} # key: old_id, value: target_id
    
    print("\n📋 合并计划:")
    for src_name, dst_name in merge_config.items():
        if src_name not in name_to_id:
            print(f"  ⚠️ 警告: 源类别 '{src_name}' 不在 classes.txt 中，跳过。")
            continue
        if dst_name not in name_to_id:
            print(f"  ❌ 错误: 目标类别 '{dst_name}' 不在 classes.txt 中，无法合并！")
            continue
            
        src_id = name_to_id[src_name]
        dst_id = name_to_id[dst_name]
        
        id_map[src_id] = dst_id
        print(f"  🔄 将 ID {src_id} ({src_name}) 合并入 -> ID {dst_id} ({dst_name})")

    if not id_map:
        print("没有有效的合并任务。")
        return

    # 3. 处理文件
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_path = Path(src_dir)
    txt_files = list(src_path.glob("*.txt"))
    
    count_files_changed = 0
    count_boxes_merged = 0
    
    print(f"\n🚀 开始处理 {len(txt_files)} 个文件...")

    for txt_file in txt_files:
        if txt_file.name == "classes.txt": continue

        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        file_modified = False
        
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            try:
                curr_id = int(parts[0])
                
                # 检查当前 ID 是否在“被合并”名单里
                if curr_id in id_map:
                    target_id = id_map[curr_id]
                    parts[0] = str(target_id) # 替换 ID
                    new_lines.append(" ".join(parts) + "\n")
                    
                    file_modified = True
                    count_boxes_merged += 1
                else:
                    # 不需要合并，原样保留
                    new_lines.append(line)
                    
            except ValueError:
                continue

        # 写入新文件
        dst_file_path = os.path.join(dst_dir, txt_file.name)
        with open(dst_file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
            
        if file_modified:
            count_files_changed += 1

    print("-" * 30)
    print(f"✅ 合并完成！")
    print(f"共修改了 {count_files_changed} 个文件。")
    print(f"共合并了 {count_boxes_merged} 个目标框。")
    print(f"新标注保存在: {dst_dir}")



def remove_small_boxes_by_pixel(
    label_dir,
    image_dir,
    dst_label_dir,
    min_pixel_w=10,
    min_pixel_h=10,
    target_class=None
):
    """
    按像素尺寸过滤 YOLO 标注框

    target_class:
        None            -> 对所有类别生效
        int / list / set-> 仅对指定类别生效
    """

    # 统一 target_class 为 set
    if target_class is not None:
        if isinstance(target_class, int):
            target_class = {target_class}
        else:
            target_class = set(target_class)

    os.makedirs(dst_label_dir, exist_ok=True)

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG'}

    total, kept, removed = 0, 0, 0

    for txt in Path(label_dir).glob("*.txt"):
        if txt.name == "classes.txt":
            continue

        # 找对应图片
        img_path = next(
            (Path(image_dir) / (txt.stem + ext)
             for ext in valid_exts
             if (Path(image_dir) / (txt.stem + ext)).exists()),
            None
        )
        if img_path is None:
            print(f"⚠️ 找不到图片: {txt.name}")
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        new_lines = []

        for line in txt.read_text().splitlines():
            total += 1
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(parts[0])
            w_px = float(parts[3]) * img_w
            h_px = float(parts[4]) * img_h

            # 是否需要过滤该类别
            need_filter = (target_class is None) or (cls in target_class)

            if need_filter and (w_px < min_pixel_w or h_px < min_pixel_h):
                removed += 1
                continue

            new_lines.append(line + '\n')
            kept += 1

        (Path(dst_label_dir) / txt.name).write_text(''.join(new_lines))

    print(
        f"✅ 完成：总框 {total} | "
        f"保留 {kept} | "
        f"删除 {removed}"
    )



# ================= 配置区域 =================
if __name__ == "__main__":
    pass
    # CLASSES_OLD_FILE = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12_v2\class_old_order.txt"
    # CLASSES_FILE = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12_v2\class.txt"
    # ORIGINAL_LABEL_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12\labels"

    # CORRECTED_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12\check\mannual_check"
    # OUTPUT_LABEL_DIR_OLD = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12_v3\labels_old_order"
    # OUTPUT_LABEL_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12_v3\labels"

    # V4_LABEL_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12_v4\labels"

    # update_labels(CLASSES_OLD_FILE, CORRECTED_DIR, ORIGINAL_LABEL_DIR, OUTPUT_LABEL_DIR_OLD)
    # remap_yolo_labels(CLASSES_OLD_FILE, CLASSES_FILE, OUTPUT_LABEL_DIR_OLD, OUTPUT_LABEL_DIR)

    # delete_specific_classes(CLASSES_FILE, OUTPUT_LABEL_DIR, V4_LABEL_DIR, ["pending"])

    # CLASSES_FILE = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p3_v4\class.txt"
    # INPUT_LABEL_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p3_v4\labels"
    # OUTPUT_LABEL_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p3_v4\labels"
    # delete_specific_classes(CLASSES_FILE, OUTPUT_LABEL_DIR, OUTPUT_LABEL_DIR,  
    # ["high_line", "high", "medium", "low", "pending", "window", "compressor", "other", "others"])

    # ================= 配置区域 =================

    LBL_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\rgb_selected_3_p12\labels"       # 标注文件夹
    IMG_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\rgb_selected_3_p12\images"       # 图片文件夹
    OUT_DIR = r"\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\rgb_selected_3_p12_v2\labels"   # 输出文件夹
    
    MIN_W = 50  # 最小宽度像素
    MIN_H = 50  # 最小高度像素

    remove_small_boxes_by_pixel(LBL_DIR, IMG_DIR, OUT_DIR, MIN_W, MIN_H)