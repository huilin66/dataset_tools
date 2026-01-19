from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
import shutil

from tqdm import tqdm


def find_all_images(folder):
    if folder is None:
        return None
    if isinstance(folder, str):
        folder = Path(folder)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"}
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix in exts]
    imgs.sort(key=lambda p: p.name)
    return imgs

def pick_image(folder, index, middle=False):
    imgs = find_all_images(folder)
    if middle:
        index = len(imgs) // 2
    return imgs[index] if imgs else None

def pick_first_image(folder):
    return pick_image(folder, 0)

def pick_last_image(folder):
    return pick_image(folder, -1)

def pick_middle_image(folder):
    return pick_image(folder, None, middle=True)

def check_and_copy(src_folder, check_tree, dst_folder):
    # 1. 准备工作：确保目标文件夹 c 存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
        print(f"已创建目标文件夹: {dst_folder}")

    # 2. 建立索引：获取 b 文件夹及其所有子文件夹下的所有文件名
    # 使用 set (集合) 查找速度最快
    files_in_b = set()
    print("正在扫描文件夹 b ...")
    for root, dirs, files in os.walk(check_tree):
        for file in files:
            files_in_b.add(file)
    
    print(f"文件夹 b (含子目录) 中共有 {len(files_in_b)} 个文件。")

    # 3. 对比并复制：遍历 a 中的文件
    count = 0
    print("正在检查文件夹 a ...")
    for file in os.listdir(src_folder):
        src_path = os.path.join(src_folder, file)
        
        # 确保是文件而不是文件夹
        if os.path.isfile(src_path):
            # 核心判断：如果文件名不在 b 的集合中
            if file not in files_in_b:
                dst_path = os.path.join(dst_folder, file)
                shutil.copy2(src_path, dst_path) # copy2 会保留文件创建时间等元数据
                print(f"复制: {file}")
                count += 1
            else:
                # 如果你想看哪些文件重复了，取消下面这行的注释
                # print(f"跳过 (已存在): {file}")
                pass

    print("-" * 30)
    print(f"处理完成。共将 {count} 个文件从 {src_folder} 复制到了 {dst_folder}。")


def find_name_duplicates(target_folder):
    # 字典结构： { '文件名': ['路径1', '路径2', ...] }
    name_map = defaultdict(list)
    
    print(f"正在扫描文件夹: {target_folder} ...")
    
    for root, dirs, files in os.walk(target_folder):
        for filename in files:
            full_path = os.path.join(root, filename)
            name_map[filename].append(full_path)

    # 打印结果
    duplicate_count = 0
    print("\n" + "="*40)
    print("发现以下同名文件：")
    print("="*40)
    
    for filename, paths in name_map.items():
        if len(paths) > 1:
            duplicate_count += 1
            print(f"\n[文件名: {filename}]")
            for p in paths:
                print(f"  - {p}")

    if duplicate_count == 0:
        print("没有发现同名文件。")

def copy_b_to_c_based_on_a(folder_a, folder_b, folder_c):
    # 1. 确保目标文件夹 C 存在
    if not os.path.exists(folder_c):
        os.makedirs(folder_c)
        print(f"✅ 已创建目标文件夹: {folder_c}")

    # 2. 获取 A 文件夹中的所有文件名（作为“白名单”）
    # 我们只取文件名，忽略后缀差异（如果你需要精确匹配，去掉 .stem）
    files_in_a = set()
    for file in os.listdir(folder_a):
        if os.path.isfile(os.path.join(folder_a, file)):
            # 这里建议使用完整文件名（含后缀）
            files_in_a.add(file)
    
    print(f"📋 文件夹 A 中共有 {len(files_in_a)} 个参考文件。")

    # 3. 递归遍历 B 文件夹，寻找匹配的文件
    count = 0
    print(f"🔍 正在从 {folder_b} 中检索并拷贝...")
    
    for root, dirs, files in os.walk(folder_b):
        for file in files:
            if file in files_in_a:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(folder_c, file)
                
                # 如果 C 中已经有了同名文件，防止覆盖（可选）
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    count += 1
                    # print(f"已拷贝: {file}")
                else:
                    # print(f"⚠️ 跳过已存在的文件: {file}")
                    pass

    print("-" * 30)
    print(f"🎉 处理完成！")
    print(f"从 B 拷贝了 {count} 个匹配 A 的文件到 C。")


def copy_by_stem_match(folder_a, folder_b, folder_c):
    # 1. 确保目标文件夹 C 存在
    if not os.path.exists(folder_c):
        os.makedirs(folder_c)
        print(f"✅ 已创建目标文件夹: {folder_c}")

    # 2. 获取 A 文件夹中所有文件的“主文件名”
    # 使用 set 存储以提高查询效率。例如: "image_01.txt" -> "image_01"
    stems_in_a = set()
    for file in os.listdir(folder_a):
        if os.path.isfile(os.path.join(folder_a, file)):
            stems_in_a.add(Path(file).stem)
    
    print(f"📋 文件夹 A 中共有 {len(stems_in_a)} 个唯一主文件名。")

    # 3. 递归遍历 B 文件夹
    count = 0
    print(f"🔍 正在从 {folder_b} 中检索匹配的文件并拷贝到 {folder_c}...")
    
    for root, dirs, files in os.walk(folder_b):
        for file in files:
            # 获取 B 中当前文件的主文件名
            stem_b = Path(file).stem
            
            # 如果主文件名在 A 的名单中
            if stem_b in stems_in_a:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(folder_c, file)
                
                # 执行拷贝
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    count += 1
                else:
                    # 如果 C 里已经有同名同后缀文件，则跳过
                    pass

    print("-" * 30)
    print(f"🎉 处理完成！")
    print(f"成功匹配并拷贝了 {count} 个文件。")


def delete_by_stem_match(folder_a, folder_b, dry_run=False):
    """
    folder_a: 参考文件夹
    folder_b: 执行删除的文件夹
    dry_run: 预览模式。True 时只打印不删除，False 时正式删除。
    """
    # 1. 获取 A 文件夹中所有文件的“主文件名”
    stems_in_a = set()
    if not os.path.exists(folder_a):
        print(f"❌ 错误：参考文件夹 A 不存在: {folder_a}")
        return

    for file in os.listdir(folder_a):
        if os.path.isfile(os.path.join(folder_a, file)):
            stems_in_a.add(Path(file).stem)
    
    print(f"📋 文件夹 A 中共有 {len(stems_in_a)} 个参考主文件名。")

    # 2. 递归遍历 B 文件夹并比对
    count = 0
    print(f"🔍 正在检查文件夹 B: {folder_b} ...")
    if dry_run:
        print("⚠️ 当前处于 [预览模式]，不会实际删除文件。")

    for root, dirs, files in os.walk(folder_b):
        for file in files:
            stem_b = Path(file).stem
            
            # 如果 B 中的主文件名在 A 的名单里
            if stem_b in stems_in_a:
                file_to_delete = os.path.join(root, file)
                
                if dry_run:
                    print(f"[待删除] {file_to_delete}")
                else:
                    try:
                        os.remove(file_to_delete)
                        print(f"🗑️ 已删除: {file_to_delete}")
                    except Exception as e:
                        print(f"❌ 删除失败 {file}: {e}")
                
                count += 1

    print("-" * 30)
    if dry_run:
        print(f"🔍 预览完成：共有 {count} 个文件符合删除条件。")
        print("💡 如果确认无误，请将脚本底部的 dry_run=True 改为 dry_run=False 正式执行。")
    else:
        print(f"✅ 处理完成！共从 B 中删除了 {count} 个匹配 A 的文件。")


def _copy_worker(src_file: Path, dst_dir: Path):
    """
    单个文件拷贝任务 (供多线程调用)
    """
    try:
        dst_file = dst_dir / src_file.name
        shutil.copy2(src_file, dst_file)
    except Exception as e:
        # 多线程环境下建议用 tqdm.write 打印错误，防止打断进度条
        tqdm.write(f"拷贝失败: {src_file.name} - {e}")

def copy_every_n_files(
    source_dir, 
    target_dir, 
    step, 
    extensions=None,
    num_workers=1
):
    """
    每 n 个文件选 1 个进行复制 (支持多线程)
    
    Args:
        source_dir: 源文件夹
        target_dir: 目标文件夹
        step: 步长 (每 n 取 1)
        extensions: 筛选后缀
        num_workers: 线程数。1 为单线程，>1 为多线程。
    """
    src_path = Path(source_dir)
    dst_path = Path(target_dir)

    if not src_path.exists():
        print(f"错误: 源目录不存在 {src_path}")
        return

    dst_path.mkdir(parents=True, exist_ok=True)

    print(f"正在扫描并筛选: {src_path} ...")

    # 1. 扫描与筛选 (必须在主线程串行完成，保证顺序)
    all_files = sorted([
        p for p in src_path.iterdir() 
        if p.is_file()
    ])

    if extensions:
        exts = set(e.lower() for e in extensions)
        all_files = [p for p in all_files if p.suffix.lower() in exts]

    if not all_files:
        print("未找到符合条件的文件。")
        return

    # 2. 切片抽取
    selected_files = all_files[::step]

    print(f"总文件数: {len(all_files)}")
    print(f"抽取策略: 每 {step} 取 1")
    print(f"待拷贝数: {len(selected_files)}")
    print(f"执行模式: {'多线程 (' + str(num_workers) + ' workers)' if num_workers > 1 else '单线程'}")

    # 3. 准备 worker 函数 (固定目标目录参数)
    worker_func = partial(_copy_worker, dst_dir=dst_path)

    # 4. 执行拷贝
    if num_workers > 1:
        # --- 多线程模式 ---
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # executor.map 负责并发，tqdm 负责显示进度
            # list(...) 强制立即执行生成器
            list(tqdm(
                executor.map(worker_func, selected_files), 
                total=len(selected_files), 
                desc="Copying (Multi)"
            ))
    else:
        # --- 单线程模式 ---
        for src_file in tqdm(selected_files, desc="Copying (Single)"):
            worker_func(src_file)

    print("\n完成！")

if __name__ == '__main__':
    # 配置路径 (根据你的实际情况修改)
    folder_a = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12\images_crop\re_high_levels\5_Dark'  # 源文件夹
    folder_b = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12\check'  # 对比文件夹（多级）
    folder_c = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data\dataset\thermal_selected_4_p12\check_remain'  # 输出文件夹

    # check_and_copy(folder_a, folder_b, folder_c)
    # find_name_duplicates(folder_b)

    # copy_by_stem_match(
    #     r'E:\data\1123_thermal\thermal data\datasets\moisture\det\images_vis', 
    #     r'e:\data\1123_thermal\thermal data\datasets\moisture\det\images', 
    #     r'e:\data\1123_thermal\thermal data\datasets\moisture\det\selected\images')

    # copy_by_stem_match(
    #     r'E:\data\1123_thermal\thermal data\datasets\moisture\det\images_vis', 
    #     r'e:\data\1123_thermal\thermal data\datasets\moisture\det\labels', 
    #     r'e:\data\1123_thermal\thermal data\datasets\moisture\det\selected\labels')

    # delete_by_stem_match(
    #     r'e:\data\1123_thermal\thermal data\datasets\moisture\det\selected\remove', 
    #     r'e:\data\1123_thermal\thermal data\datasets\moisture\det\selected\images')