"""查找第i个属性值不为0的全部label txt文件名，并支持拷贝对应图像和label。

数据格式: class_id attribute_len attr1 attr2 ... attrN polygon_points...
attribute_len 表示后面跟了几个属性值(0或1)。
"""

import os
import shutil
import argparse
from pathlib import Path

import yaml
from tqdm import tqdm


def find_attr_not_zero(label_folder, attr_index, attribute_file=None):
    """找出所有label txt中, 第attr_index个属性值不为0的文件。

    Args:
        label_folder: label txt 文件夹路径
        attr_index: 属性索引(0-based), 即第几个属性
        attribute_file: attribute_all.yaml 路径(可选, 用于显示属性名)

    Returns:
        list[tuple[str, list]]: (文件名, 该文件中不为0的检测行信息列表)
    """
    if attribute_file and os.path.exists(attribute_file):
        with open(attribute_file, 'r') as f:
            attr_dict = yaml.load(f, Loader=yaml.BaseLoader)['attributes']
        attr_names = list(attr_dict.keys())
        attr_name = attr_names[attr_index] if attr_index < len(attr_names) else f'attr_{attr_index}'
    else:
        attr_name = f'属性{attr_index}(0-based)'

    label_files = sorted(os.listdir(label_folder))
    label_files = [f for f in label_files if f.endswith('.txt')]

    results = []
    for filename in tqdm(label_files, desc=f'检查 {attr_name} != 0'):
        filepath = os.path.join(label_folder, filename)
        with open(filepath, 'r') as f:
            lines = f.read().strip().splitlines()

        bad_lines = []
        for line_idx, line in enumerate(lines):
            if not line.strip():
                continue
            parts = line.strip().split()
            class_id = int(float(parts[0]))
            attr_len = int(parts[1])
            attrs = [int(float(x)) for x in parts[2:2 + attr_len]]

            if attr_index < len(attrs) and attrs[attr_index] != 0:
                bad_lines.append({
                    'line': line_idx,
                    'class': class_id,
                    'attr_values': attrs,
                })

        if bad_lines:
            results.append((filename, bad_lines))

    return results, attr_name


def copy_results(results, img_folder, label_folder, output_img_dir, output_label_dir, img_ext='.jpg'):
    """将匹配到的文件对应的图像和label拷贝到指定文件夹。

    Args:
        results: find_attr_not_zero 的返回结果 (filename, bad_lines) 列表
        img_folder: 源图像文件夹
        label_folder: 源label文件夹
        output_img_dir: 目标图像文件夹
        output_label_dir: 目标label文件夹
        img_ext: 图像文件后缀, 如 '.jpg', '.png'
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    copied = 0
    for filename, bad_lines in tqdm(results, desc='拷贝文件'):
        stem = Path(filename).stem

        # 拷贝 label
        src_label = os.path.join(label_folder, filename)
        dst_label = os.path.join(output_label_dir, filename)
        shutil.copy2(src_label, dst_label)

        # 拷贝图像 (尝试多种后缀)
        src_img = None
        for ext in [img_ext, '.png', '.jpeg', '.tif', '.tiff']:
            candidate = os.path.join(img_folder, stem + ext)
            if os.path.exists(candidate):
                src_img = candidate
                break
        if src_img is not None:
            dst_img = os.path.join(output_img_dir, os.path.basename(src_img))
            shutil.copy2(src_img, dst_img)
            copied += 1
        else:
            print(f"  警告: 未找到图像 {stem}.*")

    print(f"\n拷贝完成: {copied} 张图像, {len(results)} 个label")


def main():
    parser = argparse.ArgumentParser(description='查找第i个属性值不为0的label文件')
    parser.add_argument('label_folder', help='label txt 文件夹路径')
    parser.add_argument('attr_index', type=int, help='属性索引(0-based)')
    parser.add_argument('--attribute-file', '-a', default=None, help='attribute_all.yaml 路径')
    parser.add_argument('--output', '-o', default=None, help='输出结果到文件')
    parser.add_argument('--img-folder', default=None, help='图像文件夹(启用拷贝时必填)')
    parser.add_argument('--copy-img-dir', default=None, help='拷贝图像到该文件夹')
    parser.add_argument('--copy-label-dir', default=None, help='拷贝label到该文件夹')
    parser.add_argument('--img-ext', default='.jpg', help='图像后缀 (默认 .jpg)')
    args = parser.parse_args()

    results, attr_name = find_attr_not_zero(
        args.label_folder, args.attr_index, args.attribute_file
    )

    print(f"\n{'='*60}")
    print(f"检查属性: {attr_name} (索引 {args.attr_index})")
    print(f"属性值 != 0 的文件数: {len(results)} / {len(os.listdir(args.label_folder))}")
    print(f"{'='*60}\n")

    for filename, bad_lines in results:
        print(f"  {filename}")
        for info in bad_lines:
            print(f"    行{info['line']}: class={info['class']}, attrs={info['attr_values']}")

    if args.output:
        with open(args.output, 'w') as f:
            for filename, _ in results:
                f.write(filename + '\n')
        print(f"\n文件名列表已保存至: {args.output}")

    if args.copy_img_dir and args.copy_label_dir:
        if not args.img_folder:
            print("错误: 拷贝需要指定 --img-folder")
            return
        copy_results(
            results, args.img_folder, args.label_folder,
            args.copy_img_dir, args.copy_label_dir, args.img_ext
        )


if __name__ == '__main__':
    # 默认直接运行使用当前 root_dir 配置
    root_dir = r"\\158.132.186.40\isds\huilin\isds\back up\final_data"
    label_folder = os.path.join(root_dir, "labels_all")
    img_folder = os.path.join(root_dir, "images")
    attribute_file = r"\\158.132.186.40\isds\huilin\isds\back up\demo_data\0612\attribute_all.yaml"
    dst_dir = 'zero_test'
    os.makedirs(dst_dir, exist_ok=True)

    # 修改这里的 attr_index 来选择检查第几个属性 (0-based)
    # 0=deformation, 1=broken, 2=abandonment, 3=corrosion

    for attr_index in range(4):
        if attr_index in [0, 2, 3]:
            continue
        results, attr_name = find_attr_not_zero(label_folder, attr_index, attribute_file)

        print(f"\n{'='*60}")
        print(f"检查属性: {attr_name} (索引 {attr_index})")
        print(f"属性值 != 0 的文件数: {len(results)}")
        print(f"{'='*60}\n")

        for filename, bad_lines in results:
            print(f"  {filename}")
            for info in bad_lines:
                print(f"    行{info['line']}: class={info['class']}, attrs={info['attr_values']}")


        # 保存文件名列表
        output_file = os.path.join(dst_dir, f"files_attr{attr_index}_not_zero.txt")
        with open(output_file, 'w') as f:
            for filename, _ in results:
                f.write(filename.replace('.txt', '') + '\n')
        print(f"\n文件名列表已保存至: {output_file}")
        # 拷贝匹配的图像和label到指定文件夹
        copy_img_dir = os.path.join(dst_dir, f"images_attr{attr_index}_not_zero")
        copy_label_dir = os.path.join(dst_dir, f"labels_attr{attr_index}_not_zero")
        copy_results(results, img_folder, label_folder, copy_img_dir, copy_label_dir)
