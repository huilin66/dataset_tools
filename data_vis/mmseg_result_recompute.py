import re


def extract_miou_per_epoch(log_file):
    miou_per_epoch = []
    class_ious = {}

    # 定义正则表达式模式来提取 IoU 值
    iou_pattern = re.compile(r'\|\s+(defect[123])\s+\|\s+(\d+\.\d+)\s+\|\s+\d+\.\d+\s+\|')


    with open(log_file, 'r') as file:
        for line in file:
            # 提取 defect 类的 IoU
            iou_match = iou_pattern.search(line)
            if iou_match is not None:
                defect_class = iou_match.group(1)
                iou_value = float(iou_match.group(2))
                class_ious[defect_class] = iou_value

            # 一旦找到所有 defect 的 IoU，计算 mIoU
            if len(class_ious) == 3:
                defect1_iou = class_ious.get('defect1', 0)
                defect2_iou = class_ious.get('defect2', 0)
                defect3_iou = class_ious.get('defect3', 0)
                miou = round((defect1_iou + defect2_iou + defect3_iou) / 3, 2)
                miou_per_epoch.append([miou, defect1_iou, defect2_iou, defect3_iou])
                class_ious = {}  # 重置当前 epoch 的 IoU

    return miou_per_epoch


if __name__ == '__main__':
    # 使用函数读取并计算 mIoU
    # log_file = r'E:\repository\mmsegmentation\work_dirs\unet3\20241009_095756\20241009_095756.log'  # 假设你的日志文件名是 training_log.txt
    # log_file = r'E:\repository\mmsegmentation\work_dirs\unet35\20241009_212918\20241009_212918.log'
    # log_file = r'E:\repository\mmsegmentation\work_dirs\upernet_convnext\20241010_105825\20241010_105825.log'
    log_file = r'E:\repository\mmsegmentation\work_dirs\upernet_swin\20241010_135034\20241010_135034.log'
    miou_per_epoch = extract_miou_per_epoch(log_file)

    # 打印每个 epoch 的 mIoU
    for idx, ious in enumerate(miou_per_epoch):
        print(idx, ious)
