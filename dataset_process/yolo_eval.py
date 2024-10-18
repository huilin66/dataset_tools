import os
from tqdm import tqdm
import numpy as np

# region tools

def box_iou(boxesA, boxesB, eps=1e-7):
    x1_A, y1_A, x2_A, y2_A = boxesA[:, 0:1], boxesA[:, 1:2], boxesA[:, 2:3], boxesA[:, 3:4]
    x1_B, y1_B, x2_B, y2_B = boxesB[:, 0], boxesB[:, 1], boxesB[:, 2], boxesB[:, 3]

    # 计算交集区域的坐标
    x1_inter = np.maximum(x1_A, x1_B)  # n*m
    y1_inter = np.maximum(y1_A, y1_B)  # n*m
    x2_inter = np.minimum(x2_A, x2_B)  # n*m
    y2_inter = np.minimum(y2_A, y2_B)  # n*m

    # 计算交集区域的宽度和高度
    inter_width = np.maximum(0, x2_inter - x1_inter)  # n*m
    inter_height = np.maximum(0, y2_inter - y1_inter) # n*m

    # 计算交集面积
    intersection_area = inter_width * inter_height  # n*m

    # 计算每个框的面积
    area_A = (x2_A - x1_A) * (y2_A - y1_A)  # n*1
    area_B = (x2_B - x1_B) * (y2_B - y1_B)  # 1*m

    # 计算并集面积
    union_area = area_A + area_B - intersection_area  # n*m

    # 计算 IoU
    iou_matrix = intersection_area / (union_area + eps)  # n*m
    return iou_matrix

def xyhw_to_xyxy(yolo_labels, img_width, img_height):
    """
    将 YOLO 格式转换为 `class, x1, y1, x2, y2` 格式
    yolo_labels: np.array, YOLO 格式的标签 (n*5) -> [class, x_center, y_center, width, height]
    img_width: 图像的宽度
    img_height: 图像的高度
    返回: 转换后的 `class, x1, y1, x2, y2` 格式的标注
    """
    # 创建一个存储转换后的标注的空数组
    xyxy_labels = np.zeros_like(yolo_labels)

    # class 保持不变
    xyxy_labels[:, 0] = yolo_labels[:, 0]  # 类别

    # 将 YOLO 中的 x_center, y_center, width, height 转换为 x1, y1, x2, y2
    xyxy_labels[:, 1] = (yolo_labels[:, 1] - yolo_labels[:, 3] / 2) * img_width  # x1
    xyxy_labels[:, 2] = (yolo_labels[:, 2] - yolo_labels[:, 4] / 2) * img_height  # y1
    xyxy_labels[:, 3] = (yolo_labels[:, 1] + yolo_labels[:, 3] / 2) * img_width  # x2
    xyxy_labels[:, 4] = (yolo_labels[:, 2] + yolo_labels[:, 4] / 2) * img_height  # y2

    return xyxy_labels

def xyhw_to_xyxypc(yolo_labels, img_width, img_height, conf):
    """
    将 YOLO 格式转换为 `class, x1, y1, x2, y2` 格式
    yolo_labels: np.array, YOLO 格式的标签 (n*5) -> [class, x_center, y_center, width, height]
    img_width: 图像的宽度
    img_height: 图像的高度
    返回: 转换后的 `class, x1, y1, x2, y2` 格式的标注
    """
    if yolo_labels.shape[1] == 0:
        return yolo_labels
    # 创建一个存储转换后的标注的空数组
    xyxy_labels = np.ones((yolo_labels.shape[0], 6))

    # 将 YOLO 中的 x_center, y_center, width, height 转换为 x1, y1, x2, y2
    xyxy_labels[:, 0] = (yolo_labels[:, 1] - yolo_labels[:, 3] / 2) * img_width  # x1
    xyxy_labels[:, 1] = (yolo_labels[:, 2] - yolo_labels[:, 4] / 2) * img_height  # y1
    xyxy_labels[:, 2] = (yolo_labels[:, 1] + yolo_labels[:, 3] / 2) * img_width  # x2
    xyxy_labels[:, 3] = (yolo_labels[:, 2] + yolo_labels[:, 4] / 2) * img_height  # y2
    # class 保持不变
    xyxy_labels[:, 4] = yolo_labels[:, 5]  # confidence
    xyxy_labels[:, 5] = yolo_labels[:, 0]  # 类别
    xyxy_labels = xyxy_labels[xyxy_labels[:, 4]>=conf]
    return xyxy_labels

# endregion

class MAP:
    def __init__(self):
        '''
        计算mAP: mAP@0.5; mAP @0.5:0.95; mAP @0.75
        '''
        self.iouv = np.linspace(0.5, 0.95, 10)  # 不同的IoU置信度 @0.5:0.95
        self.niou = len(self.iouv)  # IoU置信度数量
        self.stats = []  # 存储预测结果


    def process_batch(self, detections, labels):
        '''
        预测结果匹配(TP/FP统计)
        :param detections:(array[N,6]) x1,y1,x1,y1,conf,class (原图绝对坐标)
        :param labels:(array[M,5]) class,x1,y1,x2,y2 (原图绝对坐标)
        '''
        # 每一个预测结果在不同IoU下的预测结果匹配
        correct = np.zeros((detections.shape[0], self.niou)).astype(bool)
        if detections is None:
            self.stats.append((correct, *np.zeros((2, 0)), labels[:, 0]))
        else:
            # 计算标签与所有预测结果之间的IoU
            iou = box_iou(labels[:, 1:], detections[:, :4])
            # 计算每一个预测结果可能对应的实际标签
            correct_class = labels[:, 0:1] == detections[:, 5]
            for i in range(self.niou):  # 在不同IoU置信度下的预测结果匹配结果
                # 根据IoU置信度和类别对应得到预测结果与实际标签的对应关系
                x = np.where((iou >= self.iouv[i]) & correct_class)
                # 若存在和实际标签相匹配的预测结果
                if x[0].shape[0]:  # x[0]:存在为True的索引(实际结果索引), x[1]当前所有True的索引(预测结果索引)
                    # [label, detect, iou]
                    matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                    if x[0].shape[0] > 1:  # 存在多个与目标对应的预测结果
                        matches = matches[matches[:, 2].argsort()[::-1]]  # 根据IoU从高到低排序 [实际结果索引,预测结果索引,结果IoU]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 每一个预测结果保留一个和实际结果的对应
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 每一个实际结果和一个预测结果对应
                    correct[matches[:, 1].astype(int), i] = True  # 表面当前预测结果在当前IoU下实现了目标的预测
            # 预测结果在不同IoU是否预测正确, 预测置信度, 预测类别, 实际类别
            self.stats.append((correct, detections[:, 4], detections[:, 5], labels[:, 0]))

    def calculate_ap_per_class(self, save_dir='.', names=(), eps=1e-16):
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        # tp:所有预测结果在不同IoU下的预测结果 [n, 10]
        # conf: 所有预测结果的置信度
        # pred_cls: 所有预测结果得到的类别
        # target_cls: 所有图片上的实际类别
        tp, conf, pred_cls, target_cls = stats[0], stats[1], stats[2], stats[3]
        # 根据类别置信度从大到小排序
        i = np.argsort(-conf)  # 根据置信度从大到小排序
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # 得到所有类别及其对应数量(目标类别数)
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes

        # ap: 每一个类别在不同IoU置信度下的AP, p:每一个类别的P曲线(不同类别置信度), r:每一个类别的R(不同类别置信度)
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):  # 对每一个类别进行P,R计算
            i = pred_cls == c
            n_l = nt[ci]  # number of labels 该类别的实际数量(正样本数量)
            n_p = i.sum()  # number of predictions 预测结果数量
            if n_p == 0 or n_l == 0:
                continue

            # cumsum：轴向的累加和, 计算当前类别在不同的类别置信度下的P,R
            fpc = (1 - tp[i]).cumsum(0)  # FP累加和(预测为负样本且实际为负样本)
            tpc = tp[i].cumsum(0)  # TP累加和(预测为正样本且实际为正样本)

            # 召回率计算(不同的类别置信度下)
            recall = tpc / (n_l + eps)


            # 精确率计算(不同的类别置信度下)
            precision = tpc / (tpc + fpc)


            # 计算不同类别置信度下的AP(根据P-R曲线计算)
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = self.compute_ap(recall[:, j], precision[:, j])
        # 所有类别的ap值 @0.5:0.95
        return ap

    def compute_ap(self, recall, precision):
        # 增加初始值(P=1.0 R=0.0) 和 末尾值(P=0.0, R=1.0)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope np.maximun.accumulate
        # (返回一个数组,该数组中每个元素都是该位置及之前的元素的最大值)
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # 计算P-R曲线面积
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':  # 插值积分求面积
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO))
            # 积分(求曲线面积)
            ap = np.trapz(np.interp(x, mrec, mpre), x)
        elif method == 'continuous':  # 不插值直接求矩阵面积
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    def calculate_map(self):
        # 计算每个类别在不同IoU下的AP
        ap = self.calculate_ap_per_class()

        # mAP50: 平均所有类别在IoU=0.5时的AP
        mAP50 = ap[:, 0].mean()  # IoU=0.5时对应第0列

        # mAP50-95: 平均所有类别在IoU=0.5到0.95之间的AP
        mAP50_95 = ap.mean(axis=1).mean()  # 每个类别在所有IoU下的AP均值的平均
        print(f'mAP@0.5: {mAP50:.4f}, mAP@0.5:0.95: {mAP50_95:.4f}')
        return mAP50, mAP50_95
    def calculate_map_per_class(self):
        # 计算每个类别在不同IoU下的AP
        ap = self.calculate_ap_per_class()

        # 输出每个类别的mAP50和mAP50-95
        mAP50_per_class = ap[:, 0]  # 每个类别在IoU=0.5下的AP
        mAP50_95_per_class = ap.mean(axis=1)  # 每个类别在IoU=0.5到0.95之间的AP均值

        # 打印每个类别的mAP50和mAP50-95
        for i, class_ap50 in enumerate(mAP50_per_class):
            print(f'Class {i} - mAP@0.5: {class_ap50:.4f}, mAP@0.5:0.95: {mAP50_95_per_class[i]:.4f}')

        return mAP50_per_class, mAP50_95_per_class
    def calculate_maps(self):
        # 计算每个类别在不同IoU下的AP
        ap = self.calculate_ap_per_class()

        # 输出每个类别的mAP50和mAP50-95
        mAP50_per_class = ap[:, 0]  # 每个类别在IoU=0.5下的AP
        mAP50_95_per_class = ap.mean(axis=1)  # 每个类别在IoU=0.5到0.95之间的AP均值

        # mAP50: 平均所有类别在IoU=0.5时的AP
        mAP50 = ap[:, 0].mean()  # IoU=0.5时对应第0列
        # mAP50-95: 平均所有类别在IoU=0.5到0.95之间的AP
        mAP50_95 = ap.mean(axis=1).mean()  # 每个类别在所有IoU下的AP均值的平均

        print(f'Total   - mAP@0.5: {mAP50:.3f}, mAP@0.5:0.95: {mAP50_95:.3f}')
        # 打印每个类别的mAP50和mAP50-95
        for i, class_ap50 in enumerate(mAP50_per_class):
            print(f'Class {i} - mAP@0.5: {class_ap50:.3f}, mAP@0.5:0.95: {mAP50_95_per_class[i]:.3f}')

        return mAP50_per_class, mAP50_95_per_class, mAP50_per_class, mAP50_95_per_class

class MAP_OA(MAP):
    def __init__(self):
        super().__init__()
        self.ap = []

    def process_batch(self, detections, labels, pred_attributes_result, gt_attributes):
        '''
        预测结果匹配(TP/FP统计)
        :param detections:(array[N,6]) x1,y1,x1,y1,conf,class (原图绝对坐标)
        :param labels:(array[M,5]) class,x1,y1,x2,y2 (原图绝对坐标)
        '''


        # 每一个预测结果在不同IoU下的预测结果匹配
        correct = np.zeros((detections.shape[0], self.niou)).astype(bool)
        if detections is None:
            self.stats.append((correct, *np.zeros((2, 0)), labels[:, 0]))
        else:
            # 计算标签与所有预测结果之间的IoU
            iou = box_iou(labels[:, 1:], detections[:, :4])
            # 计算每一个预测结果可能对应的实际标签
            correct_class = labels[:, 0:1] == detections[:, 5]

            pred_attributes_result = np.where(pred_attributes_result > 0.5, 1, 0)
            iou = iou * correct_class  # zero out the wrong classes
            iou50 = iou >= 0.5
            correct_box = correct_class & iou50
            correct_attributes = gt_attributes[:, None, :] == pred_attributes_result[None, :]
            ap = []
            # correct_attributes: n * 300 * 14 --> m *14
            for i in range(correct_attributes.shape[0]):
                ca = correct_attributes[i][correct_box[i]]
                p = np.mean(ca, axis=0) if ca.shape[0] > 0 else np.zeros(gt_attributes.shape[-1])
                ap.append(p[np.newaxis, :])
            ap = np.concatenate(ap, axis=0)
            self.ap.append(ap)

            for i in range(self.niou):  # 在不同IoU置信度下的预测结果匹配结果
                # 根据IoU置信度和类别对应得到预测结果与实际标签的对应关系
                x = np.where((iou >= self.iouv[i]) & correct_class)
                # 若存在和实际标签相匹配的预测结果
                if x[0].shape[0]:  # x[0]:存在为True的索引(实际结果索引), x[1]当前所有True的索引(预测结果索引)
                    # [label, detect, iou]
                    matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                    if x[0].shape[0] > 1:  # 存在多个与目标对应的预测结果
                        matches = matches[matches[:, 2].argsort()[::-1]]  # 根据IoU从高到低排序 [实际结果索引,预测结果索引,结果IoU]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 每一个预测结果保留一个和实际结果的对应
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 每一个实际结果和一个预测结果对应
                    correct[matches[:, 1].astype(int), i] = True  # 表面当前预测结果在当前IoU下实现了目标的预测

            # 预测结果在不同IoU是否预测正确, 预测置信度, 预测类别, 实际类别
            self.stats.append((correct, detections[:, 4], detections[:, 5], labels[:, 0]))

    def calculate_oa(self):

        aps = np.mean(np.concatenate(self.ap, axis=0), axis=0)
        ap_mean = np.mean(aps)
        for i, ap in enumerate(aps):
            print(f"defect {i}: {ap:.3f}")
        print(f"total   : {ap_mean:.3f}")

def read_det_result(file_path, is_prediction=True, conf=0.0):
    """
    读取预测文件或标签文件
    :param file_path: 文件路径
    :param is_prediction: 如果是预测文件，则读取格式为 [x1, y1, x2, y2, confidence, class_id]，否则读取标签
    :return: numpy array 格式的读取数据
    """
    data = np.loadtxt(file_path)
    if len(data.shape) != 2:
        data = np.expand_dims(data, axis=0)
    if is_prediction:
        data = xyhw_to_xyxypc(data, 640, 480, conf)
    else:
        data = xyhw_to_xyxy(data, 640, 480)
    return data

def read_mdet_result(file_path, is_prediction=True, conf=0.0, with_prob=False):
    """
    读取预测文件或标签文件
    :param file_path: 文件路径
    :param is_prediction: 如果是预测文件，则读取格式为 [x1, y1, x2, y2, confidence, class_id]，否则读取标签
    :return: numpy array 格式的读取数据
    """
    data = np.loadtxt(file_path)
    if len(data.shape) != 2:
        data = np.expand_dims(data, axis=0)
    if is_prediction:
        if with_prob:
            data_det = np.hstack((data[:, :1], data[:, -5:]))
            data_att = data[:, 2:-5]
        else:
            ones_column = np.ones((data.shape[0], 1))
            data_det = np.hstack((data[:, :1], data[:, -4:], ones_column))
            data_att = data[:, 2:-4]

        data_det = xyhw_to_xyxypc(data_det, 640, 480, conf)
        data_att = data_att[data[:, -1]>=conf]
    else:
        data_det = np.hstack((data[:, :1], data[:, -4:]))
        data_det = xyhw_to_xyxy(data_det if len(data.shape)==2 else np.expand_dims(data, axis=0), 640, 480)
        data_att = data[:, 2:-4]
    return data_det, data_att

def yolo_result_check(predict_dir, label_dir):
    file_list = os.listdir(label_dir)
    for filename in tqdm(file_list):
        predict_path = os.path.join(predict_dir, filename)
        if not os.path.exists(predict_path):
            with open(predict_path, 'w') as file:
                pass  # 不写入任何内容，文件将保持为空

def yolo_det_eval(predictions_dir, ground_truth_dir):
    yolo_result_check(predictions_dir, ground_truth_dir)

    file_list = os.listdir(predictions_dir)  # 假设文件名相同且按顺序排列

    mAP_calculator1 = MAP()
    for file_name in tqdm(file_list):
        pred_path = os.path.join(predictions_dir, file_name)
        label_path = os.path.join(ground_truth_dir, file_name)

        detections = read_det_result(pred_path, is_prediction=True)
        labels = read_det_result(label_path, is_prediction=False)

        mAP_calculator1.process_batch(detections, labels)

    mAP_calculator1.calculate_maps()

def yolo_mdet_eval(predictions_dir, ground_truth_dir, conf=0.0, with_prob=False):
    yolo_result_check(predictions_dir, ground_truth_dir)

    file_list = os.listdir(predictions_dir)
    mAP_calculator1 = MAP_OA()
    for file_name in tqdm(file_list):
        pred_path = os.path.join(predictions_dir, file_name)
        label_path = os.path.join(ground_truth_dir, file_name)

        detections, detections_att = read_mdet_result(pred_path, is_prediction=True, conf=conf, with_prob=with_prob)
        if detections.shape[1] == 0:
            continue
        labels, labels_att = read_mdet_result(label_path, is_prediction=False)

        mAP_calculator1.process_batch(detections, labels, detections_att, labels_att)

    mAP_calculator1.calculate_maps()
    mAP_calculator1.calculate_oa()

if __name__ == '__main__':
    pass
    # 主程序
    # predictions_dir = r'E:\repository\ultralytics\runs\detect\val21\labels'
    # ground_truth_dir = r'E:\data\1123_thermal\ExpData\PolyUOutdoor_UAV\labels_val'
    #
    # yolo_det_eval(predictions_dir, ground_truth_dir)
    #
    # predictions_dir = r'E:\repository\ultralytics\runs\mdetect\val103\labels'
    # ground_truth_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels_val'
    #
    # yolo_mdet_eval(predictions_dir, ground_truth_dir, conf=0.5)

    # mdet_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\mayolo_infer'
    # label_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels_val'
    # yolo_mdet_eval(mdet_dir, label_dir, with_prob=False)

    mdet_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c_llava\images_infer_result_mdet'
    label_dir = r'E:\data\0417_signboard\data0806_m\dataset\yolo_rgb_detection5_10_c\labels_val'
    for infer_name in os.listdir(mdet_dir):
        print(infer_name)
        infer_path = os.path.join(mdet_dir, infer_name)
        yolo_mdet_eval(infer_path, label_dir, with_prob=False)


