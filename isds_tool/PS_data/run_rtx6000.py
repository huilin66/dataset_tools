import os.path

from yolo_mask_crop import *
from yolo_tools import copy_ref_xlsx, psdata_add_pipline, data_tf_pipline, copy_ref_dir, random_select_exclude, psdata_add_piplines, data_tf_piplines, ref_split, copy_exclude_xlsx
from data_vis.yolo_sta import yolo_sta
from data_vis.yolo_vis import yolo_mdet_vis
from dataformat_swift.yolo2xanylabeling import yolo_to_xanylabeling_dir
if __name__ == '__main__':
    pass
    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0922'
    # image_dir = os.path.join(dataset_dir, 'images')
    # labels_dir = os.path.join(dataset_dir, 'labels')
    # labels_sta_dir = os.path.join(dataset_dir, 'labels_sta')
    # image_crop_dir = os.path.join(dataset_dir, 'result_analysis', 'images_crop')
    # class_file = os.path.join(dataset_dir, 'class.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute.yaml')
    # yolo_sta(
    #     # img_dir=os.path.join(dataset_dir, "images"),
    #     gt_dir=labels_dir,
    #     result_dir=labels_sta_dir,
    #     class_path=class_file,
    #     attribute_path=attribute_file,
    #     seg=True,
    # )
    #
    # myolo_crop(image_dir, labels_dir, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='attribute', only_defect=True, with_boundary=True,
    #            crop_method='with_background_image_shape')

    # data_tf_piplines(
    #     r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0922',
    #     train_ratio_list=[0.8, 0.75, 0.7, 0.65, 0.6],
    #     selected_suffix_list=['_80p', '_75p', '_70p', '_65p', '_60p'],
    #     copy=True
    # )

    # ref_split(
    #     '/localnvme/data/billboard/fused_data/data3617_mseg_c6_0915/val_80p.txt',
    #     '/localnvme/data/billboard/fused_data/data7436_mseg_c5_0922/images',
    #     add_suffix = '_80p_ref'
    # )
    # ref_split(
    #     '/localnvme/data/billboard/fused_data/data3617_mseg_c6_0915/val_80p.txt',
    #     '/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0922/images',
    #     add_suffix = '_80p_ref'
    # )
    # ref_split(
    #     '/localnvme/data/billboard/fused_data/data3617_mseg_c6_0915/val_80p.txt',
    #     '/localnvme/data/billboard/fused_data/data7436_seg_c5_0922/images',
    #     add_suffix = '_80p_ref'
    # )

    # data_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0917'
    # yolo_sta(
    #     # img_dir=os.path.join(data_dir, "images"),
    #     gt_dir=os.path.join(data_dir, "labels"),
    #     result_dir=os.path.join(data_dir, "labels_sta"),
    #     class_path=os.path.join(data_dir, "class.txt"),
    #     attribute_path=os.path.join(data_dir, "attribute.yaml"),
    #     seg=True,
    # )
    # data_dir = r'/localnvme/data/billboard/fused_data/data4197_mseg_c6_0914'
    # yolo_sta(
    #     # img_dir=os.path.join(data_dir, "images"),
    #     gt_dir=os.path.join(data_dir, "labels"),
    #     result_dir=os.path.join(data_dir, "labels_sta"),
    #     class_path=os.path.join(data_dir, "class.txt"),
    #     attribute_path=os.path.join(data_dir, "attribute.yaml"),
    #     seg=True,
    # )
    # data_dir = r'/localnvme/data/billboard/fused_data/data3617_mseg_c6_0915'
    # yolo_sta(
    #     # img_dir=os.path.join(data_dir, "images"),
    #     gt_dir=os.path.join(data_dir, "labels"),
    #     result_dir=os.path.join(data_dir, "labels_sta"),
    #     class_path=os.path.join(data_dir, "class.txt"),
    #     attribute_path=os.path.join(data_dir, "attribute.yaml"),
    #     seg=True,
    # )
    # root_dir = r'/localnvme/data/billboard/fused_data/data5894_mseg_c5_0822/val'
    # attribute_file = os.path.join(root_dir, 'attribute.yaml')
    # class_file = os.path.join(root_dir, 'class_c5.txt')
    # image_dir = os.path.join(root_dir, 'images')
    # label_dir = os.path.join(root_dir, 'prediction_score')
    # txt_dir = os.path.join(label_dir, 'txt')
    # vis_dir = os.path.join(label_dir, 'img')
    # yolo_mdet_vis(image_dir, txt_dir, vis_dir, class_file, crop_dir=None, seg=True, attribute_file=attribute_file, filter_no=False, att_score_vis=True)

    # data3899_dir = r'/localnvme/data/billboard/fused_data/data3899_mseg_c6_0818'
    # data626_dr = r'/localnvme/data/billboard/bd_data/data626_mseg_c6_check0624'
    #
    # random_select_exclude(data3899_dir, data626_dr, train_ratio=0.75, suffix='_75p')
    # random_select_exclude(data3899_dir, data626_dr, train_ratio=0.7, suffix='_70p')
    # random_select_exclude(data3899_dir, data626_dr, train_ratio=0.65, suffix='_65p')
    # random_select_exclude(data3899_dir, data626_dr, train_ratio=0.6, suffix='_60p')

    # psdata_add_pipline(
    #     r'/localnvme/data/billboard/ps_data/psdata_add2177_0911_mseg_c6',
    #     r'/localnvme/data/billboard/fused_data/data3899_mseg_c6_0818',
    #     r'/localnvme/data/billboard/fused_data/data6010_mseg_c6_0911',
    #     add_train_ratio=1, selected_suffix='',copy=True
    # )
    # psdata_add_pipline(
    #     r'/localnvme/data/billboard/ps_data/psdata_add2177_0911_mseg_c6',
    #     r'/localnvme/data/billboard/fused_data/data3899_mseg_c6_0818',
    #     r'/localnvme/data/billboard/fused_data/data6010_mseg_c6_0911',
    #     add_train_ratio=1, selected_suffix='_80p',copy=False
    # )
    #
    # psdata_add_piplines(
    #     r'/localnvme/data/billboard/ps_data/psdata_add4924_0912_mseg_c6',
    #     r'/localnvme/data/billboard/fused_data/data6010_mseg_c6_0911',
    #     r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0914',
    #     add_train_ratio=1, selected_suffix_list=['_80p', '_75p', '_70p', '_65p', '_60p'], copy=True
    # )

    # data_tf_piplines(
    #     r'/localnvme/data/billboard/fused_data/data4197_mseg_c6_0917',
    #     train_ratio_list=[0.8, 0.75, 0.7, 0.65, 0.6],
    #     selected_suffix_list=['_80p', '_75p', '_70p', '_65p', '_60p'],
    #     copy=True
    # )
    #
    #
    # ref_split(
    #     '/localnvme/data/billboard/fused_data/data3617_mseg_c6_0915/val_80p.txt',
    #     '/localnvme/data/billboard/fused_data/data4197_mseg_c5_0917/images',
    #     add_suffix = '_80p_ref'
    # )
    # ref_split(
    #     '/localnvme/data/billboard/fused_data/data3617_mseg_c6_0915/val_80p.txt',
    #     '/localnvme/data/billboard/fused_data/data4197_mseg_c5_l2_0917/images',
    #     add_suffix = '_80p_ref'
    # )
    #
    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917'
    # dataset_dir = root_dir
    # image_dir = os.path.join(dataset_dir, 'images')
    # labels_dir = os.path.join(dataset_dir, 'result_analysis', 'pred_filter_pre_match_defect')
    # image_crop_dir = os.path.join(dataset_dir, 'images_crop_pre')
    # class_file = os.path.join(dataset_dir, 'class_c5.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute_l2.yaml')
    # myolo_crop(image_dir, labels_dir, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='attribute', only_defect=True, with_boundary=True,
    #            crop_method='with_background_image_shape')

    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917'
    # dataset_dir = root_dir
    # image_dir = os.path.join(dataset_dir, 'images')
    # labels_dir = os.path.join(dataset_dir, 'result_analysis', 'filter_pre_match_defect')
    # image_crop_dir = os.path.join(dataset_dir, 'result_analysis', 'images_crop_pre')
    # class_file = os.path.join(dataset_dir, 'class_c5.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute_l2.yaml')
    # myolo_crop(image_dir, labels_dir, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='attribute', only_defect=True, with_boundary=True,
    #            crop_method='with_background_image_shape')

    #
    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917'
    # dataset_dir = root_dir
    # image_dir = os.path.join(dataset_dir, 'images')
    # labels_dir = os.path.join(dataset_dir, 'result_analysis', 'filter_gt_match_defect')
    # image_crop_dir = os.path.join(dataset_dir, 'result_analysis', 'images_crop_gt')
    # class_file = os.path.join(dataset_dir, 'class_c5.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute_l2.yaml')
    # myolo_crop(image_dir, labels_dir, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='attribute', only_defect=True, with_boundary=True,
    #            crop_method='with_background_image_shape')



    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917'
    # dataset_dir = root_dir
    # image_dir = os.path.join(dataset_dir, 'images')
    # labels_dir = os.path.join(dataset_dir, 'result_analysis', 'filter_pre_match_defect')
    # vis_dir = os.path.join(dataset_dir, 'result_analysis', 'images_vis_pre')
    # class_file = os.path.join(dataset_dir, 'class_c5.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute_l2.yaml')
    # yolo_mdet_vis(
    #     image_dir,
    #     labels_dir,
    #     vis_dir,
    #     class_file,
    #     attribute_file=attribute_file,
    #     seg=True,
    #     )
    #
    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917'
    # dataset_dir = root_dir
    # image_dir = os.path.join(dataset_dir, 'images')
    # labels_dir = os.path.join(dataset_dir, 'result_analysis', 'filter_gt_match_defect')
    # vis_dir = os.path.join(dataset_dir, 'result_analysis', 'images_vis_gt')
    # class_file = os.path.join(dataset_dir, 'class_c5.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute_l2.yaml')
    # yolo_mdet_vis(
    #     image_dir,
    #     labels_dir,
    #     vis_dir,
    #     class_file,
    #     attribute_file=attribute_file,
    #     seg=True,
    #     )

    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917'
    # dataset_dir = root_dir
    # image_dir = os.path.join(dataset_dir, 'images')
    # labels_dir = os.path.join(dataset_dir, 'labels')
    # vis_dir = os.path.join(dataset_dir, 'result_analysis', 'images_vis')
    # class_file = os.path.join(dataset_dir, 'class_c5.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute_l2.yaml')
    # yolo_mdet_vis(
    #     image_dir,
    #     labels_dir,
    #     vis_dir,
    #     class_file,
    #     attribute_file=attribute_file,
    #     seg=True,
    #     )

    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917'
    # dataset_dir = root_dir
    # image_dir = os.path.join(dataset_dir, 'images')
    # image_copy_dir = os.path.join(dataset_dir, 'result_analysis', 'images_gt')
    # labels_dir = os.path.join(dataset_dir, 'result_analysis', 'filter_gt_match_defect')
    # copy_ref(image_dir, image_copy_dir, labels_dir)
    #
    # root_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917'
    # dataset_dir = root_dir
    # image_dir = os.path.join(dataset_dir, 'images')
    # image_copy_dir = os.path.join(dataset_dir, 'result_analysis', 'images_pre')
    # labels_dir = os.path.join(dataset_dir, 'result_analysis', 'filter_pre_match_defect')
    # copy_ref(image_dir, image_copy_dir, labels_dir)

    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0917'
    # image_dir = os.path.join(dataset_dir, 'images')
    # label_dir = os.path.join(dataset_dir, 'labels')
    # select_dir = os.path.join(dataset_dir, 'select')
    # image_select_dir = os.path.join(select_dir, 'images')
    # label_select_dir = os.path.join(select_dir, 'labels')
    # img_vis_gt_select_dir = os.path.join(select_dir, 'img_vis_gt')
    # img_vis_gt_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917/result_analysis/images_vis_gt'
    # ref_path = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917/result_analysis/check0919.xlsx'
    # # copy_ref_xlsx(image_dir, image_select_dir, ref_path, column='file_name')
    # # copy_ref_xlsx(label_dir, label_select_dir, ref_path, column='file_name')
    # copy_ref_xlsx(img_vis_gt_dir, img_vis_gt_select_dir, ref_path, column='file_name')


    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0917'
    # image_dir = os.path.join(dataset_dir, 'images')
    # label_dir = os.path.join(dataset_dir, 'labels')
    # select_dir = os.path.join(dataset_dir, 'select_pre')
    # image_select_dir = os.path.join(select_dir, 'images')
    # label_select_dir = os.path.join(select_dir, 'labels')
    # img_vis_pre_select_dir = os.path.join(select_dir, 'img_vis_pre')
    # img_vis_pre_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917/result_analysis/images_vis_pre'
    # exclude_path = r'/localnvme/data/billboard/fused_data/data7436_mseg_c5_l2_0917/result_analysis/check0920.xlsx'
    # copy_exclude_xlsx(img_vis_pre_dir, img_vis_pre_select_dir, exclude_path, column='file_name')
    # copy_ref_dir(image_dir, image_select_dir, img_vis_pre_select_dir)
    # copy_ref_dir(label_dir, label_select_dir, img_vis_pre_select_dir)


    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0917'
    # select_dir = os.path.join(dataset_dir, 'select_pre')
    # image_select_dir = os.path.join(select_dir, 'images')
    # label_select_dir = os.path.join(select_dir, 'labels')
    # json_dir = os.path.join(select_dir, 'json')
    # class_file = os.path.join(dataset_dir, 'class.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute.yaml')
    # yolo_to_xanylabeling_dir(label_select_dir, image_select_dir, json_dir, class_file, attribute_file)



    # psdata_add_pipline(
    #     r'/localnvme/data/billboard/ps_data/psdata_add284_1002_mseg_c6',
    #     r'/localnvme/data/billboard/fused_data/data7436_mseg_c6_0922',
    #     r'/localnvme/data/billboard/fused_data/data7720_mseg_c6_1002',
    #     add_train_ratio=1, selected_suffix='_80p_ref', copy=True
    # )

    dataset_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c6_1021'
    image_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    labels_sta_dir = os.path.join(dataset_dir, 'labels_sta')
    image_crop_dir = os.path.join(dataset_dir, 'result_analysis', 'images_crop')
    class_file = os.path.join(dataset_dir, 'class.txt')
    attribute_file = os.path.join(dataset_dir, 'attribute.yaml')
    # yolo_sta(
    #     # img_dir=os.path.join(dataset_dir, "images"),
    #     gt_dir=labels_dir,
    #     result_dir=labels_sta_dir,
    #     class_path=class_file,
    #     attribute_path=attribute_file,
    #     seg=True,
    # )

    data_tf_pipline(dataset_dir, selected_suffix='_80p_ref')