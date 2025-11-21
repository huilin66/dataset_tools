import os.path
import sys
from yolo_mask_crop import *
from yolo_tools import copy_ref_xlsx, psdata_add_pipline, data_tf_pipline, copy_ref_dir, random_select_exclude, psdata_add_piplines, data_tf_piplines, ref_split, copy_exclude_xlsx
from data_vis.yolo_sta import yolo_sta
from data_vis.yolo_vis import yolo_mdet_vis
from dataformat_swift.yolo2xanylabeling import yolo_to_xanylabeling_dir
from att_tools import get_single_high, get_all_high

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

    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c6_1023'
    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c6_1022'
    # dataset_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c6_1021_broken_refine'
    # dataset_dir = r'/localnvme/data/added_data/check1021/data29_check1021_mseg_c6_broken_refine'
    # dataset_dir = r'/localnvme/data/added_data/check1022/data_mseg_c6_1023'
    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c6_1023'
    # dataset_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c6_1021_broken_refine_defect'
    # dataset_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c5_l2_1021_broken_refine'
    # dataset_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c6_1021_broken_refine'
    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1023_src'
    # dataset_dir = r'/localnvme/data/added_data/test_data/test_data_mseg_c6_1021_broken_refine'
    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c6_1030_abandonment_refine'
    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c6_1030'
    # dataset_dir = r'/localnvme/data/billboard/all_data/mseg_c6/data7961_mseg_c6_1030'
    # dataset_dir = r'/localnvme/data/billboard/all_data/mseg_c6/data7961_mseg_c6_1104_v9'
    # dataset_dir = r'/localnvme/data/billboard/all_data/mseg_c6/data7961_mseg_c6_1104_v6'
    # dataset_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1106_v12'
    # dataset_dir = r'/localnvme/data/billboard/all_data/mseg_c6/data7961_mseg_c6_1112_v16'
    # dataset_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c6_1015'
    # dataset_dir = r'/localnvme/data/billboard/all_data/mseg_c6/data7961_mseg_c6_1112_v16'
    # dataset_dir = r'/localnvme/data/billboard/all_data/mseg_c6/data80_v16'
    # image_dir = os.path.join(dataset_dir, 'images')
    # labels_dir = os.path.join(dataset_dir, 'labels')
    # # labels_dir = os.path.join(dataset_dir, 'labels_v4')
    # json_dir = os.path.join(dataset_dir, 'jsons')
    # image_vis_dir = os.path.join(dataset_dir, 'label_analysis', 'image_vis')
    # labels_sta_dir = os.path.join(dataset_dir, 'label_analysis','labels_sta')
    # image_crop_dir = os.path.join(dataset_dir, 'label_analysis', 'all_gt')
    # class_file = os.path.join(dataset_dir, 'class.txt')
    # attribute_file = os.path.join(dataset_dir, 'attribute.yaml')
    # ref_txt = os.path.join(dataset_dir, "val_test.txt")
    # yolo_sta(
    #     # img_dir=os.path.join(dataset_dir, "images"),
    #     gt_dir=labels_dir,
    #     result_dir=labels_sta_dir,
    #     class_path=class_file,
    #     attribute_path=attribute_file,
    #     # ref_txt=ref_txt,
    #     seg=True,
    # )

    # data_tf_pipline(dataset_dir, train_ratio=1, split_mseg_c6=True)
    # data_tf_pipline(dataset_dir, selected_suffix='_80p_ref', split_mseg_c6=False)

    # myolo_crop(image_dir, labels_dir, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='attribute', only_defect=True, with_boundary=True,
    #            crop_method='with_background_image_shape')

    # myolo_crop(image_dir, labels_dir, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='all', only_defect=False, with_boundary=True,
    #            crop_method='with_background_box_shape', with_conf=False)

    # myolo_crop(image_dir, labels_dir, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='attribute', only_defect=True, with_boundary=False,
    #            crop_method='with_background_box_shape')

    # yolo_to_xanylabeling_dir(labels_dir, image_dir, json_dir, class_file, attribute_file)

    # yolo_mdet_vis(image_dir, labels_dir, image_vis_dir, class_file, crop_dir=None, seg=True,
    #               attribute_file=attribute_file, filter_no=True, crop_keep_shape=False, seg_crop=False)
    # print(len(os.listdir(image_crop_dir)))

    # base_dir = r'/localnvme/data/billboard/fused_data/data7961_mseg_c5_l2_1029_abandonment_refine'
    # image_dir = os.path.join(base_dir, 'images')
    # label_dir_root = os.path.join(base_dir, 'result_analysis', 'keep', 'pred_no_label_background')
    # label_dir = os.path.join(label_dir_root, 'merge_1105_labels')
    # image_crop_dir = os.path.join(label_dir_root, 'merge_1105_labels_crop')
    # attribute_file = os.path.join(base_dir, 'attribute.yaml')
    # class_file = os.path.join(base_dir, 'class.txt')
    # myolo_crop(image_dir, label_dir, image_crop_dir, class_file,
    #            attribute_file=attribute_file, seg=True, annotation=False,
    #            save_method='attribute', only_defect=True, with_boundary=False,
    #            crop_method='with_background_box_shape')


    # input_label_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1110_v13/labels'
    # att_path = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1110_v13/attribute.yaml'
    # get_all_high(input_label_dir, attributes=att_path)

    # input_label_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1110_v13_single_risk/b/labels'
    # att_path = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1110_v13_single_risk/b/attribute.yaml'
    # get_all_high(input_label_dir, attributes=att_path)

    # from yolo_tools import data_tf_pipline_new
    #
    # datav16_dir = r'/localnvme/data/billboard/all_data/mseg_c6/data7961_mseg_c6_1112_v16'
    # data_tf_pipline_new(datav16_dir, copy_list=['seg_c5'])

    # datav16_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1113_v17'
    # data80_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v15'
    # input_image_dir = os.path.join(datav16_dir, 'images')
    # input_label_dir = os.path.join(datav16_dir, 'labels')
    # input_val_test_path = os.path.join(datav16_dir, 'val_test.txt')
    # data80_image_dir = os.path.join(data80_dir, 'images')
    # data80_label_dir = os.path.join(data80_dir, 'labels')
    # image_crop_dir = os.path.join(data80_dir, 'labels_crop')
    # class_c5_path = r'/localnvme/data/billboard/class_c5.txt'
    # att_path = r'/localnvme/data/billboard/attribute.yaml'
    # data80_class_path = os.path.join(data80_dir, 'class.txt')
    # data80_att_path = os.path.join(data80_dir, 'attribute.yaml')
    # import shutil
    # from yolo_tools import copy_ref_csv
    # copy_ref_csv(input_image_dir, data80_image_dir, input_val_test_path)
    # copy_ref_csv(input_label_dir, data80_label_dir, input_val_test_path)
    # shutil.copy(class_c5_path, data80_class_path)
    # shutil.copy(att_path, data80_att_path)
    #
    #
    # myolo_crop(
    #     data80_image_dir,
    #     data80_label_dir,
    #     image_crop_dir,
    #     data80_class_path,
    #     attribute_file=data80_att_path,
    #     seg=True,
    #     annotation=False,
    #     save_method='attribute',
    #     only_defect=False,
    #     with_boundary=True,
    #     crop_method='with_background_box_shape'
    # )
    # from att_tools import remove_conf, vis_matched_result
    # val_dir = r'/localnvme/project/ultralytics/runs/msegment/val844/labels'
    # base_dir = r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17'
    # image_dir = os.path.join(base_dir, 'images')
    # label_dir = os.path.join(base_dir, 'labels')
    # val_test_dir = os.path.join(base_dir, 'val_test')
    # image_test_dir = os.path.join(val_test_dir, 'images')
    # label_test_dir = os.path.join(val_test_dir, 'labels')
    # result_analysis_dir = os.path.join(base_dir, 'result_analysis')
    # vis_dir = os.path.join(result_analysis_dir, 'vis')
    # class_path = os.path.join(base_dir, 'class.txt')
    # att_path = os.path.join(base_dir, 'attribute.yaml')
    # val_test_path = os.path.join(base_dir, 'val_test.txt')
    # pred_dir = os.path.join(base_dir, 'val844')

    # remove_conf(r'/localnvme/project/ultralytics/runs/msegment/val767/labels',
    #             r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17/val767_no_conf',
    #             conf_threshold=0.4, filter_small=0.05,
    #             )
    # get_all_high(r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17/val767_no_conf', attributes=att_path)
    #
    # remove_conf(r'/localnvme/project/ultralytics/runs/msegment/val767/labels',
    #             r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17/val767_no_conf',
    #             conf_threshold=0.1, filter_small=0.05,
    #             )
    # get_all_high(r'/localnvme/data/billboard/all_data/mseg_c5_l2/data80_v17/val767_no_conf', attributes=att_path)
    # val_dir = r'/localnvme/data/billboard/infer8/labels'
    # get_all_high(val_dir, attributes=att_path, with_conf=True, conf_threshold=0.1)
    # remove_conf(val_dir, pred_dir, conf_threshold=0.1, filter_small=0.05,)
    # get_all_high(pred_dir, attributes=att_path)
    #
    # vis_matched_result(
    #     image_dir,
    #     label_dir,
    #     pred_dir,
    #     vis_dir,
    #     class_path,
    #     att_path,
    #     with_conf=False,
    #     annotation=True,
    #     iou_thr=0.3,
    #     conf_threshold=0.4,
    #     defect_conf_threshold=0.1,
    #     filter_small=0.05,
    #     save_method='attribute',
    #     crop_method='with_background_box_shape',
    # )

    get_all_high(r'/localnvme/data/added_data/test_data1121/images_infer5/labels',
                 attributes=r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1117_v21_single_risk/b/attribute.yaml')
    get_all_high(r'/localnvme/data/added_data/test_data1121/images_infer6/labels',
                 attributes=r'/localnvme/data/billboard/all_data/mseg_c5_l2/data7961_mseg_c5_l2_1117_v21_single_risk/b/attribute.yaml')
    # myolo_crop(
    #     r'/scrinvme/huilin/isds/check_data/synthetic_data_add4_v1',
    #     r'/scrinvme/huilin/isds/check_data/synthetic_data_add4_v1_labels',
    #     r'/scrinvme/huilin/isds/check_data/synthetic_data_add4_v1_labels_crop',
    #     r'/scrinvme/huilin/isds/check_data/class.txt',
    #     attribute_file=r'/scrinvme/huilin/isds/check_data/attribute.yaml',
    #     seg=True,
    #     annotation=False,
    #     save_method='attribute',
    #     only_defect=False,
    #     with_boundary=True,
    #     crop_method='with_background_box_shape'
    # )