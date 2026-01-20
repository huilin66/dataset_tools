import os
import config
from sua_bdd_tool.utils import load_class_names, load_json
from sua_bdd_tool.structure.floor_manager import FloorManager
from sua_bdd_tool.deduplicator.yolo_dedup import yolo_projecting, yolo_grouping, group_dets_by_image, merge_boxes_by_id, dets_write, analyze_and_vis_conflicts, export_projection_details_json, export_grouping_info
from sua_bdd_tool.data.visulizer import dedup_vis
from sua_bdd_tool.data.visulizer import FacadeVisualizer 

def main():

    # thermal process
    print(">>> 0. load class, floor, exif data...")
    class_names = load_class_names(config.CLASS_RGB_TXT)
    target_classes = [class_names.index(cls) for cls in config.TARGET_CLASSES_NAME_RGB] if config.TARGET_CLASSES_NAME_RGB is not None else list(range(len(class_names)))
    floor_manager = FloorManager(cache_file=config.HEIGHT_MAP_JSON)
    visualizer = FacadeVisualizer(floor_manager)
    exif_db = load_json(config.RGB_VIEWS_EXIF_UPDATE_JSON)
    views_distance = load_json(config.VIEW_DIST_STATISTICS_JSON)
    print(">>> Done!\n")

    views_list = os.listdir(config.VIEWS_RGB_PATH)
    print(f">>> Start to process {len(views_list)} views")
    for view_name in views_list:
        view_id = int(view_name[1:])
        view_id_offset = view_id * config.VIEW_MAX_RECORD_NUM
        view_distance = views_distance[view_name]
        print(f">>> Processing view [{view_name}] with distance {view_distance}...")
        image_dir = os.path.join(config.VIEWS_RGB_PATH, view_name)
        yolo_dir = os.path.join(config.VIEWS_RGB_YOLO_PATH, view_name, "labels")
        
        YOLO_DEDUP_DIR = os.path.join(config.VIEWS_RGB_YOLO_DEDUP_DIR, view_name)
        YOLO_DEDUP_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_NAME)
        YOLO_DEDUP_VIS_ALL_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_VIS_ALL_NAME)
        YOLO_DEDUP_VIS_BY_ID_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_VIS_BY_ID_NAME)

        YOLO_DEDUP_FUSE_ANA_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_FUSE_ANA_NAME)
        YOLO_DEDUP_FUSE_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_FUSE_NAME)
        YOLO_DEDUP_FUSE_VIS_ALL_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_FUSE_VIS_ALL_NAME)
        YOLO_DEDUP_FUSE_VIS_BY_ID_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_FUSE_VIS_BY_ID_NAME)

        YOLO_DEDUP_PROJ_INFO_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_PROJ_INFO_NAME)
        YOLO_DEDUP_PROJ_VIS_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_PROJ_VIS_NAME)
        YOLO_DEDUP_GROUP_INFO_PATH = os.path.join(YOLO_DEDUP_DIR, config.YOLO_DEDUP_GROUP_INFO_NAME)

        print(f">>>  [{view_name}] 1. loading and projecting yolo result...")
        all_dets = yolo_projecting(image_dir, yolo_dir, exif_db, floor_manager, conf_thresh=config.CONF_THRESH_PRIMARY, reid_model_path=config.REID_ONNX_MODEL_PATH, target_classes=target_classes, start_gid=view_id_offset)
        if all_dets is None or len(all_dets) == 0:
            continue
        print(">>> Done!\n")
 
        print(f">>>  [{view_name}] 2. assign id to yolo result...")
        all_dets_with_id = yolo_grouping(all_dets, config.IOU_THRESH, config.IOS_THRESH, config.REID_DEUP_THRESH_RGB, config.SPATIAL_LIMIT_THRESHOLD, id_offset=view_id_offset)
        print(">>> Done!\n")

        print(f">>>  [{view_name}] 3. grouping yolo result by image...")
        dets_by_img = group_dets_by_image(all_dets_with_id)
        dets_write(dets_by_img, YOLO_DEDUP_PATH)
        dedup_vis(dets_by_img, image_dir, YOLO_DEDUP_VIS_ALL_PATH, YOLO_DEDUP_VIS_BY_ID_PATH, vis=config.DEDUP_VIS, num_workers=config.NUM_WORKERS)
        print(">>> Done!\n")

        print(f">>>  [{view_name}] 4. merge boxes by id...")
        analyze_and_vis_conflicts(dets_by_img, image_dir, YOLO_DEDUP_FUSE_ANA_PATH, class_names=class_names, vis_font_size=config.VIS_FONT_SIZE, vis=config.DEDUP_VIS)
        dets_by_img_fuse = merge_boxes_by_id(dets_by_img, conf_thresh=config.CONF_THRESH_FINAL)
        dets_write(dets_by_img_fuse, YOLO_DEDUP_FUSE_PATH)
        dedup_vis(dets_by_img_fuse, image_dir, YOLO_DEDUP_FUSE_VIS_ALL_PATH, YOLO_DEDUP_FUSE_VIS_BY_ID_PATH, vis=config.DEDUP_VIS, num_workers=config.NUM_WORKERS)
        print(">>> Done!\n")

        print(f">>>  [{view_name}] 5. export projection, grouping info...")
        export_projection_details_json(all_dets_with_id, YOLO_DEDUP_PROJ_INFO_PATH)
        visualizer.load_and_plot(YOLO_DEDUP_PROJ_INFO_PATH, YOLO_DEDUP_PROJ_VIS_PATH, view_name)
        export_grouping_info(all_dets_with_id, YOLO_DEDUP_GROUP_INFO_PATH)
        print(">>> Done!\n")

        print(">>> View Done!\n\n")

if __name__ == '__main__':
    main()