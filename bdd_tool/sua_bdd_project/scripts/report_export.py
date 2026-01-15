import os
import config
import pandas as pd
from sua_bdd_tool.reporting.engine import BatchDedupEngine
from sua_bdd_tool.utils import load_class_names, load_json

def main():
    class_names = load_class_names(config.CLASS_TXT)
    exif_db = load_json(config.T_VIEWS_EXIF_UPDATE_JSON)
    engine = BatchDedupEngine(
        exif_db=exif_db,
        views_csv_path=config.VIEWS_MAP_CSV,
        views_png_path=config.VIEWS_MAP_OVERVIEW_PNG,
        explanation_json=config.EXPLANATION_JSON,
        floor_map_path=config.HEIGHT_MAP_JSON,
    )
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    views_list = os.listdir(config.VIEWS_T_PATH)[30:31]
    all_dfs = []
    print(f">>> Start to process {len(views_list)} views")
    for view_name in views_list:
        image_dir = os.path.join(config.VIEWS_T_PATH, view_name)
        image_aux_dir = os.path.join(config.VIEWS_RGB_ALIGN_PATH, view_name)
        image_aux_vis_dir = os.path.join(config.VIEWS_RGB_ALIGN_VIS_PATH, view_name)
        yolo_dedup_dir = os.path.join(config.VIEWS_T_YOLO_DEDUP_DIR, view_name)
        
        label_dir = os.path.join(yolo_dedup_dir, config.YOLO_DEDUP_FUSE_NAME)
        proj_info = os.path.join(yolo_dedup_dir, config.YOLO_DEDUP_PROJ_INFO_NAME)

        if not os.path.exists(label_dir):
            print(f"Skipping {view_name}: labels_dedup_fuse not found.")
            continue

        df = engine.process_view_data_aux(
            view_id=view_name,
            img_dir=image_dir,
            img_aux_dir=image_aux_dir,
            label_dir=label_dir,
            project_info_path=proj_info,
            class_path=config.CLASS_TXT,
            max_workers=config.NUM_WORKERS,
        )
        
        if config.VIEW_REPORT:
            engine.export_aggregated_report(
                all_df=df, 
                output_path=config.REPORT_VIEW_PATH.replace('.pdf', f'_{view_name}.pdf'),
                style_id=config.REPORT_VIEW_STYLE_ID,
                logo_left=config.LOGO1,
                logo_right=config.LOGO2,
                target_cls_names=config.TARGET_CLASSES_NAME,
                max_workers=config.NUM_WORKERS,
            )

        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        print(">>> Merging data from all views...")
        final_df = pd.concat(all_dfs, ignore_index=True)
        print(f'get {len(final_df)} rows of data after merging {len(all_dfs)}')

        engine.export_aggregated_report(
            all_df=final_df, 
            output_path=config.REPORT_OVERALL_PATH,
            style_id=config.REPORT_OVERALL_STYLE_ID,
            logo_left=config.LOGO1,
            logo_right=config.LOGO2,
            target_cls_names=config.TARGET_CLASSES_NAME,
            max_workers=config.NUM_WORKERS,
        )
    else:
        print("No data collected from any view.")

if __name__ == "__main__":
    main()