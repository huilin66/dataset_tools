import config
from sua_bdd_tool.data.align import batch_align
from sua_bdd_tool.data.dataset_preparer import (
    build_index,
    export_single_modal_data,
    split_data_by_index,
)
from sua_bdd_tool.data.image_meta import build_metadata_json
from sua_bdd_tool.data.lrf_tool import statistics_lrf_data
from sua_bdd_tool.deduplicator.ssim_deduplication import filter_deduplication_ssim
from sua_bdd_tool.structure.floor_manager import FloorManager
from sua_bdd_tool.structure.view_manager import process_views_data
from sua_bdd_tool.utils.file_opt import copy_every_n_files

def main():
    pass

    print(">>> 1. match RGB-T pair data...")
    build_index(config.RAW_DATA_PATH, config.RGBT_INDEX_FILE, standard_DJI_format=config.STANDARD_DJI_FORMAT)
    print(">>> Done!\n")


    print(">>> 2. export all single modal data to single folder...")
    export_single_modal_data(config.RGBT_INDEX_FILE, config.SPLIT_DATA_PATH, num_workers=config.NUM_WORKERS)
    print(">>> Done!\n")
    

    # TODO: get views
    # print(">>> 3. get views...")
    # split_views()


    print(">>> 4. generate view folder...")
    split_data_by_index(config.VIEWS_MAP_XLSX, config.RGBT_INDEX_FILE, config.VIEWS_RGB_PATH, config.VIEWS_T_PATH, num_workers=config.NUM_WORKERS)
    print(">>> Done!\n")


    print(">>> 5. generate views exif information...")
    build_metadata_json(config.VIEWS_RGB_PATH, config.RGB_VIEWS_EXIF_JSON, num_workers=config.NUM_WORKERS)
    build_metadata_json(config.VIEWS_T_PATH, config.T_VIEWS_EXIF_JSON, num_workers=config.NUM_WORKERS)
    print(">>> Done!\n")
    

    print(">>> 6. statistics overvall lrf distance information...")
    statistics_lrf_data(config.RGB_VIEWS_EXIF_JSON, config.RGB_VIEWS_EXIF_UPDATE_JSON, config.VIEW_DIST_STATISTICS_JSON)
    statistics_lrf_data(config.T_VIEWS_EXIF_JSON, config.T_VIEWS_EXIF_UPDATE_JSON)
    print(">>> Done!\n")


    print(">>> 7. generate building views information...")
    process_views_data(config.VIEWS_RGB_PATH, config.DOCUMENT_PATH, config.RGB_VIEWS_EXIF_JSON, view_shot_all=config.VIEWS_SHOT_ALL, view_shot_each=config.VIEWS_SHOT_EACH)
    print(">>> Done!\n")
    

    print(">>> 8. generate building floors information...")
    FloorManager(floor_params=config.FLOOR_PARAM, cache_file=config.HEIGHT_MAP_JSON)
    print(">>> Done!\n")


    print(">>> 9. select data for annotation...")
    copy_every_n_files(config.DATA_RGB_PATH, config.ANNO_DATA_SELECT_RGB, config.RGB_SELECT_STEP, num_workers=config.NUM_WORKERS)
    filter_deduplication_ssim(config.ANNO_DATA_SELECT_RGB, config.ANNO_DATA_FILTER_RGB, num_workers=config.NUM_WORKERS)
    copy_every_n_files(config.DATA_T_PATH, config.ANNO_DATA_SELECT_T, config.T_SELECT_STEM, num_workers=config.NUM_WORKERS)
    filter_deduplication_ssim(config.ANNO_DATA_SELECT_T, config.ANNO_DATA_FILTER_T, num_workers=config.NUM_WORKERS)
    print(">>> Done!\n")


    print(">>> 10. align rgb extent with t extent...")
    batch_align(config.RGBT_INDEX_FILE, config.VIEWS_RGB_PATH, config.VIEWS_T_PATH, config.VIEWS_RGB_ALIGN_PATH, config.VIEWS_RGB_ALIGN_COMPARE_PATH, num_workers=config.NUM_WORKERS)
    print(">>> Done!\n")


if __name__ == "__main__":
    main()