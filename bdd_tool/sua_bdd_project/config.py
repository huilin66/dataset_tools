import os

ROOT_DIR = r'\\158.132.186.40\isds\huilin\bdd\collected_data\HMT_data'

RAW_DATA_PATH = os.path.join(ROOT_DIR, "collected data")

DOCUMENT_PATH = os.path.join(ROOT_DIR, "docs")
RGBT_INDEX_FILE = os.path.join(DOCUMENT_PATH, "rgbt_index.json")
VIEWS_MAP_XLSX = os.path.join(DOCUMENT_PATH, "views_map.xlsx")
VIEWS_MAP_CSV = os.path.join(DOCUMENT_PATH, "views_map.csv")
HEIGHT_MAP_JSON = os.path.join(DOCUMENT_PATH, "height_map.json")
RGB_VIEWS_EXIF_JSON = os.path.join(DOCUMENT_PATH, "exif_visible_views.json")
RGB_VIEWS_EXIF_UPDATE_JSON = os.path.join(DOCUMENT_PATH, "exif_visible_views_update.json")
T_VIEWS_EXIF_JSON = os.path.join(DOCUMENT_PATH, "exif_thermal_views.json")
T_VIEWS_EXIF_UPDATE_JSON = os.path.join(DOCUMENT_PATH, "exif_thermal_views_update.json")
VIEW_DIST_STATISTICS_JSON = os.path.join(DOCUMENT_PATH, "view_dist_statistics.json")
CLASS_TXT = os.path.join(DOCUMENT_PATH, "class.txt")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")

SPLIT_DATA_PATH = os.path.join(ROOT_DIR, "data_split")

DATA_RGB_PATH = os.path.join(SPLIT_DATA_PATH, "visible")
VIEWS_RGB_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views")
VIEWS_RGB_YOLO_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views_infer")
VIEWS_RGB_YOLO_DEDUP_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views_infer_dedup")
VIEWS_RGB_YOLO_DEDUP_VIS_ALL_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views_infer_dedup_all")
VIEWS_RGB_YOLO_DEDUP_VIS_BY_ID_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views_infer_dedup_by_id")
VIEWS_RGB_YOLO_DEDUP_FUSE_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views_infer_dedup_fuse")
VIEWS_RGB_YOLO_DEDUP_FUSE_VIS_ALL_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views_infer_dedup_fuse_all")
VIEWS_RGB_YOLO_DEDUP_FUSE_VIS_BY_ID_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views_infer_dedup_fuse_by_id")

VIEWS_RGB_YOLO_DEDUP_PROJ_INFO_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views_infer_dedup_project_info.json")
VIEWS_RGB_YOLO_DEDUP_GROUP_INFO_PATH = os.path.join(SPLIT_DATA_PATH, "visible_views_infer_dedup_group_info.json")

DATA_T_PATH = os.path.join(SPLIT_DATA_PATH, "thermal")
VIEWS_T_PATH = os.path.join(SPLIT_DATA_PATH, "thermal_views")
VIEWS_T_YOLO_PATH = os.path.join(SPLIT_DATA_PATH, "thermal_views_infer")
VIEWS_T_YOLO_DEDUP_DIR = os.path.join(SPLIT_DATA_PATH, "thermal_views_infer_dedup")

YOLO_DEDUP_NAME = "labels_dedup"
YOLO_DEDUP_VIS_ALL_NAME = "labels_dedup_vis_all"
YOLO_DEDUP_VIS_BY_ID_NAME = "labels_dedup_vis_by_id"

YOLO_DEDUP_FUSE_ANA_NAME  = "labels_dedup_fuse_ana"
YOLO_DEDUP_FUSE_NAME = "labels_dedup_fuse"
YOLO_DEDUP_FUSE_VIS_ALL_NAME  = "labels_dedup_fuse_vis_all"
YOLO_DEDUP_FUSE_VIS_BY_ID_NAME  = "labels_dedup_fuse_vis_by_id"

YOLO_DEDUP_PROJ_INFO_NAME = "project_info.json"
YOLO_DEDUP_GROUP_INFO_NAME = "group_info.json"


REPORT_OVERALL_PATH = os.path.join(REPORT_DIR, "report_overall.pdf")
REPORT_PATH = os.path.join(REPORT_DIR, "report_view.pdf")

ANNO_DATA_PATH = os.path.join(ROOT_DIR, "data_anno")
RGB_SELECT_STEP = 3
ANNO_DATA_SELECT_RGB = os.path.join(ANNO_DATA_PATH, f"visible_selected_{RGB_SELECT_STEP}")
ANNO_DATA_FILTER_RGB = ANNO_DATA_SELECT_RGB+'_filter'
T_SELECT_STEM = 4
ANNO_DATA_SELECT_T = os.path.join(ANNO_DATA_PATH, f"thermal_selected_{T_SELECT_STEM}")
ANNO_DATA_FILTER_T = ANNO_DATA_SELECT_T+'_filter'

NUM_WORKERS = 8

VIEWS_SHOT_ALL = True
VIEWS_SHOT_EACH = False

STANDARD_DJI_FORMAT = True

FLOOR_PARAM = {
    'base_height':22500,
    'final height':123800,
    'normal floor height':3150,
    'podium heights': [6000, 5000, 4500, 5500],
    'top heights': [6650],
    'podium names': ['LG', 'G', '1', '2'],
    'top names': ['ROOF'],
    'normal height numbers': 23,
    'normal height number list': [3, 25],
    'special heights': {
        '4': 3450,
        '11': 3450,
        '18': 3450,
        '23': 3450,
    }
}

COLOR_PALETTE = [
    (255, 50, 50),    # 0: Red
    (50, 255, 50),    # 1: Green
    (50, 50, 255),    # 2: Blue
    (255, 255, 50),   # 3: Yellow
    (50, 255, 255),   # 4: Cyan
    (255, 50, 255),   # 5: Magenta
    (255, 128, 0),    # 6: Orange
    (128, 0, 255),    # 7: Purple
    (0, 128, 128),    # 8: Teal
    (128, 128, 0)     # 9: Olive
]

TARGET_CLASSES_NAME = ['high', 'medium', 'low', 'leakage']

IOU_THRESH = 0.5
HEIGHT_THRESH_M = 0.3
X_THRESH_M = 1.5

DEDUP_VIS = True

VIS_FONT_SIZE=30

DISPLAY_LEVELS = ['Minor', 'Moderate', 'Major'] 

REPORT_STYLE_ID=4

LEVELS_THRESHOLD = {
    'pix': [100, 500],
    'ratio': [0.1, 0.5],
    'cm': [5, 20],
    'mm': [50, 200],
    }


DRONE_PARAMS = {
    # === 默认兜底配置 ===
    'default': {
        'sensor_width_mm': 9.6, 
        'focal_length_mm': 6.7
    },

    # M4T: 专为巡检/安防设计，主摄为 1/1.3 CMOS
    'M4T_Wide': {
        'sensor_width_mm': 9.6,  # 1/1.3 inch (注意：之前估算的 6.4mm 是错误的)
        'focal_length_mm': 6.7   # 基于等效24mm换算 (9.6 * 24 / 36 ≈ 6.4, 实际上通常在 6.7mm 左右)
    },
    
    # M4T 热成像: 640x512, 12um, 等效焦距 53mm
    'M4T_Thermal': {
        'sensor_width_mm': 7.68, # 640 pixels * 12 um
        'focal_length_mm': 12.0  # 物理焦距 (根据 DFOV 45度反推约为 11.9~12mm)
    },
}