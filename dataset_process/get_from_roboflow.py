import os
from roboflow import Roboflow

location = r'E:\data\2024_defect\2024_defect_pure_yolo'
rf = Roboflow(api_key="AZpYfvetSYpLXI5Fb4sp")

data_list = [
    # "102-1-huexz",
# "200-im-tkajb",
# "400-img-gwe21",
# "brick-wall-ib4yx-6asnp",
# "building-defect-5yuza-b2plv",
# "building-defect-on-walls-4k5c2",
# "concrete-cracks-detection-kyzaz-y76gn",
# "crack-bpxku-hcu46",
# "cracks-and-spalling-xyiog",
# "dam_data-agekv",
# "damage_level_detection-w7nim",
# "defects-jkoqd-a3con",
# "detr_crack_dataset-bnrlv",
# "dipl_final-yfvdo",
# "drill-hole-detction-3pko9",
# "dsa-7erpa-cn6os",
# "hole-vjngv-dp2s0",
# "metal-qdtrr-kspev",
# "mold-vms1d-tk8cd",
# "new-dataset-vafsu-ncx0g",
# "old-building-damage-detectionv2-e55o0",
# "spalling-qu4c3-gd43m",
# "stain-4gofo-7lxez",
# "structural-defects-v2-eq59e",
# "tile-jspbo-jhjfh",
# "wall-defects-khcmf",
# "wall-qmf7f-hpgv6",
# "wall-defect-ogum1-3wsxo",
# "crack-annotation-4ae0i-ufeyr", # https://universe.roboflow.com/crack-image/crack-annotation-4ae0i
# "building-defect-l8nxj",        # https://universe.roboflow.com/joe-i4soa/building-defect
# "damage-type-acavn",            # https://universe.roboflow.com/isaac-agyemang/damage-type
# "building-surface-bj2ov",       # https://universe.roboflow.com/hku-sdauc/building-surface
# "building_paper-a1poq",         # https://universe.roboflow.com/harrisburg-university/building_paper
# "crack-images-labeling-duz3e-rodgw", # https://universe.roboflow.com/crack-detector/crack-images-labeling-duz3e
# "dipl_final-yxa9t",             # https://universe.roboflow.com/technical-university-of-crete/dipl_final
# "stvsnonst-ssyz3",              # https://universe.roboflow.com/saitama-university/stvsnonst
# "-xwyzo-8kvij",                 # https://universe.roboflow.com/hongzhe-yue-3a4b0/-xwyzo
#     "ceiling-y9dkm-yxoz0",    # https://universe.roboflow.com/scipublic/ceiling-y9dkm
#     "finderrors-r3xln", # https://universe.roboflow.com/dankook-dfq0f/finderrors
#     "dacon-x4sy3-noctn", # https://universe.roboflow.com/konkuk-unvi/dacon-x4sy3
#     "butler-defect-yevm2", # https://universe.roboflow.com/yolo-3t5gp/butler-defect
#     "stm-qg3pa", # https://universe.roboflow.com/foxlanders/stm
#     "only_labeling_all-images-jm1am", # https://universe.roboflow.com/dankook-dfq0f/only_labeling_all-images
#     "common-defect-2-9jmal", # https://universe.roboflow.com/defect-mjkx9/common-defect-2
#     "bd1-9hgll-94afa", # https://universe.roboflow.com/rmit-6tccc/bd1-9hgll
#     "plastering-detect-ztqn9", # https://universe.roboflow.com/jilin-university-fjjjl/plastering-detect
#     "8.10-train-e3pt6-rfruo", # https://universe.roboflow.com/cctvtaipai/8.10-train-e3pt6
#     "sep-xmrpq-w3x7u", # https://universe.roboflow.com/cctvtaipai/sep-xmrpq
# "maintanance-oyjxd", # https://universe.roboflow.com/oke-project/maintanance
# "cracks-spalls-segmentation-ybu6m", # https://universe.roboflow.com/moin-exnkz/cracks-spalls-segmentation
#     "corrosion-detections-2wfdu", # https://universe.roboflow.com/yolo-learning/corrosion-detections
    "rust-vmwoa-klgu3", # https://universe.roboflow.com/kiot-utdwr/rust-vmwoa
    "corrosion-of-automation-ioqzy", # https://universe.roboflow.com/corrosion-project/corrosion-of-automation
    "corrosion-detection-npdhd-ekx8c", # https://universe.roboflow.com/corrosion-l4cqs/corrosion-detection-npdhd
]


for data_name in data_list:
    save_path = os.path.join(location, data_name)
    os.makedirs(save_path, exist_ok=True)
    project = rf.workspace("defectdetection-qdxvy").project(data_name)
    version = project.version(1)
    dataset = version.download("yolov9", save_path, True)





# project = rf.workspace("defectdetection-qdxvy").project("defect-detection-bbmeq-gtq6d")
# version = project.version(1)
# dataset = version.download("coco-segmentation")
#
# project = rf.workspace("defectdetection-qdxvy").project("urbanroof-q0has")
# version = project.version(1)
# dataset = version.download("folder")


# project = rf.workspace("defectdetection-qdxvy").project("water-damage-g3mus-elnoy")
# version = project.version(1)
# dataset = version.download("coco-segmentation")

