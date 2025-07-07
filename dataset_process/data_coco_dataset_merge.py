# region coco datasets
ROOT_PATH = r'E:\data\2024_defect\2024_defect_det'

error_set = {
    'ConcreteCracksDetection':['data repeat'], # with hole category
    'Dam_data.v14i.coco': ['hole category mixed with peeling and spalling'], # with hole category, mixed with peeling and spalling
    'defectdetection': ['low resolution', 'Contamination is confusing'],
    'Defects.v7-last-one.coco': ['cropped data'],
    'detr_crack_dataset.v1i.coco': ['too much crack'],
    'dsa.v1i.coco': ['blister mixed with water seepage'],
    'new dataset.v3i.coco': ['too fragment'],
    'tile.v6i.coco': ['too fragment'],
    'wall.v1i.coco': ['few data'],
    # 'walldefect': ['peeling is  crack'] # useful crack
}

categories_map = {
    '200im': {
        'crack': "crack",
        'spall': "concrete_spalling",
    },
    '400img': {
        'crack': "crack",
        'spall': "concrete_spalling",
    },
    'Building Defect.v3i.coco': {
        'crack': "crack",
        'spall': "concrete_spalling",
    },
}
categories_final = [
    {
        "id": 0,
        "name": "background",
        "supercategory": "none"
    },
    {
        "id": 1,
        "name": "crack",
        "supercategory": "defect"
    },
    {
        "id": 2,
        "name": "concrete_spalling",
        "supercategory": "defect"
    },
    {
        "id": 3,
        "name": "finishes_peeling",
        "supercategory": "defect"
    },
    {
        "id": 4,
        "name": "water_seepage",
        "supercategory": "defect"
    },
    {
        "id": 5,
        "name": "stain",
        "supercategory": "defect"
    },
    {
        "id": 6,
        "name": "vegetation",
        "supercategory": "defect"
    }
]

# endregion

# region coco tools

def json_load(js_path):
    with open(js_path, 'r') as load_f:
        data = json.load(load_f)
    return data

def json_save(js_path, data):
    with open(js_path, 'w') as save_f:
        json.dump(data, save_f)


def dataset_merge_coco(dst_dir, data_prefixs=None, merge_dir='train', img_gap=10000, anno_gap=10000):
    dst_dir = os.path.join(dst_dir, merge_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(os.path.join(dst_dir, merge_dir))
    dst_json = os.path.join(dst_dir, '_annotations.coco.json')

    data_list = os.listdir(dst_dir)
    data_list.remove('00data_fuse')
    data_key_list = [i for i in data_list if i not in b]


    # data_key_list = [
    #     '200im', '400img', 'Building Defect.v3i.coco',
    #     'oldbuildingdamagedetection', 'defectdetection', 'walldefect',
    #     # '200im', 'defectdetection',
    # ]
    # img_list = [
    #     os.path.join(ROOT_PATH, data_key, merge_dir) for data_key in data_key_list
    # ]
    # json_list = [
    #     os.path.join(ROOT_PATH, data_key, merge_dir, '_annotations.coco.json') for data_key in data_key_list
    # ]
    # categories_list = [
    #     {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    #     {0: 0, 1: 1, 2: 6},
    #     {0: 0, 1: 1, 2: 6},
    #     {0: 0, 1: 1, 2: 6},
    #     {0: 0, 1: 1, 2: 6},
    #     {0: 0, 1: 1, 2: 6, 3: 2},
    #     {0: 0, 1: 1, 2: 2, 3: 3},
    #     # {0: 0, 1: 1, 2: 6},
    #     # {0: 0, 1: 1, 2: 6, 3: 2},
    # ]
    if data_prefixs is None or len(data_prefixs) != len(data_key_list):
        data_prefixs = ['data%02d_' % idx for idx in range(len(data_key_list))]

    images_new = []
    annos_new = []
    for idx, train_js in enumerate(json_list):
        data = json_load(train_js)
        categories, images, annotations = data['categories'], data['images'], data['annotations']

        data_prefix = data_prefixs[idx]
        img_records = []
        for img_record in tqdm(images, desc='%s img %s' % (merge_dir, data_prefix)):
            img_record['id'] += img_gap*idx
            src_name = img_record['file_name']
            dst_name = data_prefix + src_name
            img_record['file_name'] = dst_name
            img_records.append(img_record)
            src_path = os.path.join(img_list[idx], src_name)
            dst_path = os.path.join(dst_dir, dst_name)
            shutil.copy(src_path, dst_path)
        anno_records = []
        for anno_record in tqdm(annotations, desc='%s anno %s' % (merge_dir, data_prefix)):
            anno_record['id'] += anno_gap*idx
            anno_record['image_id'] += anno_gap * idx
            anno_record['category_id'] =  categories_list[idx][anno_record['category_id']]
            anno_records.append(anno_record)
        images_new += img_records
        annos_new += anno_records
    data_new = {}
    data_new['images'] = images_new
    data_new['annotations'] = annos_new
    data_new['categories'] = categories_final
    json_save(dst_json, data_new)


def dataset_sta_coco(root_dir, save_path):
    pass

    df = pd.DataFrame(None, columns=['name', 'train_img', 'train_box', 'val_img', 'val_num', 'test_img','test_num', 'cats'])

    dir_list = os.listdir(root_dir)
    dir_list.remove('00data_fuse')
    for data_name in dir_list:
        data_dir = os.path.join(root_dir, data_name)
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'valid')
        test_dir = os.path.join(data_dir, 'test')
        train_data = json_load(os.path.join(train_dir, '_annotations.coco.json'))
        val_data = json_load(os.path.join(val_dir, '_annotations.coco.json'))
        test_data = json_load(os.path.join(test_dir, '_annotations.coco.json'))
        cat_list = train_data['categories']
        cats = []
        for category in cat_list:
            cat_name = category['name']
            cats.append(cat_name)
        train_img = len(train_data['images'])
        train_box = len(train_data['annotations'])
        val_img = len(val_data['images'])
        val_box = len(val_data['annotations'])
        test_img = len(test_data['images'])
        test_box = len(test_data['annotations'])
        record = [data_name, train_img, train_box, val_img, val_box, test_img, test_box, cats]
        df.loc[len(df)] = record
    print(df)
    df.to_csv(save_path)


def dataset_vis_coco(root_dir):
    dir_list = os.listdir(root_dir)
    dir_list.remove('00data_fuse')
    for data_name in dir_list:
        data_dir = os.path.join(root_dir, data_name)
        # val_dir = os.path.join(data_dir, 'valid')
        # val_vis_dir = os.path.join(data_dir, 'val_vis')
        # anno_vis(os.path.join(val_dir, '_annotations.coco.json'), img_dir=val_dir, vis_dir=val_vis_dir)
        train_dir = os.path.join(data_dir, 'train')
        train_vis_dir = os.path.join(data_dir, 'train_vis')
        anno_vis(os.path.join(train_dir, '_annotations.coco.json'), img_dir=train_dir, vis_dir=train_vis_dir)


def get_root_csv_coco(data_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    dir_list = os.listdir(data_root)
    for data_name in dir_list:
        data_dir = os.path.join(data_root, data_name)
        if os.path.isdir(data_dir):
            data_path = os.path.join(data_dir, r'README.dataset.txt')
            if os.path.exists(data_path):
                save_path = os.path.join(save_dir, data_name+'_url.txt')
                shutil.copy(data_path, save_path)
                print('save to',save_path)
            else:
                print(data_name, 'not exist')

# endregion
