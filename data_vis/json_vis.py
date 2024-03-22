import os, cv2
import json
import argparse
from skimage import io
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings("ignore")

# catid2name1 = {
#     0: 'Concrete-Cracks',
#     1: 'Bullet_imapct',
#     2: 'Explosion_Impact',
#     3: 'normal_crack',
#     4: 'severe_crack',
# }
# catid2name2 = {
#     0: 'Concrete-Cracks',
#     1: 'crack',
#     2: 'spall',
# }
# catid2name3 = {
#     0: 'building-defect',
#     1: 'crack',
#     2: 'mold',
#     3: 'peeling_paint',
#     4: 'stairstep_crack',
#     5: 'water_seepage',
# }
# catid2name = catid2name3


def img_read(img_path):
    img = io.imread(img_path)
    return img

def json_read(json_path):
    data = json.load(open(json_path))
    return data

def build_dict(images, annos):
    image_name2id_dict = {}
    image_name2record_dict = {}
    for record in images:
        image_name2id_dict[record['file_name']] = record['id']
        image_name2record_dict[record['file_name']] = record

    imageid2annoid_dict = {}
    for record in annos:
        if record['image_id'] in list(imageid2annoid_dict.key()):
            imageid2annoid_dict[record['image_id']].append(record['id'])
        else:
            imageid2annoid_dict[record['image_id']] = [record['id']]

    anno_id2idx_dict = {}
    for idx,record in enumerate(annos):
        anno_id2idx_dict[record['id']] = idx
    return image_name2id_dict, imageid2annoid_dict, anno_id2idx_dict, image_name2record_dict

def build_img_id2name_dict(images):
    image_name2id_dict = {}
    for record in images:
        image_name2id_dict[record['id']] = record['file_name']
    return image_name2id_dict

def build_img_name2id_dict(images):
    image_name2id_dict = {}
    for record in images:
        image_name2id_dict[record['file_name']] = record['id']
    return image_name2id_dict

def build_img2anno_dict(annos):
    imageid2annoid_dict = {}
    for record in annos:
        if record['image_id'] in list(imageid2annoid_dict.keys()):
            imageid2annoid_dict[record['image_id']].append(record['id'])
        else:
            imageid2annoid_dict[record['image_id']] = [record['id']]
    return imageid2annoid_dict

def build_anno_id2idx_dict(annos):
    anno_id2idx_dict = {}
    for idx,record in enumerate(annos):
        anno_id2idx_dict[record['id']] = idx
    return anno_id2idx_dict

def get_vis_list(data, img_names, cat_ids):
    if img_names == [] and cat_ids == []:
        vis_all = True
    else:
        vis_all = False

    images = data['images']
    annos = data['annotations']
    categories = data['categories']

    # print('cat_ids', cat_ids)
    if cat_ids == []:
        for cat in categories:
            cat_ids.append(cat['id'])
    # print('cat_ids', cat_ids)
    images_vis = []
    if not vis_all and len(img_names) != 0:
        for record in images:
            if record['file_name'] in img_names:
                images_vis.append(record)
    else:
        images_vis = images
    # print(images_vis)
    image_name2id_dict = build_img_name2id_dict(images_vis)
    image_id2name_dict = build_img_id2name_dict(images_vis)
    # imageid2annoid_dict = build_img2anno_dict(annos)

    anno_vis = []
    for record in annos:
        if record['image_id'] in list(image_name2id_dict.values()) and record['category_id'] in cat_ids:
            anno_vis.append(record)
    return anno_vis, image_id2name_dict


def colormap(rgb=False):
    """
    Get colormap

    The code of this function is copied from https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/colormap.py
    """
    color_list = np.array([
        0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
        0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
        1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
        0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
        0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
        0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
        1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
        0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
        0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
        0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
        0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
        0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
        1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
        1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
        0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
        0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
        0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
        0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
        0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
        0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list.astype('int32')

def get_catid2name(data):
    categories = data['categories']
    catid2name = {}
    for category in categories:
        catid2name[category['id']] = category['name']
    return catid2name

def anno_vis(json_path, img_dir, vis_dir, img_names=[], cat_ids=[], nums_img=0, nums_bbox=0, random_seed=322):
    data = json_read(json_path)
    catid2name = get_catid2name(data)
    print('catid2name', catid2name)
    # catid2color = {}
    catid2color = {0: (0, 0, 0),
                   1: (255, 0, 0),
                   2: (0, 255, 0),
                   3: (0, 0, 255),
                   4: (255, 255, 0),
                   5: (0, 255, 255),
                   6: (255, 0, 255),
                   }
    color_list = colormap(rgb=True)
    for catid in range(len(catid2name)):
        idx = np.random.randint(len(color_list))
        catid2color[catid] = color_list[idx]

    if img_names == [] and nums_img >0:
        file_list = os.listdir(img_dir)
        np.random.seed(random_seed)
        np.random.shuffle(file_list)
        img_names = file_list[:nums_img]
    if len(cat_ids)>0:
        cat_ids = [int(id) for id in cat_ids]

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)


    anno_vis, image_id2name_dict = get_vis_list(data, img_names, cat_ids)

    if nums_bbox>0:
        np.random.seed(random_seed)
        np.random.shuffle(anno_vis)
        anno_vis = anno_vis[:nums_bbox]

    imageid2annoid_dict = build_img2anno_dict(anno_vis)
    anno_id2idx_dict = build_anno_id2idx_dict(anno_vis)

    for img_id, anno_ids in tqdm(imageid2annoid_dict.items()):
        if img_id==9:
            print()
        img_name = image_id2name_dict[img_id]
        annos4img = []
        for anno_id in anno_ids:
            anno_idx = anno_id2idx_dict[anno_id]
            anno = anno_vis[anno_idx]
            annos4img.append(anno)
        img_path = os.path.join(img_dir, img_name)
        vis_path = os.path.join(vis_dir, img_name)
        img = img_read(img_path)
        image = Image.fromarray(img)
        draw = ImageDraw.Draw(image)
        for idx in range(len(annos4img)):
            xmin, ymin, w, h = annos4img[idx]['bbox']
            catid = annos4img[idx]['category_id']
            id = annos4img[idx]['id']
            color = tuple(catid2color[catid])
            xmax = xmin + w
            ymax = ymin + h
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=2,
                fill=color)
            text = catid2name[catid]+':%d'%id
            tw, th = draw.textsize(text)
            # font = ImageFont.truetype('arial.ttf', 30)
            font = ImageFont.truetype('arial.ttf')
            draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255), font=font)

        image.save(vis_path, quality=95)

def get_args():
    parser = argparse.ArgumentParser(description='Json Images Information Statistic')

    # parameters
    parser.add_argument('--json_path', type=str, default=r'E:\Huilin\2308_concretespalling\data\ConcreteCracksDetection\train\_annotations.coco.json',
                        help='json path to statistic images information')
    parser.add_argument('--img_dir', type=str, default=r'E:\Huilin\2308_concretespalling\data\ConcreteCracksDetection\train',
                        help='image dir')
    parser.add_argument('--vis_dir', type=str, default=r'E:\Huilin\2308_concretespalling\data\ConcreteCracksDetection\train_vis',
                        help='vis dir')
    parser.add_argument('--img_names', default=[], nargs='+',
                        help='image names list')
    parser.add_argument('--cat_ids', default=[], nargs='+',
                        help='category ids list')
    parser.add_argument('--nums_img', type=int, default=0,
                        help='numbers of imgs to show')
    parser.add_argument('--nums_bbox', type=int, default=0,
                        help='numbers of bboxs to show')
    parser.add_argument('-Args_show', '--Args_show', type=bool, default=False,
                        help='Args_show(default: True), if True, show args info')

    args = parser.parse_args()

    if args.Args_show:
        print('Args'.center(100,'-'))
        for k, v in vars(args).items():
            print('%s = %s' % (k, v))
        print()
    return args

if __name__ == '__main__':
    pass
    # args = get_args()
    # anno_vis(args.json_path, args.img_dir, args.vis_dir,
    #          img_names=args.img_names, cat_ids=args.cat_ids,
    #          nums_img=args.nums_img, nums_bbox=args.nums_bbox)

    # anno_vis(json_path=r'E:\Huilin\2308_concretespalling\data\0111_testdata\coco5_wt\annotations\instance_val.json',
    #          img_dir=r'E:\Huilin\2308_concretespalling\data\0111_testdata\coco5_wt\images_val_w',
    #          vis_dir=r'E:\Huilin\2308_concretespalling\data\0111_testdata\coco5_wt\images_val_w_vis',)

    # anno_vis(json_path=r'E:\Huilin\2308_concretespalling\data\0111_testdata\coco5_wt\annotations\instance_train.json',
    #          img_dir=r'E:\Huilin\2308_concretespalling\data\0111_testdata\coco5_wt\images_train_w',
    #          vis_dir=r'E:\Huilin\2308_concretespalling\data\0111_testdata\coco5_wt\images_train_w_vis',)

    anno_vis(json_path=r'E:\Huilin\2308_concretespalling\data\0111_testdata\\coco6s640_wt\annotations\instance_val.json',
             img_dir=r'E:\Huilin\2308_concretespalling\data\0111_testdata\\coco6s640_wt\images_val_w',
             vis_dir=r'E:\Huilin\2308_concretespalling\data\0111_testdata\\coco6s640_wt\images_val_w_vis',)