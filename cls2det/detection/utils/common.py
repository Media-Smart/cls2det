import os
import xml.etree.ElementTree as ET

import matplotlib.font_manager as fm
import numpy as np
from PIL import ImageDraw, ImageFont


def get_label(path):
    """get label info from txt file"""
    labels = []
    with open(path, encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            splited = line.strip().split()
            labels.append(splited[-1])
    return labels


def parse_rec(filename):
    """get annotations from xml files"""
    tree = ET.parse(filename)
    annots = []
    ann_tag = tree.getroot()
    size_tag = ann_tag.find('size')
    image_width = int(size_tag.find('width').text)
    image_height = int(size_tag.find('height').text)
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        annots.append(obj_struct)

    return annots, image_width, image_height


def get_img_annotations(img_list, annotations_dir, standard='dog', single_dog=False):
    annotation = []
    with open(img_list) as f:
        lines = f.readlines()
        for line in lines:
            splited = line.strip().split()
            if splited[1] == '1':
                fname = splited[0] + '.jpg'
                box = []
                label = []
                ann = os.path.join(annotations_dir, splited[0] + '.xml')
                rec, w, h = parse_rec(ann)
                for r in rec:
                    if r['name'] == standard:
                        box.append(r['bbox'])
                        label.append(r['name'])
                if single_dog and len(box) > 1:
                    continue
                else:
                    annotation.append({'fname': fname, 'annots': {'w': w, 'h': h, 'boxes': box, 'labels': label}})
    return annotation


def draw(fname, img, dets, scores, save=None):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), size=24)
    if len(dets):
        for (box, score) in zip(dets.tolist(), scores.tolist()):
            draw.rectangle(box, outline='blue', width=2)
            draw.text((box[0], box[1]), str(round(score, 2)), fill=(0, 255, 0), font=font)

    # img.show()
    if save is not None:
        if not os.path.exists(save):
            os.makedirs(save)
        save_name = fname.split('/')[-1]
        img.save(os.path.join(save, save_name))


def get_scale(params):
    max_scale = 130 / params.min_size
    min_scale = 130 / params.max_size
    factor = np.power(max_scale / min_scale, 1 / (params.scale_num - 1))
    scales = [min_scale * np.power(factor, i) for i in range(params.scale_num)]
    return scales


def get_dog_gt(anno):
    gt = []
    for an in anno:
        if an['name'] == 'dog':
            gt.append(an['bbox'])
    return gt


def data2coco(data, file, voc_categories):
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    categories = voc_categories
    bnd_id = 1
    for d in data:
        filename = d['fname']
        image_id = int(os.path.splitext(filename)[0])
        width = d['annots']['w']
        height = d['annots']['h']
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)
        i = 0
        for box, label in zip(d['annots']['boxes'], d['annots']['labels']):
            category_id = categories[label]
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            if file == 'Dt':
                ann.update({'score': d['annots']['scores'][i]})
            json_dict['annotations'].append(ann)
            bnd_id += 1
            i += 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    return json_dict
