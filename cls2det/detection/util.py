import math
import os
import xml.etree.ElementTree as ET
from functools import partial

import matplotlib.font_manager as fm
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def get_std(data):
    return np.var(data)


def nms(dets, scores, thresh):
    """
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    """
    if isinstance(scores, list):
        scores = np.array(scores)

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    Connected_region = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        same_region = np.where(ovr > thresh)[0]

        Connected_region.append([i, order[same_region + 1]])
        order = order[inds + 1]

    return keep, Connected_region


def index2box(item, h, w, ratio=1.0, area=130):
    centery = min(16 + 32 * item[0], h)
    centerx = min(16 + 32 * item[1], w)

    ratioed_w = area * math.sqrt(ratio) / 2
    ratioed_h = area / math.sqrt(ratio) / 2

    x1 = max(0, int(centerx - ratioed_w))
    y1 = max(0, int(centery - ratioed_h))
    x2 = min(w, int(centerx + ratioed_w))
    y2 = min(h, int(centery + ratioed_h))
    return [x1, y1, x2, y2]


def cell2box(coord_list, value, img_size, cfg):
    w, h = img_size
    boxes, confs = [], []
    if cfg.type == 'direct':
        for item in coord_list:
            box = index2box(item, h, w)
            conf = value[item[0], item[1]]
            boxes.append(box)
            confs.append(conf)
        boxes = np.array(boxes)
        confs = np.array(confs)
        keep, _ = nms(boxes, confs, 0.01)
        boxes = boxes[keep]
        confs = confs[keep]
    else:
        regions = connected_region(coord_list, value.shape, cfg.low, cfg.high)
        result = get_center(regions, value, cfg.type)
        for res in result:
            box = index2box(res[0], h, w, res[2], cfg.size)
            boxes.append(box)
            confs.append(res[1])
    return np.array(boxes), np.array(confs)


def connected_region(coord_list, fm_shape, low, high):
    (h, w) = fm_shape
    if len(coord_list) == 1:
        return [coord_list]

    regions = []
    mask = [[True if (i, j) in coord_list else False for j in range(w)] for i in range(h)]
    mask = np.array(mask)
    visited = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if mask[i][j] and visited[i][j] == 0:
                reg = []
                dfs(mask, visited, i, j, reg)
                if low < len(reg) < high:
                    regions.append(reg)
    return regions


def dfs(mask, visited, i, j, reg):
    # direct = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    direct = [[-1, 0], [0, 1], [1, 0], [0, -1], [-1, -1], [-1, 1], [1, 1], [1, -1]]
    if i < 0 or i >= mask.shape[0]:
        return
    if j < 0 or j >= mask.shape[1]:
        return
    if not mask[i][j] or visited[i][j] == 1:
        return
    reg.append((i, j))
    visited[i][j] = 1
    for d in direct:
        dfs(mask, visited, i + d[0], j + d[1], reg)


def get_center(regions, value, type):
    result = []
    for region in regions:
        conf_sum = 0
        weight_coord = np.zeros((2,))
        left, right, top, bottle = 9999, 0, 9999, 0
        for p in region:
            left = min(left, p[1])
            right = max(right, p[1])
            top = min(top, p[0])
            bottle = max(bottle, p[0])
            p = np.array(p)
            conf_sum += value[p[0]][p[1]]
            weight_coord += (p * value[p[0]][p[1]])
        w = right - left + 1
        h = bottle - top + 1
        ratio = w / h
        if type == 'gravity':
            center = weight_coord / conf_sum
        elif type == 'geometry':
            center = np.array([(top + bottle) / 2, (left + right) / 2])
        else:
            raise Exception(f'no such type: {type}, '
                            f'it should be among "direct", "gravity" and "geometry"')
        index = get_nearest_index(center, region)
        score = value[index[0]][index[1]]
        result.append([center, score, ratio])
    return result


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


def get_nearest_index(cenetr, region):
    region = np.array(region)
    dist = np.linalg.norm(cenetr - region, axis=1)
    index = np.argmin(dist, axis=0)
    return region[index]


def get_scale(cfg):
    max_scale = cfg.size / cfg.scales.min_size
    min_scale = cfg.size / cfg.scales.max_size
    factor = math.sqrt(2) / 2
    scales = []
    while max_scale > min_scale:
        scales.append(max_scale)
        max_scale *= factor
    return scales


def resize(img, scale, keep_ratio=True):
    w, h = img.size
    width_new = int(w * scale)
    height_new = int(h * scale)
    if keep_ratio:
        img_new = img.resize((width_new, height_new), Image.BILINEAR)
    return img_new


def box_rescale(x, size, scale):
    w, h = size
    x1 = max(0, int(x[0] / scale))
    y1 = max(0, int(x[1] / scale))
    x2 = min(w, int(x[2] / scale))
    y2 = min(h, int(x[3] / scale))
    return np.array([x1, y1, x2, y2])


def xyxy2xywh(x, scale):
    return np.array([(x[0] + x[2]) / 2, (x[1] + x[3]) / 2, (x[2] - x[0]) / scale, (x[3] - x[1]) / scale])


def xywh2xyxy(x, w, h):
    x1 = max(0, int(x[0] - x[2] / 2))
    y1 = max(0, int(x[1] - x[3] / 2))
    x2 = min(w, int(x[0] + x[2] / 2))
    y2 = min(h, int(x[1] + x[3] / 2))
    return np.array([x1, y1, x2, y2], dtype=np.int)


def get_dog_gt(anno):
    gt = []
    for an in anno:
        if an['name'] == 'dog':
            gt.append(an['bbox'])
    return gt


def remover_outlier(reg, hs, ws, scale):
    res = []
    res.append(reg[0])
    for it in reg[1]:
        maskh = hs[it] > (scale * hs[reg[0]])
        maskw = ws[it] > (scale * ws[reg[0]])
        if maskw & maskh:
            res.append(it)
    return res


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


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
