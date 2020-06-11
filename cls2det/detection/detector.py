import json
import os

import numpy as np
from PIL import Image

from .utils import (box_rescale, cell2box, data2coco, draw,
                    get_img_annotations, get_label, get_scale, nms,
                    remove_outlier, resize)
from .classifier import Classifier
from cls2det.utils import Config


class Detector:
    def __init__(self, config_file):
        self.cfg = Config.fromfile(config_file)
        self.classifier = Classifier(self.cfg)
        self.labels = get_label(self.cfg.data.class_txt)
        self.scales = get_scale(self.cfg.scales)
        self.cls = 'dog'
        self.voc_categories = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
                               'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9,
                               'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13,
                               'motorbike': 14, 'person': 15, 'pottedplant': 16,
                               'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

    def get_fg(self, img):
        params = self.cfg.thres
        feature_map = self.classifier.predict(img, mode='fm')
        h, w, c = feature_map.shape
        score, indices = feature_map.max(dim=2)
        value, indices, feature_map = score.cpu().numpy(), indices.cpu().numpy(), feature_map.cpu().numpy()

        std_fm = np.std(feature_map, axis=2)
        bg_mask = std_fm > params.std_thres

        cls_mask = [[True if self.labels[indices[i][j]] != 'bg' else False for j in range(w)] for i in range(h)]
        cls_mask = np.array(cls_mask)

        conf_mask = value > params.conf_thres
        total_mask = bg_mask & cls_mask & conf_mask
        class_dict = dict()
        for i in range(h):
            for j in range(w):
                if total_mask[i][j]:
                    cls = self.labels[indices[i][j]]
                    if cls in class_dict.keys():
                        class_dict[cls].append((i, j))
                    else:
                        class_dict[cls] = [(i, j)]
        return class_dict, value

    def get_proposal(self, img):
        class_dict, value = self.get_fg(img)
        boxes, confs = [], []
        if self.cls in class_dict.keys():
            coord_list = class_dict[self.cls]
            boxes, confs = cell2box(coord_list, value, img.size, self.cfg.region_params)
        return boxes, confs

    def filter_proposal_single(self, partial_img):
        label, value = self.classifier.predict(partial_img, mode='cls')
        if label == self.cls and value > self.cfg.thres.conf_thres:
            return np.array(value)
        else:
            return None

    def size_filter(self, dets, scores):
        mask_all = []
        hs = dets[:, 2] - dets[:, 0]
        ws = dets[:, 3] - dets[:, 1]
        areas = np.sqrt(hs * ws)
        _, regions = nms(dets, areas, 0)
        for reg in regions:
            res = remove_outlier(reg, hs, ws, scale=self.cfg.post_params.percent)
            mask_all.extend(res)
        dets = dets[mask_all]
        scores = scores[mask_all]
        return dets, scores

    def detect_single(self, fname):
        params = self.cfg.post_params
        img = Image.open(fname)
        dets, scores = [], []
        for scale in self.scales:
            scaled_img = resize(img, scale)
            boxes, confs = self.get_proposal(scaled_img)
            if len(boxes) > 0:
                for box, conf in zip(boxes, confs):
                    if params.use_twostage:
                        partial_img = scaled_img.crop(box)
                        value = self.filter_proposal_single(partial_img)
                        if value:
                            box = box_rescale(box, img.size, scale)
                            dets.append(box)
                            scores.append(value)
                    else:
                        box = box_rescale(box, img.size, scale)
                        dets.append(box)
                        scores.append(conf)

        dets, scores = np.array(dets), np.array(scores)
        if params.use_size_filter and len(dets):
            dets, scores = self.size_filter(dets, scores)

        if params.use_nms and len(dets):
            keep, _ = nms(dets, scores, thresh=params.nms_thres)
            dets = dets[keep]
            scores = scores[keep]
        if params.save_images:
            draw(fname, img, dets, scores, save=params.save_folder)
        return np.array(dets), np.array(scores), np.array(len(dets) * [self.cls])

    def generate_json(self, json_save, f='Gt', folder='train'):
        fname_list = self.cfg.data.fname_list.train if folder == 'train' else self.cfg.data.fname_list.val
        annos = get_img_annotations(fname_list, self.cfg.data.ann_dir, single_dog=False)
        if f != 'Gt':
            for anno in annos:
                fname = anno['fname']
                fname = os.path.join(self.cfg.data.img_dir, fname)
                dets, scores, labels = self.detect_single(fname)
                anno['annots']['boxes'] = dets.tolist()
                anno['annots']['labels'] = labels.tolist()
                anno['annots']['scores'] = scores.tolist()
        data = data2coco(annos, f, self.voc_categories)
        json_fp = open(json_save, 'w')
        json_str = json.dumps(data)
        json_fp.write(json_str)
        json_fp.close()
