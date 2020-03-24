import json
import os

import numpy as np
from PIL import Image

from .util import (box_rescale, cell2box, data2coco, draw, get_std,
                   get_img_annotations, get_label, get_scale, nms,
                   remover_outlier, resize)
from cls2det.utils import Classifier
from cls2det.utils import Config


class Detector:
    def __init__(self, config_file):
        self.cfg = Config.fromfile(config_file)
        self.classifier = Classifier(self.cfg)
        self.labels = get_label(self.cfg.data.class_txt)
        self.scales = get_scale(self.cfg)
        if self.cfg.cls != 'dog':
            raise Exception('currently this tool only support class "dog", '
                            'other classes scoming soon!')

    def get_fg(self, img):
        fm = self.classifier.predict(img, type='fm')
        h, w, c = fm.shape
        score, indices = fm.max(dim=2)
        value, indices, fm = score.cpu().numpy(), indices.cpu().numpy(), fm.cpu().numpy()

        std_fm = [[get_std(fm[i][j]) for j in range(w)] for i in range(h)]
        std_fm = np.array(std_fm)
        bg_mask = std_fm > self.cfg.std_thres

        cls_mask = [[True if self.labels[indices[i][j]] != 'bg' else False for j in range(w)] for i in range(h)]
        cls_mask = np.array(cls_mask)

        conf_mask = value > self.cfg.conf_thres
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
        if self.cfg.cls in class_dict.keys():
            coord_list = class_dict[self.cfg.cls]
            boxes, confs = cell2box(coord_list, value, img.size, self.cfg)
        return boxes, confs

    def filter_proposal_single(self, partial_img):
        label, value = self.classifier.predict(partial_img, type='cls')
        if label == self.cfg.cls and value > self.cfg.conf_thres:
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
            res = remover_outlier(reg, hs, ws, scale=self.cfg.percent)
            mask_all.extend(res)
        dets = dets[mask_all]
        scores = scores[mask_all]
        return dets, scores

    def detect_single(self, fname):
        img = Image.open(fname)
        dets, scores = [], []
        for scale in self.scales:
            scaled_img = resize(img, scale)
            boxes, confs = self.get_proposal(scaled_img)
            if len(boxes) > 0:
                for box, conf in zip(boxes, confs):
                    if self.cfg.use_twostage:
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
        if self.cfg.use_size_filter and len(dets):
            dets, scores = self.size_filter(dets, scores)

        if self.cfg.use_nms and len(dets):
            keep, _ = nms(dets, scores, thresh=self.cfg.nms_thres)
            dets = dets[keep]
            scores = scores[keep]
        draw(fname, img, dets, scores, save=self.cfg.save_folder)
        return np.array(dets), np.array(scores), np.array(len(dets) * [self.cfg.cls])

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
        data = data2coco(annos, f, self.cfg.voc_categories)
        json_fp = open(json_save, 'w')
        json_str = json.dumps(data)
        json_fp.write(json_str)
        json_fp.close()
