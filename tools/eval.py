import os
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../cls2det'))

import numpy as np
from pycocotools.coco import COCO

from cls2det.detection import Detector
from cls2det.utils import Config
from cls2det.utils import COCOeval


def parse_args():
    parser = argparse.ArgumentParser(description='Train RetinaNet model')
    parser.add_argument('--config', help='config file path', type=str, default='configs/detection.py')
    parser.add_argument('--dataset', help='dataset for evaluation, train or val', type=str, default='train')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    cfg = Config.fromfile(cfg_fp)
    detect = Detector(cfg_fp)

    if not os.path.exists(cfg.eval[args.dataset].gt):
        detect.generate_json(cfg.eval[args.dataset].gt, 'Gt', args.dataset)
    print('Gt file generated')
    if not os.path.exists(cfg.eval[args.dataset].dt):
        detect.generate_json(cfg.eval[args.dataset].dt, 'Dt', args.dataset)
    print('Dt file generated')

    gt = COCO(cfg.eval[args.dataset].gt)
    dt = COCO(cfg.eval[args.dataset].dt)

    coco_eval = COCOeval(gt, dt, iouType='bbox')
    coco_eval.params.catIds = [cfg.voc_categories.dog]
    coco_eval.params.iouThrs = np.array([cfg.eval.iou_thres])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
