import os
import time
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../cls2det'))

from cls2det.detection import Detector


def parse_args():
    parser = argparse.ArgumentParser(description='cls2det')
    parser.add_argument('--config', help='config file path', type=str, default='configs/cls2det_resnet18.py')
    parser.add_argument('--img_path', help='img for demo', type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_fp = args.config
    detector = Detector(cfg_fp)
    print(f'detect img {args.img_path}')
    starttime = time.time()
    dets, scores, labels = detector.detect_single(args.img_path)
    print(f'{len(dets)} dog are detected in {time.time() - starttime} second')


if __name__ == '__main__':
    main()
