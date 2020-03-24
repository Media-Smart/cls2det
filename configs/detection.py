# commom
stride = 32
gpu = 0
debug = False
cls = 'dog'

# model
model = dict(
    arch='resnet18',
    num_classes=1000,
    pre_trained=True,
)

# data
work_dir = 'cls2det/'
norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = dict(
    fname_list=dict(
        train=work_dir + 'data/VOC2012/ImageSets/main/dog_train.txt',
        val=work_dir + 'data/VOC2012/ImageSets/main/dog_val.txt',
    ),
    img_dir=work_dir + 'data/VOC2012/JPEGImages',
    ann_dir=work_dir + 'data/VOC2012/Annotations',
    class_txt=work_dir + 'data/imagenet.txt',
)
voc_categories = {'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
                  'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9,
                  'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13,
                  'motorbike': 14, 'person': 15, 'pottedplant': 16,
                  'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}

# background hyper-parameter
std_thres = 2e-4
conf_thres = 0.3

# connect_region limits
low = 14
high = 29

# pyramid range
scales = dict(
    min_size=35,
    max_size=474,
)

# region proposal
size = 130
type = 'gravity'

# post process
use_twostage = True
use_size_filter = True
percent = 0.15
use_nms = True
nms_thres = 0.01

# set True if you want to save the dectected imgs
save_images = True

# folder where the result imgs are stored
save_folder = work_dir + 'data/result' if save_images else None

# json files and parameter for evaluation
eval = dict(
    train=dict(
        gt=work_dir + 'data/eval/train_Gt.json',
        dt=work_dir + 'data/eval/train_Dt.json'
    ),
    val=dict(
        gt=work_dir + 'data/eval/val_Gt.json',
        dt=work_dir + 'data/eval/val_Dt.json'
    ),
    iou_thres=0.5
)