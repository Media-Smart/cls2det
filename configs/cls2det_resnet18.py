# commom
gpu = 0
debug = False

root = 'cls2det/data/'

data = dict(
    fname_list=dict(
        train=root + 'VOC2012/ImageSets/Main/dog_train.txt',
        val=root + 'VOC2012/ImageSets/Main/dog_val.txt',
    ),
    img_dir=root + 'VOC2012/JPEGImages',
    ann_dir=root + 'VOC2012/Annotations',
    class_txt=root + 'imagenet.txt',
)

# background hyper-parameter
thres = dict(
    std_thres=2e-4,
    conf_thres=0.3,
)

# connected region
region_params = dict(
    center_mode='gravity',
    low=14,
    high=29,
)

# multi-scale parameter
scales = dict(
    scale_num=7,
    min_size=35,
    max_size=474,
)

# post process
post_params = dict(
    use_twostage=True,
    use_size_filter=True,
    percent=0.15,
    use_nms=True,
    nms_thres=0.01,
    save_images=True,
    save_folder=root + 'result',  # folder for image saving
)

# json files and parameter for evaluation
evaluation = dict(
    folder=(root + 'eval/'),
    train=dict(
        gt=root + 'eval/train_Gt.json',
        dt=root + 'eval/train_Dt.json'
    ),
    val=dict(
        gt=root + 'eval/val_Gt.json',
        dt=root + 'eval/val_Dt.json'
    ),
    iou_thres=0.5
)
