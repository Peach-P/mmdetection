# dataset settings
dataset_type = 'VOCDataset'
#data_root = 'data/VOCdevkit/'
data_root = '/xiaopeng/cascade-13class-dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
#     dict(type='LoadImageFromFile'),  # commented by xp
    dict(type='LoadImageFromFile', to_float32=True), # add by xp
    dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),  
    dict(type='Resize', img_scale=(300, 350), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),   # add by xp
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
#         img_scale=(1000, 600),
        img_scale=(300, 350),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
#     samples_per_gpu=2, # commented by xp
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            # ann_file=[
                #data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                # data_root + 'VOC2012/ImageSets/Main/trainval.txt' 
            # ],  commented these lines by xp
#             img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'], commeted this line by xp
            ann_file = data_root + 'ImageSets/Main/train.txt',
            img_prefix= data_root,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        #img_prefix=data_root + 'VOC2007/',   commeted this line by xp
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        # img_prefix=data_root + 'VOC2007/',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
