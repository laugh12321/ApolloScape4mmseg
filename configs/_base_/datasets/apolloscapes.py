# dataset settings
dataset_type = 'ApolloScapesDataset'
data_root = './apolloscapes/lane_segmentation'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(3384, 1020), ratio_range=(0.5, 2.0)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3384, 1020),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='ColorImage/train',
        ann_dir='MaskLabels/train',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type, 
        data_root=data_root, 
        img_dir='ColorImage/val', 
        ann_dir='MaskLabels/val', 
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type, 
        data_root=data_root, 
        img_dir='ColorImage/val', 
        ann_dir='MaskLabels/val', 
        pipeline=test_pipeline
    )
)