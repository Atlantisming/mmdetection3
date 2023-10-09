_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
]

dataset_type = 'CocoDataset'
data_root = './datasets/zeroshot_coco/'

seen_class = ('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck',
              'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra',
              'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis',
              'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana',
              'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza',
              'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse',
              'remote', 'microwave', 'oven', 'toaster', 'refrigerator', 'book',
              'clock', 'vase', 'toothbrush')

unseen_class = ('umbrella', 'cow', 'cup', 'bus', 'keyboard', 'skateboard',
                'dog', 'couch', 'tie', 'snowboard', 'sink', 'elephant', 'cake',
                'scissors', 'airplane', 'cat', 'knife')

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', )
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='zero-shot/instances_train2017_seen_2.json',
        metainfo=dict(classes=seen_class),
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='zero-shot/instances_val2017_seen_2.json',
        metainfo=dict(classes=seen_class),
        # ann_file='zero-shot/instances_val2017_unseen_2.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'zero-shot/instances_val2017_seen_2.json',
    # ann_file=data_root + 'zero-shot/instances_val2017_unseen_2.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
