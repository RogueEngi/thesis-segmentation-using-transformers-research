# ===== TOP-LEVEL FINE-TUNING =====
_base_ = [
    '../mmsegmentation-main/configs/_base_/models/setr_naive.py',
    '../mmsegmentation-main/configs/_base_/default_runtime.py'
]

# Load weights from the ADE20K pre-trained model
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_512x512_160k_b16_ade20k/setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth'

# ===== DATASET & LOADER =====
dataset_type = 'ISPRSDataset'
data_root = 'data/vaihingen' # Make sure this points to your Google Drive path
crop_size = (512, 512)
batch_size = 2  # This is your "New Batch Size" (total_batch_size=2)
num_workers = 2
num_classes = 6 # Vaihingen has 6 classes

# Data Preprocessor
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# Use standard Batch Norm for single-GPU
norm_cfg = dict(type='BN', requires_grad=True) 

# ===== MODEL =====
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None, # We use load_from, so this is None
    backbone=dict(
        img_size=(512, 512),
        drop_rate=0.,
        # init_cfg is removed because we use load_from
    ),
    decode_head=dict(
        num_classes=num_classes, 
        norm_cfg=norm_cfg), # Make sure head uses new norm_cfg
    auxiliary_head=[
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=0,
            num_classes=num_classes, # Fix class num
            dropout_ratio=0,
            norm_cfg=norm_cfg,        # Fix norm type
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=1,
            num_classes=num_classes, # Fix class num
            dropout_ratio=0,
            norm_cfg=norm_cfg,        # Fix norm type
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='SETRUPHead',
            in_channels=1024,
            channels=256,
            in_index=2,
            num_classes=num_classes, # Fix class num
            dropout_ratio=0,
            norm_cfg=norm_cfg,        # Fix norm type
            act_cfg=dict(type='ReLU'),
            num_convs=2,
            kernel_size=1,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4))
    ],
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

# ===== PIPELINES =====
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True), # Correct: 0 is background
    dict(
        type='RandomResize',
        scale=(512, 512), # Original config had (2048, 512), fixed to (512,512)
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True), # Correct: 0 is background
    dict(type='PackSegInputs')
]

# Dataloaders
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers, # Reduced from 4
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ===== OPTIMIZER & SCHEDULE =====
# Scaled LR for fine-tuning: 0.0001 * (2 / 16) = 0.0000125
optimizer = dict(
    type='AdamW', lr=0.00005, weight_decay=0.01, betas=(0.9, 0.999))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1), # Fine-tune backbone slowly
        }
    )
)

# Learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=30000,
        by_epoch=False)
]

# Training schedule
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=30000, val_interval=3000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=3000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))


# Resuming Training
# After your Colab session dies, change these to resume
# load_from = '/content/mmsegmentation/work_dirs/YOUR_CONFIG_NAME/iter_1000.pth'
# resume = True
