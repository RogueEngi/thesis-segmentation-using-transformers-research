# ===== BASE =====
_base_ = [
    # Get the base model structure (we will override this to B5)
    '../mmsegmentation-main/configs/_base_/models/segformer_mit-b0.py',
    # Get the default runtime (hooks, etc.)
    '../mmsegmentation-main/configs/_base_/default_runtime.py'
]

# ===== TOP-LEVEL FINE-TUNING =====
# Load weights from the ADE20K pre-trained model
# YOU MUST DOWNLOAD THIS FILE:
# https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth
load_from = 'downloaded_models/segformer/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth'

# ===== DATASET & LOADER (from SETR) =====
dataset_type = 'ISPRSDataset'
data_root = 'data/vaihingen' # Make sure this points to your Google Drive path
crop_size = (512, 512)
batch_size = 2  # This is your "New Batch Size" (total_batch_size=2)
num_workers = 2
num_classes = 6 # Vaihingen has 6 classes

# ===== DATA PREPROCESSOR (SegFormer uses standard ImageNet norm) =====
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

# ===== MODEL (Override base MiT-B0 with MiT-B5) =====
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None, # We use load_from
    backbone=dict(
        type='MixVisionTransformer',
        # These params override the mit-b0 base
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
        num_heads=[1, 2, 5, 8],
        # Must also set norm_cfg for single GPU
    ),
    decode_head=dict(
        type='SegformerHead',
        # These params override the mit-b0 base
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes, # Set to 6
        norm_cfg=norm_cfg, # Use BN for single GPU
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    # Use the same "fair" test method as SETR
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

# ===== PIPELINES (from SETR) =====
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True), # Correct: 0 is background
    dict(
        type='RandomResize',
        scale=(512, 512),
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

# ===== DATALOADERS (from SETR) =====
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
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# ===== EVALUATOR (from SETR) =====
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ===== OPTIMIZER & SCHEDULE (from SETR) =====
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
            # Segformer's official config has different rules, but 
            # for fairness, we stick to the "slow backbone" strategy.
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

# ===== HOOKS (from SETR) =====
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=3000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))


