# ===== BASE =====
_base_ = [
    # Get the base model structure
    '../mmsegmentation-main/configs/_base_/models/segmenter_vit-b16_mask.py',
    # Get the default runtime (hooks, etc.)
    '../mmsegmentation-main/configs/_base_/default_runtime.py'
]

# ===== TOP-LEVEL FINE-TUNING =====
# Load weights from the ADE20K pre-trained model
load_from = 'downloaded_models/segmenter/segmenter_vit-l_mask_8x1_512x512_160k_ade20k_20220105_162750-7ef345be.pth'

# ===== DATASET & LOADER (from SETR) =====
dataset_type = 'LoveDADataset'
data_root = 'data/loveDA' # Make sure this points to your Google Drive path
crop_size = (512, 512)
batch_size = 2  # This is your "New Batch Size" (total_batch_size=2)
num_workers = 2
num_classes = 7 # Vaihingen has 6 classes LoveDa 7

# ===== DATA PREPROCESSOR (CRITICAL - MUST USE SEGMENTER'S) =====
# This is the ONLY change from the SETR "fairness" template, and it's
# necessary because Segmenter was pre-trained with this normalization.
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# ===== MODEL (ViT-Large) =====
# We must override the base ViT-Base model with the ViT-Large parameters
# to match our checkpoint.
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)

model = dict(
    data_preprocessor=data_preprocessor, # Use our new preprocessor
    pretrained=None, # We use load_from, so this is None
    backbone=dict(
        type='VisionTransformer',
        img_size=crop_size,
        patch_size=16,
        in_channels=3,
        embed_dims=1024,    # ViT-Large param
        num_layers=24,    # ViT-Large param
        num_heads=16,     # ViT-Large param
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode='bicubic',
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,   # ViT-Large param
        channels=1024,    # ViT-Large param
        num_classes=num_classes, # Our 6 classes
        num_layers=2,
        num_heads=16,     # ViT-Large param
        embed_dims=1024,  # ViT-Large param
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
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
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
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


