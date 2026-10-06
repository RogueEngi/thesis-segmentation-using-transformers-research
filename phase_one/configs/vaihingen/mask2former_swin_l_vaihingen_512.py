# ===== BASE =====
_base_ = [
    # Inherit the full Mask2Former (Swin-B) model structure
    '../mmsegmentation-main/configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640.py'
]

# ===== TOP-LEVEL FINE-TUNING =====
# Load weights from the ADE20K pre-trained Swin-Large model
load_from = 'downloaded_models/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth'

# ===== DATASET & LOADER (from SETR) =====
dataset_type = 'ISPRSDataset'
data_root = 'data/vaihingen' # Make sure this points to your Google Drive path
crop_size = (512, 512)
batch_size = 2  # This is your "New Batch Size" (total_batch_size=2)
num_workers = 2
num_classes = 6 # Vaihingen has 6 classes

# ===== DATA PREPROCESSOR (Override base's 640x640 size) =====
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size) # Set to (512, 512)

# ===== MODEL (Override Swin-B with Swin-L) =====
model = dict(
    data_preprocessor=data_preprocessor, # Use our 512x512 preprocessor
    backbone=dict(
        # These overrides change Swin-B to Swin-L
        embed_dims=192,
        num_heads=[6, 12, 24, 48],
        # Remove base's init_cfg, we use load_from
        init_cfg=None), 
    decode_head=dict(
        in_channels=[192, 384, 768, 1536], # Match Swin-L
        num_classes=num_classes, # Set to 6
        # Must update the class_weight for the new num_classes
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]) # [1.0, ..., 1.0, 0.1]
        ),
    # Use the same "fair" test method as SETR
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

# ===== PIPELINES (from SETR) =====
# Override the base's 'RandomChoiceResize' pipeline for fairness
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
# Override base dataloaders to use our fair dataset and pipelines
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

# ===== OPTIMIZER & SCHEDULE (MODEL-SPECIFIC) =====

# This `depths` variable is from the base Swin-B config and is
# required for the `custom_keys` logic to work.
depths = [2, 2, 18, 2]

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00005, # SCALED LR: 0.0001 * (2 / 16)
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))
optim_wrapper = dict(
    _delete_=True, # DELETE the OptimWrapper from the base config
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

# ===== END OF MODEL-SPECIFIC OPTIMIZER =====

# Learning policy (from SETR)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=30000,
        by_epoch=False)
]

# Training schedule (from SETR)
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=30000, val_interval=3000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ===== HOOKS (from SETR) =====
# Override base hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=3000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))


