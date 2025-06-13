# data files/settings
file_train_labels = "data/multisubjects/train_list.txt"
file_val_labels = "data/multisubjects/val_list.txt"
file_test_labels = "data/multisubjects/test_list.txt"
data_root_train = "data/multisubjects/videos"
data_root_val = "data/multisubjects/videos"
dataset_type = "VideoDataset"
file_client_args = dict(io_backend='disk')

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        arch='tiny',  
        pretrained=None,
        pretrained2d=True,
        patch_size=(2, 4, 4),
        in_channels=3,
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        patch_norm=True,
        frozen_stages=-1,
        with_cp=False,
        out_indices=(3,),
        out_after_downsample=False),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=3, # dribbling, layup, shooting
        spatial_type='avg',
        dropout_ratio=0.5),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=None,
    test_cfg=None
)

# pipelines
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375])),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=4, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375])),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=1,  # reduced for CPU training
    num_workers=2,  # reduced for CPU
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=file_train_labels,
        data_prefix=dict(video=data_root_train),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,  # reduced for CPU training
    num_workers=2,  # reduced for CPU
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=file_val_labels,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,  # reduced for CPU
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=file_test_labels,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

# evaluators
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    accumulative_counts=8,  # accumulation to simulate larger batch (1 * 8 = 8 effective batch size)
    optimizer=dict(
        type='AdamW',
        lr=1e-5,  # Rreduced LR for CPU training
        betas=(0.9, 0.999),
        weight_decay=0.01),  # Reduced weight decay
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),  # lower LR for pretrained backbone
            'cls_head': dict(lr_mult=1.0),  # normal LR for new head -- higher than backbone as per Video Swin Transformer paper
            'norm': dict(decay_mult=0.), 
            'bias': dict(decay_mult=0.),
        }),
    clip_grad=dict(max_norm=1.0))

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2),
    dict(
        type='CosineAnnealingLR',
        T_max=13, 
        eta_min=1e-7,
        by_epoch=True,
        begin=2,
        end=15)
]

# configurations
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=15,  # reduced epochs for CPU training
    val_begin=1,
    val_interval=2)  # validate every 2 epochs to save time

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# runtime hooks
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),  #lLog less frequently
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=3,  # save checkpoints every 3 epochs
        save_best='auto',
        max_keep_ckpts=2),  # keep fewer checkpoints to save disk space
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

# environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

# visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=vis_backends)

auto_scale_lr = dict(batch_size=2, enable=False)

log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
log_level = 'INFO'

load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_base_patch244_window877_kinetics400_22k.pth" # pretrained model

resume = False

default_scope = 'mmaction'