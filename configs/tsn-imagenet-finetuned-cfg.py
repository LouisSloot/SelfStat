file_train_labels = "mmaction2/MultiSubjects_train_val_test/MultiSubjects_train_val_test/labels_train.txt"
file_val_labels = "mmaction2/MultiSubjects_train_val_test/MultiSubjects_train_val_test/labels_val.txt"
data_root_train = "mmaction2/MultiSubjects_train_val_test/MultiSubjects_train_val_test/videos_train"
data_root_val = "mmaction2/MultiSubjects_train_val_test/MultiSubjects_train_val_test/videos_val"
dataset_type = "VideoDataset"
auto_scale_lr = dict(batch_size=2, enable=False)
default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
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
        num_classes=3, # adjust as action classes change
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