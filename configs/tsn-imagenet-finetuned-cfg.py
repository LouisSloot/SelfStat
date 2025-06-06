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