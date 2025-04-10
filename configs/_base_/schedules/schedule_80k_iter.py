# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False)
]
randomness =dict(seed=0)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000,save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
