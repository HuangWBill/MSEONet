# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

_base_ = [
    '../_base_/models/ResNet_MSCA_EPFO_r50-d8.py',
    '../_base_/datasets/FLAIR1_my_dataset_512.py', '../_base_/default_runtime_iter.py',
    '../_base_/schedules/schedule_80k_iter.py'
]

custom_imports = dict(imports=['MSEONet.mmseg.datasets.FLAIR1_my_dataset','MSEONet.mmseg.decode_head.msca_head','MSEONet.mmseg.decode_head.epfo_head'])

crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=[
        dict(
            type='MSCA_Head',
            in_channels=2048,
            in_index=3,
            channels=512,
            pool_scales=(1, 2, 3, 6),
            use_CRF=False,
            dropout_ratio=0.1,
            num_classes=12,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='EPFO_Head',
            in_channels=[256],
            in_index=[0],
            channels=256,
            num_fcs=3,
            theta=5,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            num_classes=12,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    # model training and testing settings
    train_cfg=dict(ratio=0.75),
    test_cfg=dict(mode='whole', ratio=0.75)
)
