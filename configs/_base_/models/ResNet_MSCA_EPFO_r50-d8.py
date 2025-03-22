# Copyright (c) Wubiao Huang (https://github.com/HuangWBill).

norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='CascadeEncoderDecoder',
    data_preprocessor=data_preprocessor,
    num_stages=2,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=3,
        num_stages=4,
        base_channels=64,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=[
        dict(
            type='PSP_New_Head',
            in_channels=2048,
            in_index=3,
            channels=512,
            pool_scales=(1, 2, 3, 6),
            use_CRF=False,
            dropout_ratio=0.1,
            num_classes=6,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='Boundry_Point_Head',
            in_channels=[256],
            in_index=[0],
            channels=256,
            num_fcs=3,
            theta=5,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            num_classes=6,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
