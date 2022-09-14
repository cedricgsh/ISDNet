_base_ = [
    '../_base_/models/isdnet_r50-d8.py', '../_base_/datasets/deepglobe_1224x1224.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://resnet50_v1c',
    down_ratio=4,
    backbone=dict(depth=50),
    decode_head=[
        dict(
            type='RefineASPPHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='STDCLpLsRefineHeadArchAff2',
            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            channels=128,
            num_classes=7,
            reduce=True,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    auxiliary_head=dict(in_channels=1024, channels=256, num_classes=7))