_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderRefine',
    pretrained='open-mmlab://resnet18_v1c',
    down_ratio=4,
    num_stages=2,
    backbone=dict(depth=18),
    decode_head=[
        dict(
            type='RefinePSPHead',
            in_channels=512,
            in_index=3,
            channels=128,
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
            ),
        dict(
            type='STDCSimRefineHead',
            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            channels=128,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
            )
    ],
    auxiliary_head=dict(in_channels=256, channels=64))
