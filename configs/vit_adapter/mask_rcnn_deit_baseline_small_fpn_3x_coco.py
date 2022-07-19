# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth' # noqa: E501
pretrained = 'pretrained/deit_small_patch16_224-cd65a155_release.pth'
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='ViTBaseline',
        # adapter
        pretrain_size=224,  # default
        embed_dims=384,
        num_heads=6,
        drop_path_rate=0.2,
        pretrained=pretrained,
        # vit
        img_size=224,  # default
        patch_size=16,
        in_channels=3,  # default
        num_classes=80,  # default
        depth=12,
        mlp_ratio=4.,
        drop_rate=0.,  # default
        attn_drop_rate=0.,  # default
        num_fcs=2,  # default
        qkv_bias=True,
        layer_scale=True,  # default
        patch_norm=False,  # default
        norm_cfg=dict(type='LN'),  # default
        act_cfg=dict(type='GELU'),  # default
        window_attn=[
            True, True, False, True, True, False, True, True, False, True,
            True, False
        ],
        window_size=[14, 14, None, 14, 14, None, 14, 14, None, 14, 14, None],
    ),
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],
        out_channels=256,
        num_outs=5))
# optimizer
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                     (576, 1333), (608, 1333), (640, 1333),
                                     (672, 1333), (704, 1333), (736, 1333),
                                     (768, 1333), (800, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1024, 1024),
        allow_negative_crop=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'level_embed': dict(decay_mult=0.),
            'pos_embed': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'bias': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)
