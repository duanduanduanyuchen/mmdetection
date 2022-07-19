# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# pretrained = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth' # noqa: E501
pretrained = 'pretrained/deit_tiny_patch16_224-a1311bcf.pth'
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        # adapter
        pretrain_size=224,  # default
        conv_inplane=64,
        num_points=4,  # default
        depth=12,
        embed_dims=192,
        deform_num_heads=6,
        init_values=0.,  # default
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        with_cffn=True,  # default
        cffn_ratio=0.25,
        cffn_drop_rate=0.,  # default
        deform_ratio=1.0,
        add_vit_feature=True,  # default
        pretrained=pretrained,
        use_extra_extractor=True,
        with_cp=False,  # default
        # vit
        img_size=224,  # default
        patch_size=16,
        in_channels=3,  # default
        num_classes=80,  # default
        num_heads=3,
        mlp_ratio=4.,
        drop_rate=0.,  # default
        attn_drop_rate=0.,  # default
        drop_path_rate=0.1,
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
        in_channels=[192, 192, 192, 192],
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
