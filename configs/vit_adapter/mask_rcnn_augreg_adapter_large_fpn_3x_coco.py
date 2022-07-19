_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_augreg.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# pretrained = 'https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz' # noqa: E501
# pretrained = 'https://github.com/czczup/ViT-Adapter/releases/download/v0.1.6/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth' # noqa: E501
pretrained = 'pretrained/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.pth'  # noqa: E501
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='ViTAdapter',
        # adapter
        pretrain_size=384,  # default
        conv_inplane=64,
        num_points=4,  # default
        depth=24,
        embed_dims=1024,
        deform_num_heads=16,
        init_values=0.,  # default
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        with_cffn=True,  # default
        cffn_ratio=0.25,
        cffn_drop_rate=0.,  # default
        deform_ratio=0.5,
        add_vit_feature=True,  # default
        pretrained=pretrained,
        use_extra_extractor=True,
        with_cp=False,  # default
        # vit
        img_size=384,  # default
        patch_size=16,
        in_channels=3,  # default
        num_classes=1000,  # default
        num_heads=16,
        mlp_ratio=4.,
        drop_rate=0.,  # default
        attn_drop_rate=0.,  # default
        drop_path_rate=0.4,
        num_fcs=2,  # default
        qkv_bias=True,
        layer_scale=True,  # default
        patch_norm=False,  # default
        norm_cfg=dict(type='LN'),  # default
        act_cfg=dict(type='GELU'),  # default
        window_attn=[
            True, True, True, True, True, False, True, True, True, True, True,
            False, True, True, True, True, True, False, True, True, True, True,
            True, False
        ],
        window_size=[
            14, 14, 14, 14, 14, None, 14, 14, 14, 14, 14, None, 14, 14, 14, 14,
            14, None, 14, 14, 14, 14, 14, None
        ],
    ),
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='MMSyncBN',
                      requires_grad=True)),  # BN can be removed
    roi_head=dict(
        bbox_head=dict(norm_cfg=dict(type='MMSyncBN',
                                     requires_grad=True)),  # BN can be removed
        mask_head=dict(norm_cfg=dict(type='MMSyncBN',
                                     requires_grad=True)))  # BN can be removed
)
# optimizer
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
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
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.90))
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True,
)
