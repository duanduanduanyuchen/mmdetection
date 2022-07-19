# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv import deprecated_api_warning
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.ops.multi_scale_deform_attn import (MultiScaleDeformableAttention,
                                              MultiScaleDeformableAttnFunction)
from mmcv.runner import BaseModule
from torch.nn.init import normal_
from torch.nn.modules.utils import _pair as to_2tuple

from ..builder import BACKBONES
from .vit_baseline import VisionTransformer


class ConvFFN(BaseModule):
    """Implements the ConvFFN from PVTv2.

    Args:
        in_features (int): Channels of input feature.
        hidden_features (int): Channels of hidden feature.
        out_features (int): Channels of output feature.
        act_cfg (dict): Config dict for activation layer.
        cffn_drop_rate (float): Dropout rate. Default: 0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 cffn_drop_rate=0.):
        super(ConvFFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(cffn_drop_rate)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(BaseModule):
    """Implements the depth-wise and scale-wise convolution layer."""

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2,
                                                    W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H,
                                                         W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2,
                                                   W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class MSDeformAttn(MultiScaleDeformableAttention):
    """Implements multi scale deformable attention with mlp ratio.

    Args:
        embed_dims (int): Embedding dimension. Default: 768.
        num_levels (int): Number of feature levels in deformable attn.
            Default: 1.
        num_heads (int): Parallel attention heads in deformable attn.
            Default: 6.
        num_points (int): Number of sampling points per attention head
            per feature level in deformable attn. Default: 4.
        ratio (float): embed_dim / input_dims in deformable attn linear
            proj layer. Default: 1.0
    """

    def __init__(self,
                 embed_dims=768,
                 num_levels=1,
                 num_heads=6,
                 num_points=4,
                 ratio=1.0):

        super(MSDeformAttn, self).__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            im2col_step=64,
            dropout=0.1,
            batch_first=True,
            norm_cfg=None,
            init_cfg=None)
        self.ratio = ratio
        self.value_proj = nn.Linear(embed_dims, int(embed_dims * ratio))
        self.output_proj = nn.Linear(int(embed_dims * ratio), embed_dims)
        self.init_weights()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, Len_in, self.num_heads,
                           int(self.ratio * self.embed_dims) // self.num_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, -1). \
            view(N, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]],
                -1)
            sampling_locations = \
                reference_points[:, :, None, :, None, :] + \
                sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = \
                reference_points[:, :, None, :, None, :2] + \
                sampling_offsets / self.num_points * \
                reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'  # noqa: E501
                .format(reference_points.shape[-1]))
        output = MultiScaleDeformableAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index,
            sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


class Extractor(BaseModule):
    """Implements the extractor of ViT-adapter.

    Args:
        embed_dims (int): Embedding dimension. Default: 768.
        num_heads (int): Parallel attention heads in deformable attn.
            Default: 6.
        num_points (int): Number of sampling points per attention head
            per feature level in deformable attn. Default: 4.
        num_levels (int): Number of feature levels in deformable attn.
            Default: 1.
        deform_ratio (float): embed_dim / input_dims in deformable attn linear
            proj layer. Default: 1.0
        with_cffn (bool):  Whether add CFFN to Extractor. Default: True.
        cffn_ratio (float): embed_dims / input_dims in CFFN. Default: 0.25
        cffn_drop_rate (float): Dropout rate in CFFN. Default: 0.0
        drop_path (float): The stochastic depth rate of CFFN. Default 0.0
        norm_cfg (dict): Config dict for normalization layer in ConvModule.
            Default: dict(type='LN')
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims=768,
                 num_heads=6,
                 num_points=4,
                 num_levels=1,
                 deform_ratio=1.0,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 cffn_drop_rate=0.,
                 drop_path=0.,
                 norm_cfg=None,
                 with_cp=False):

        super(Extractor, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        self.query_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.feat_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MSDeformAttn(
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
            ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=embed_dims,
                hidden_features=int(embed_dims * cffn_ratio),
                cffn_drop_rate=cffn_drop_rate)
            self.ffn_norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.drop_path = build_dropout(
                dict(type='DropPath', drop_prob=drop_path))

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index, H, W):

        def _inner_forward(query, feat):

            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat),
                spatial_shapes, level_start_index, None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(
                    self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(BaseModule):
    """Implements the injector of ViT-adapter.

    Args:
        embed_dims (int): Embedding dimension. Default: 768.
        num_heads (int): Parallel attention heads in deformable attn.
            Default: 6.
        num_points (int): Number of sampling points per attention head
            per feature level in deformable attn. Default: 4.
        num_levels (int): Number of feature levels in deformable attn.
            Default: 3.
        deform_ratio (float): embed_dim / input_dims in deformable attn linear
            proj layer. Default: 1.0
        norm_cfg (dict): Config dict for normalization layer in ConvModule.
            Default: dict(type='LN')
        init_values (float): Initial value of the learnable vector between ViT
            and Injector. Default: 4.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims=768,
                 num_heads=6,
                 num_points=4,
                 num_levels=3,
                 deform_ratio=1.0,
                 norm_cfg=None,
                 init_values=0.,
                 with_cp=False):

        super(Injector, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        self.with_cp = with_cp
        self.query_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.feat_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = MSDeformAttn(
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
            ratio=deform_ratio)
        self.gamma = nn.Parameter(
            init_values * torch.ones((embed_dims)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes,
                level_start_index):

        def _inner_forward(query, feat):

            attn = self.attn(
                self.query_norm(query), reference_points, self.feat_norm(feat),
                spatial_shapes, level_start_index, None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class InteractionBlock(BaseModule):
    """implements an interaction process in ViT-Adapter.

    This process contains several ViT blocks ,an injector on the top and an
    extractor(or multiple extractors) at the bottom.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=6,
                 num_points=4,
                 norm_cfg=None,
                 cffn_drop_rate=0.,
                 drop_path=0.,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 init_values=0.,
                 deform_ratio=1.0,
                 extra_extractor=False,
                 with_cp=False):

        super(InteractionBlock, self).__init__()

        self.injector = Injector(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points,
            num_levels=3,
            deform_ratio=deform_ratio,
            norm_cfg=norm_cfg,
            init_values=init_values,
            with_cp=with_cp)
        self.extractor = Extractor(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_points=num_points,
            num_levels=1,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            cffn_drop_rate=cffn_drop_rate,
            drop_path=drop_path,
            norm_cfg=norm_cfg,
            with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_points=num_points,
                    num_levels=1,
                    norm_cfg=norm_cfg,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    cffn_drop_rate=cffn_drop_rate,
                    drop_path=drop_path,
                    with_cp=with_cp) for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self,
                x,
                c,
                blocks,
                deform_inputs1,
                deform_inputs2,
                H,
                W,
                input_HW=True,
                cls_token=False):
        if cls_token:
            cls, x = x[:, :1, :], x[:, 1:, :]
        x = self.injector(
            query=x,
            reference_points=deform_inputs1[0],
            feat=c,
            spatial_shapes=deform_inputs1[1],
            level_start_index=deform_inputs1[2])
        if cls_token:
            x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            if input_HW:
                x = blk(x, H, W)
            else:
                x = blk(x)
        if cls_token:
            cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.extractor(
            query=c,
            reference_points=deform_inputs2[0],
            feat=x,
            spatial_shapes=deform_inputs2[1],
            level_start_index=deform_inputs2[2],
            H=H,
            W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(
                    query=c,
                    reference_points=deform_inputs2[0],
                    feat=x,
                    spatial_shapes=deform_inputs2[1],
                    level_start_index=deform_inputs2[2],
                    H=H,
                    W=W)
        if cls_token:
            x = torch.cat((cls, x), dim=1)
        return x, c


class SpatialPriorModule(BaseModule):
    """Implements Spatial Prior Module in ViT-Adapter.

    Args:
        inplanes (int): The embeding dimension of convolution layers
            in the middle. Default: 64.
        embed_dims (int): Embedding dimension. Default: 768.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict): Config dict for normalization layer in ConvModule.
            Default: dict(type='SyncBN')
        act_cfg: Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True)
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 inplanes=64,
                 embed_dims=768,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False):

        super(SpatialPriorModule, self).__init__()
        self.with_cp = with_cp
        self.stem = nn.Sequential(
            ConvModule(
                in_channels=3,
                out_channels=inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=inplanes,
                out_channels=inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=inplanes,
                out_channels=inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = ConvModule(
            in_channels=inplanes,
            out_channels=2 * inplanes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            in_channels=2 * inplanes,
            out_channels=4 * inplanes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv4 = ConvModule(
            in_channels=4 * inplanes,
            out_channels=4 * inplanes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.fc1 = nn.Conv2d(
            inplanes,
            embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.fc2 = nn.Conv2d(
            2 * inplanes,
            embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.fc3 = nn.Conv2d(
            4 * inplanes,
            embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.fc4 = nn.Conv2d(
            4 * inplanes,
            embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


@BACKBONES.register_module()
class ViTAdapter(VisionTransformer):
    """The backbone of ViT-Adapter.

    This backbone is the implementation of `Vision Transformer Adapter for
    Dense Predictions <https://arxiv.org/abs/2205.08534>`_.

    Args:
        pretrain_size (int | tuple): Input size for the pretrained model.
            Default: 224.
        conv_inplane (int): The embedding dimension of conv modules in the
            middle of SPM. Default: 64.
        num_points (int): Number of sampling points per attention head
            per feature level in deformable attn. Default: 4.
        depth (int): Number of transformer layers in ViT. Default: 12.
        embed_dims (int): Embedding dimension. Default: 768.
        deform_num_heads (int): Parallel attention heads in deformable attn.
            Default: 6.
        init_values (float): Initial value of the learnable vector between ViT
            and Injector. Default: 0.
        interaction_indexes (list[list]): The ViT layer index pairs of
            injection and extraction. Default: None.
        with_cffn (bool): Whether add CFFN to Extractor.
            Default: True.
        cffn_ratio (float): embed_dims / input_dims in CFFN. Default: 0.25
        cffn_drop_rate (float): Dropout rate in CFFN. Default: 0.
        deform_ratio (float): embed_dim / input_dims in deformable attn linear
            proj layer. Default: 1.0
        add_vit_feature (bool): Whether add ViT output feature to the adapter
            output. Default: True.
        pretrained (str, optional): Model pretrained path. Default: None.
        use_extra_extractor (bool): Whether add 2 more extractor after the
            final extractor. Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        **kwargs: Other arguments for transformer backbone.
    """

    def __init__(self,
                 pretrain_size=224,
                 conv_inplane=64,
                 num_points=4,
                 depth=12,
                 embed_dims=768,
                 deform_num_heads=6,
                 init_values=0.,
                 interaction_indexes=None,
                 with_cffn=True,
                 cffn_ratio=0.25,
                 cffn_drop_rate=0.,
                 deform_ratio=1.0,
                 add_vit_feature=True,
                 pretrained=None,
                 use_extra_extractor=True,
                 with_cp=False,
                 **kwargs):
        super(ViTAdapter, self).__init__(
            pretrained=pretrained,
            depth=depth,
            embed_dims=embed_dims,
            with_cp=with_cp,
            **kwargs)
        self.cls_token = None
        self.num_block = depth
        self.pretrain_size = to_2tuple(pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dims))
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dims=embed_dims, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(
                embed_dims=embed_dims,
                num_heads=deform_num_heads,
                num_points=num_points,
                init_values=init_values,
                drop_path=self.drop_path_rate,
                norm_cfg=self.norm_cfg,
                cffn_drop_rate=cffn_drop_rate,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(interaction_indexes) -
                                  1 else False) and use_extra_extractor),
                with_cp=with_cp) for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dims)
        self.norm2 = nn.SyncBatchNorm(embed_dims)
        self.norm3 = nn.SyncBatchNorm(embed_dims)
        self.norm4 = nn.SyncBatchNorm(embed_dims)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(1, self.pretrain_size[0] // 16,
                                      self.pretrain_size[1] // 16,
                                      -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic',
                                  align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m.init_weights()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def _get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def _deform_inputs(self, x):
        bs, c, h, w = x.shape
        spatial_shapes = torch.as_tensor([(h // 8, w // 8), (h // 16, w // 16),
                                          (h // 32, w // 32)],
                                         dtype=torch.long,
                                         device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self._get_reference_points([(h // 16, w // 16)],
                                                      x.device)
        deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

        spatial_shapes = torch.as_tensor([(h // 16, w // 16)],
                                         dtype=torch.long,
                                         device=x.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self._get_reference_points([(h // 8, w // 8),
                                                       (h // 16, w // 16),
                                                       (h // 32, w // 32)],
                                                      x.device)
        deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

        return deform_inputs1, deform_inputs2

    def forward(self, x):
        deform_inputs1, deform_inputs2 = self._deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, (H, W) = self.patch_embed(x)
        bs, n, dim = x.shape
        if self.pos_embed is not None:
            pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
            x = self.pos_drop(x + pos_embed)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(
                x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(
                x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(
                x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
