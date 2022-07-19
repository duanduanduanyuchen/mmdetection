# Copyright (c) OpenMMLab. All rights reserved.
import logging
import math
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import BaseModule, CheckpointLoader, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed
from ..utils.ckpt_convert import vit_converter


class WindowedAttention(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 window_size=14,
                 pad_mode='constant',
                 **kwargs):
        super(WindowedAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size
        self.pad_mode = pad_mode

    def forward(self, x, H, W):
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size

        qkv = self.qkv(x)  # [B, N, C]
        qkv = qkv.transpose(1, 2).reshape(B, C * 3, H, W)  # [B, C, H, W]
        qkv = F.pad(qkv, [0, W_ - W, 0, H_ - H], mode=self.pad_mode)

        qkv = F.unfold(
            qkv,
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size))
        B, C_kw_kw, L = qkv.shape  # L - the num of windows
        qkv = qkv.reshape(B, C * 3, N_, L).permute(0, 3, 2, 1)  # [B, L, N_, C]
        qkv = qkv.reshape(B, L, N_, 3, self.num_heads,
                          C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv.unbind(0)
        # make torchscript happy (cannot use tensor as tuple)

        # q,k,v [B, L, num_head, N_, C/num_head]
        # [B, L, num_head, N_, N_]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if self.mask:
        #     attn = attn * mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        # attn @ v = [B, L, num_head, N_, C/num_head]
        x = (attn @ v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L)

        x = F.fold(
            x,
            output_size=(H_, W_),
            kernel_size=(self.window_size, self.window_size),
            stride=(self.window_size, self.window_size))  # [B, C, H_, W_]
        x = x[:, :, :H, :W].reshape(B, C, N).transpose(-1, -2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 **kwargs):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        num_fcs=2,
        qkv_bias=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        batch_first=True,
        attn_cfg=dict(),  # window_size
        ffn_cfg=dict(add_identity=False),
        windowed=False,
        layer_scale=True,
        with_cp=False,
    ):
        super(Block, self).__init__()
        feedforward_channels = int(embed_dims * mlp_ratio)
        self.windowed = windowed

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0 else nn.Identity()
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = nn.Parameter(
                torch.ones((embed_dims)), requires_grad=True)
            self.gamma2 = nn.Parameter(
                torch.ones((embed_dims)), requires_grad=True)

    def build_attn(self, attn_cfg):
        if self.windowed:
            self.attn = WindowedAttention(**attn_cfg)
        else:
            self.attn = Attention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, H, W):

        def _inner_forward(x):
            if self.layer_scale:
                x = x + self.drop_path(
                    self.gamma1 * self.attn(self.norm1(x), H, W))
                x = x + self.gamma2 * self.ffn(self.norm2(x))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W))
                x = x + self.ffn(self.norm2(x))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class VisionTransformer(BaseModule):
    """Implements Vision Transformer with window attention and layer scale.

    Args:
        img_size (int | tuple): Input image size for the pretrained model.
            Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        num_classes (int): Number of classification classes. Default: 1000.
        embed_dims (int): Embedding dimension. Default: 768.
        depth (int): Depth of transformer. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.
        drop_path_rate (float): The stochastic depth rate. Default 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        layer_scale (bool): Enable layer scale. Default: True.
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        act_cfg (dict): The activation config for FFNs and CFFNs.
            Default: dict(type='GELU').
        window_attn (bool or list[bool]): Whether to use window attn in
            each ViT layer. Default: False.
        window_size (int or list[int]): The window size of window attn.
            Default: 14.
        pretrained (str, optional): Model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dims=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 layer_scale=True,
                 patch_norm=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 window_attn=False,
                 window_size=14,
                 pretrained=None,
                 convert_weights=True,
                 with_cp=False,
                 init_cfg=None):
        super(VisionTransformer, self).__init__()
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.convert_weights = convert_weights
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dims = embed_dims
        self.num_tokens = 1
        self.norm_cfg = norm_cfg
        self.pretrain_size = img_size
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate

        window_attn = [window_attn] * depth if not isinstance(
            window_attn, list) else window_attn
        window_size = [window_size] * depth if not isinstance(
            window_size, list) else window_size
        logging.info('window attention:', window_attn)
        logging.info('window size:', window_size)
        logging.info('layer scale:', layer_scale)

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )
        num_patches = (img_size[0] // patch_size) * \
                      (img_size[1] // patch_size)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dims))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                num_fcs=num_fcs,
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                attn_cfg=dict(window_size=window_size[i], pad_mode='constant'),
                ffn_cfg=dict(add_identity=False),
                windowed=window_attn[i],
                layer_scale=layer_scale,
                with_cp=with_cp) for i in range(depth)
        ])
        self.init_weights()

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            print('ckpt keys:', checkpoint.keys())
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # get state_dict from checkpoint
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']  # for classification weights
            else:
                state_dict = checkpoint

            if self.convert_weights:
                # supported loading weight from original repo,
                state_dict = vit_converter(state_dict)

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {
                    k[7:]: v
                    for k, v in checkpoint['state_dict'].items()
                }

            if hasattr(self, 'module'):
                load_state_dict(
                    self.module, state_dict, strict=False, logger=logger)
            else:
                load_state_dict(self, state_dict, strict=False, logger=logger)
        elif self.init_cfg is not None:
            super(VisionTransformer, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            # trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


@BACKBONES.register_module()
class ViTBaseline(VisionTransformer):
    """ViT backbone with multi-scale feature map.

    This backbone is the implementation of `Benchmarking detection transfer
    learning with vision transformers <https://arxiv.org/abs/2111.11429>`_.
    """

    def __init__(self, pretrain_size=224, **kwargs):
        super(ViTBaseline, self).__init__(**kwargs)
        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.flags = [
            i for i in range(-1, self.num_block, self.num_block // 4)
        ][1:]

        embed_dims = self.embed_dims
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.norm4 = nn.LayerNorm(embed_dims)

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2),
            nn.GroupNorm(32, embed_dims),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2)
        ])
        self.up2 = nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)

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
        pos_embed = F.interpolate(
            pos_embed, size=(H, W), mode='bicubic',
            align_corners=False).reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward_features(self, x):
        outs = []
        x, (H, W) = self.patch_embed(x)
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        for index, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            if index in self.flags:
                outs.append(x)
        return outs, H, W

    def forward(self, x):
        outs, H, W = self.forward_features(x)
        f1, f2, f3, f4 = outs
        bs, n, dim = f1.shape
        f1 = self.norm1(f1).transpose(1, 2).reshape(bs, dim, H, W)
        f2 = self.norm2(f2).transpose(1, 2).reshape(bs, dim, H, W)
        f3 = self.norm3(f3).transpose(1, 2).reshape(bs, dim, H, W)
        f4 = self.norm4(f4).transpose(1, 2).reshape(bs, dim, H, W)

        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()

        return [f1, f2, f3, f4]
