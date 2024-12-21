# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import math
from collections import OrderedDict
from torchvision.models import get_model_weights
# 获取对应模型的默认权重

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from detr.util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            if isinstance(backbone, DINOv2BackBone):
                return_layers = {"0": "0"}  # 根据 DINOv2BackBone 的输出定义
            else:
                return_layers = {"layer1": "0", "layer2": "1.md", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        weights = get_model_weights(name).DEFAULT
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=weights, norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 384 if name in ('resnet18', 'resnet34','dino_v2') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

def pad_image(image, patch_size=14):
    """
    填充图像，使其高度和宽度成为补丁大小的整数倍。

    参数:
        image (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)。
        patch_size (int): 每个补丁的大小。

    返回:
        torch.Tensor: 填充后的图像张量。
    """
    _, _, H, W = image.shape
    pad_H = (patch_size - H % patch_size) % patch_size
    pad_W = (patch_size - W % patch_size) % patch_size
    padded_image = F.pad(image, (0, pad_W, 0, pad_H), mode='constant', value=0)
    return padded_image

class DINOv2BackBone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14',pretrained=True)
        self.body.eval()
        self.num_channels = 384

    # @torch.no_grad()   原程序-DINOv2
    # def forward(self, tensor):
    #     # tensor = pad_image(tensor)
    #     xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
    #     print(f"features.shape: {xs.shape}")
    #     od = OrderedDict()
    #     od["0"] = xs.reshape(xs.shape[0], 22, 16, 384).permute(0, 3, 2, 1)
    #     return od


    @torch.no_grad()
    def forward(self, tensor):
        # 对图像进行填充，确保尺寸为补丁大小的整数倍
        tensor = pad_image(tensor, patch_size=14)

        # 获取特征
        features = self.body.forward_features(tensor)["x_norm_patchtokens"]
        print(f"features.shape: {features.shape}")  # 例如 [16, 1201, 384]

        batch_size, seq_len, dim = features.shape
        num_patches = seq_len - 1  # 假设第一个 token 是 class token

        # 根据填充后的图像计算 grid_size
        grid_size_h = tensor.shape[2] // 14  # 填充后的高度
        grid_size_w = tensor.shape[3] // 14  # 填充后的宽度
        expected_num_patches = grid_size_h * grid_size_w  # 例如 34 * 45 = 1530

        if num_patches != expected_num_patches:
            print(f"num_patches {num_patches} 不符合预期 {expected_num_patches}。正在调整。")
            # 剪裁多余的补丁
            if num_patches > expected_num_patches:
                features = features[:, :expected_num_patches + 1, :]  # 保留 class token
                num_patches = expected_num_patches
            elif num_patches < expected_num_patches:
                # 用零填充缺失的补丁
                pad_size = expected_num_patches - num_patches
                padding = torch.zeros(batch_size, pad_size, dim, device=tensor.device)
                features = torch.cat([features, padding], dim=1)
                num_patches = expected_num_patches

        patches = features[:, 1:, :].reshape(batch_size, grid_size_h, grid_size_w, dim)
        patches = patches.permute(0, 3, 1, 2)  # [batch_size, dim, grid_size_h, grid_size_w]
        od = OrderedDict()
        od["0"] = patches
        return od

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.backbone == 'dino_v2':
        backbone = DINOv2BackBone()
    else:
        assert args.backbone in ['resnet18', 'resnet34'], f"Unsupported backbone: {args.backbone}"
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
 