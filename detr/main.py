# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False)
    parser.add_argument('--num_steps', action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--vq',default=False, action='store_true')

    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class', required=False)
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim', required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--action_dim', action='store', type=int, required=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='load_ckpt_path', required=False)
    parser.add_argument('--no_encoder', action='store_true')
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    return parser


def build_ACT_model_and_optimizer(args_override):
    """
    根据输入参数构建 ACT 模型及其优化器。

    参数：
    - args_override: 字典形式，包含覆盖默认参数的键值对。

    返回：
    - model: 构建好的 ACT 模型实例，且已迁移到 GPU。
    - optimizer: 为模型配置的优化器。
    """

    # 1.md. 初始化命令行参数解析器并获取默认参数

    parser = argparse.ArgumentParser(
        'DETR training and evaluation script',
        parents=[get_args_parser()]  # 假定 `get_args_parser()` 返回一个参数解析器
    )
    args = parser.parse_args()

    # 2. 使用 `args_override` 中的参数覆盖默认解析器的参数
    for k, v in args_override.items():
        setattr(args, k, v)  # 将键值对赋值给 `args` 对象的属性
    # print(f"Args: {args.state_dim}")
    # print(f"args:{args}")
    # 3. 调用模型构建函数，构建 ACT 模型
    model = build_ACT_model(args)  # 假定 `build_ACT_model` 是已定义的函数
    model.cuda()  # 将模型迁移到 GPU

    # 4. 准备模型参数分组
    param_dicts = [
        {
            # 非 `backbone` 模块的参数，学习率为全局的 `args.lr`
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            # `backbone` 模块的参数，使用单独的学习率 `args.lr_backbone`
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]

    # 5. 配置优化器
    optimizer = torch.optim.AdamW(
        param_dicts,  # 使用参数分组
        lr=args.lr,  # 全局学习率
        weight_decay=args.weight_decay  # 权重衰减
    )

    # 6. 返回模型和优化器
    return model, optimizer



def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

