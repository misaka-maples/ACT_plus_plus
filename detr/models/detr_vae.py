# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython

e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1.md

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ DETR 模块，带有变分自编码器 (VAE) 架构，用于目标检测 """

    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim):
        """
        初始化模型。
        参数：
            backbones: 用于特征提取的骨干网络模块列表，每个相机对应一个模块。
            transformer: Transformer 模块，用于特征处理和基于查询的检测。
            encoder: （可选）用于潜在空间建模的编码器（例如 CVAE 中的编码器）。
            state_dim: 机器人状态的维度。
            num_queries: 最大的目标查询数量，即检测槽位。最大检测对象数。
            camera_names: 与骨干网络对应的相机名称列表。
            vq: 是否在潜在空间中使用向量量化 (Vector Quantization)。
            vq_class: VQ 中类别的数量。
            vq_dim: 每个 VQ 向量的维度。
            action_dim: 动作空间的维度。
        """
        super().__init__()
        self.num_queries = num_queries  # 查询数量（检测目标的最大数量）
        self.camera_names = camera_names  # 相机名称列表
        self.transformer = transformer  # Transformer 模块
        self.encoder = encoder  # 编码器模块（CVAE 中的 encoder）
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim  # VQ 参数
        self.state_dim, self.action_dim = state_dim, action_dim  # 状态维度和动作维度

        # Transformer 的隐藏维度（d_model）
        hidden_dim = transformer.d_model

        # 动作预测头：将 Transformer 的输出映射到动作空间
        self.action_head = nn.Linear(hidden_dim, action_dim)
        # 用于判断 padding 的预测头
        self.is_pad_head = nn.Linear(hidden_dim, 1)

        # 查询嵌入，用于目标检测的查询
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 如果有骨干网络（用于处理图像输入）
        if backbones is not None:
            # 将骨干网络的输出通道数投影到 Transformer 的隐藏维度
            # self.input_proj = nn.Conv2d(backbones[0].num_channels, 384, kernel_size=1)
            # print(backbones[0].num_channels)
            self.input_proj = nn.ModuleList([
                nn.Conv2d(512, 512, kernel_size=1) for backbone in backbones
            ])
            # 在 detr_vae.py 的模型初始化部分
            # self.input_proj = nn.ModuleList([
            #     nn.Conv2d(512, 384, kernel_size=1) for _ in range(num_cameras)
            # ])

            self.backbones = nn.ModuleList(backbones)  # 保存多个骨干网络
            # 将机器人状态映射到 Transformer 的隐藏维度
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # 如果没有骨干网络，则可能是基于状态的模型
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)  # 环境状态映射
            self.pos = torch.nn.Embedding(2, hidden_dim)  # 位置编码（仅用于状态）
            self.backbones = None
        for i, backbone in enumerate(self.backbones):
            print(f"Backbone {i}: Expected output channels = {backbone.num_channels}")

        # 编码器额外参数
        self.latent_dim = 32  # 潜在空间的维度（可调节）
        self.cls_embed = nn.Embedding(1, hidden_dim)  # 分类令牌嵌入
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)  # 将动作嵌入到 Transformer 的隐藏维度
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # 将机器人状态（关节位置）嵌入到隐藏维度

        # 打印 VQ 使用信息
        # print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            # 如果使用 VQ，将隐藏状态投影到 VQ 空间
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            # 如果不使用 VQ，将隐藏状态投影到潜在均值和方差
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)

        # 注册位置编码表（包括 [CLS] 和查询嵌入）
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))

        # 解码器额外参数
        if self.vq:
            # 如果使用 VQ，将 VQ 输出映射回隐藏维度
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            # 如果不使用 VQ，将潜在样本映射回隐藏维度
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)

        # 额外位置嵌入，用于机器人状态和潜在变量
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        """
        编码阶段：处理机器人状态、动作序列并生成潜在空间的输入。
        参数：
            qpos: 机器人状态（关节位置），形状为 (batch_size, state_dim)。
            actions: 动作序列，形状为 (batch_size, seq_len, action_dim)。
            is_pad: 动作序列的 padding mask，形状为 (batch_size, seq_len)。
            vq_sample: 可选的 VQ 样本，用于推理阶段。
        """
        bs, _ = qpos.shape  # 获取 batch 大小
        if self.encoder is None:  # 如果没有编码器
            # 生成一个全零的潜在变量样本
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # CVAE 编码器处理
            is_training = actions is not None  # 判断是训练模式还是推理模式
            if is_training:
                # print(f"is_training_{qpos.shape}")
                # 将动作序列投影到嵌入维度，并与分类令牌和状态嵌入拼接
                action_embed = self.encoder_action_proj(actions)  # 动作序列嵌入
                qpos_embed = self.encoder_joint_proj(qpos)  # 关节位置嵌入
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # 添加序列维度
                cls_embed = self.cls_embed.weight  # 分类令牌嵌入
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # 扩展为批次大小
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # 拼接输入
                encoder_input = encoder_input.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, hidden_dim)

                # Mask 的处理，确保分类令牌和状态不被 mask
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)

                # 获取位置编码
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)

                # 通过编码器
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0]  # 只取分类令牌的输出

                # 获取潜在空间信息
                latent_info = self.latent_proj(encoder_output)

                if self.vq:
                    # 如果使用 VQ，执行向量量化
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    # 不使用 VQ，生成均值和方差
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                # 推理模式，仅使用潜在样本
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        前向传播：从输入生成动作预测和 padding 标志。
        参数：
            qpos: 机器人状态，形状为 (batch_size, state_dim)。
            image: 图像输入，形状为 (batch_size, num_cameras, channels, height, width)。
            env_state: 环境状态（如果存在）。
            actions: 动作序列，形状为 (batch_size, seq_len, action_dim)。
            is_pad: 动作序列的 padding 标志。
            vq_sample: VQ 的潜在样本（推理时可用）。
        """
        # 编码阶段，获取潜在变量输入
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # 解码阶段
        if self.backbones is not None:  # 如果使用了骨干网络处理图像
            # 处理每个相机的图像特征和位置编码
            all_cam_features = []
            all_cam_pos = []

            for cam_id, cam_name in enumerate(self.camera_names):
                # features, pos = self.backbones[cam_id](image[:, cam_id])
                features, pos = self.backbones[cam_id](image[:, cam_id])


                features = features[0]  # 使用最后一层特征
                # print(features.shape)
                pos = pos[0]
                features_proj = self.input_proj[cam_id](features)  # 投影到统一通道数
                # all_cam_features.append(self.input_proj(features))
                all_cam_features.append(features_proj)
                all_cam_pos.append(pos)

            # 处理机器人状态特征
            proprio_input = self.input_proj_robot_state(qpos)
            # 将所有相机特征合并
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            # 通过 Transformer
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:  # 如果没有骨干网络，仅基于状态进行推理
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # 序列长度为 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]

        # 动作预测和 padding 标志预测
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)  # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            pos = pos[0]  # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, qpos], axis=1)  # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    # 从传入的参数中获取 encoder 的超参数配置
    d_model = args.hidden_dim  # 隐藏层的维度（例如256）
    dropout = args.dropout  # dropout 概率（例如0.1.md，用于防止过拟合）
    nhead = args.nheads  # Transformer 中的多头注意力机制中的头数（例如8）
    dim_feedforward = args.dim_feedforward  # 前馈网络的维度（例如2048）
    num_encoder_layers = args.enc_layers  # 编码器的层数（例如4层）
    normalize_before = args.pre_norm  # 是否在层归一化之前执行（True/False）
    activation = "relu"  # 激活函数（默认是 ReLU）

    # 创建单个 TransformerEncoderLayer，这一层定义了多头自注意力和前馈网络
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)

    # 如果设置了 `normalize_before`，则为编码器添加层归一化，否则不添加
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

    # 使用多层 Transformer 编码器，层数由 `num_encoder_layers` 指定
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    # 返回构建的编码器
    return encoder


def build(args):
    # 设置状态维度为14。 TODO: 后期可能需要调整或动态计算
    state_dim = 14  # TODO hardcode

    # 初始化一个列表来存储每个相机的 backbone（特征提取网络）
    backbones = []
    for _ in args.camera_names:  # 针对每个相机名称
        # 为每个相机构建一个 backbone（特征提取网络）
        backbone = build_backbone(args)
        backbones.append(backbone)

    # 构建 transformer 模型
    transformer = build_transformer(args)

    # 根据 `args.no_encoder` 的值来决定是否构建编码器（encoder）
    if args.no_encoder:
        encoder = None  # 如果不需要编码器，设置为 None
    else:
        # 如果需要编码器，调用 `build_encoder` 来构建
        encoder = build_encoder(args)

    # 创建 DETRVAE 模型，并将所有组件传入
    model = DETRVAE(
        backbones,  # 每个相机的 backbone
        transformer,  # transformer 模型
        encoder,  # 编码器，可能为 None
        state_dim=args.state_dim,  # 状态维度
        num_queries=args.num_queries,  # 查询数量
        camera_names=args.camera_names,  # 相机名称列表
        vq=args.vq,  # 是否使用矢量量化（vector quantization）# 是否使用矢量量化（vector quantization）
        vq_class=args.vq_class,  # 矢量量化的类别数
        vq_dim=args.vq_dim,  # 矢量量化的维度
        action_dim=args.action_dim,  # 动作维度
    )

    # 计算模型的参数总数
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 打印模型的参数数量（以百万为单位）
    # print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    # 返回构建好的模型
    return model


def build_cnnmlp(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model

