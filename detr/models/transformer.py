# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import IPython
e = IPython.embed


class Transformer(nn.Module):
    """
    Transformer 模型，包括编码器（Encoder）和解码器（Decoder）。
    适用于序列到序列建模任务，如机器翻译、图像处理等。
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.3,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        """
        初始化 Transformer 模型。

        参数：
        - d_model: 输入特征维度（编码器和解码器共享）。
        - nhead: 多头注意力机制中的注意力头数。
        - num_encoder_layers: 编码器的层数。
        - num_decoder_layers: 解码器的层数。
        - dim_feedforward: 前馈网络的隐藏层维度。
        - dropout: 丢弃率。
        - activation: 前馈网络的激活函数（默认为 ReLU）。
        - normalize_before: 是否在多头注意力和前馈网络之前进行归一化。
        - return_intermediate_dec: 是否返回解码器中间层的结果。
        """
        super().__init__()

        # 定义编码器
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 定义解码器
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # 初始化权重参数
        self._reset_parameters()

        # 保存模型参数
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """
        初始化模型参数，使用 Xavier 均匀分布。
        """
        for p in self.parameters():
            if p.dim() > 1:  # 仅初始化权重参数，不处理偏置
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None):
        """
        Transformer 的前向传播。

        参数：
        - src: 输入特征，形状为 (N, C, H, W) 或 (N, HW, C)。
        - mask: 填充掩码，形状为 (N, H*W)。
        - query_embed: 查询嵌入（query embedding），通常用于解码器的初始目标。
        - pos_embed: 位置嵌入（position embedding）。
        - latent_input: 额外的潜在输入，形状为 (N, latent_dim)。
        - proprio_input: 自体输入（proprioceptive input），形状为 (N, proprio_dim)。
        - additional_pos_embed: 额外的位置嵌入，形状为 (seq_len, d_model)。

        返回：
        - hs: 解码器输出的特征序列，形状为 (num_layers, N, num_queries, d_model)。
        """
        # 如果输入具有高度和宽度（如图像特征图）
        # print(f"src: {src}")
        if len(src.shape) == 4:  # 输入形状为 (N, C, H, W)
            # 展平特征图：将 NxCxHxW 转换为 HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)  # 形状变为 (HW, N, C)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)  # 位置嵌入重复以匹配批次
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # 查询嵌入扩展批次维度

            # 处理额外位置嵌入和潜在输入
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)  # (seq, N, dim)
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)  # 拼接额外位置嵌入

            # 将潜在输入和自体输入拼接到源输入之前
            addition_input = torch.stack([latent_input, proprio_input], axis=0)  # (2, N, dim)
            src = torch.cat([addition_input, src], axis=0)  # (HW+2, N, C)
        else:  # 如果输入没有 H 和 W，例如序列输入
            assert len(src.shape) == 3  # 输入形状为 (N, HW, C)
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)  # 转换为 (HW, N, C)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)  # 重复位置嵌入以匹配批次
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # 重复查询嵌入以匹配批次

        # 解码器的初始目标张量（全为零）
        tgt = torch.zeros_like(query_embed)  # (num_queries, N, d_model)

        # 编码器：处理输入特征
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (HW+2, N, d_model)

        # 解码器：生成目标特征
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)  # (num_layers, num_queries, N, d_model)

        # 调整输出形状以匹配 (num_layers, N, num_queries, d_model)
        hs = hs.transpose(1, 2)  # 转换为 (num_layers, N, num_queries, d_model)

        return hs  # 返回解码器的输出


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    一个 Transformer 编码器层模块。
    它包含多头自注意力机制和前馈神经网络，支持两种归一化方式：前归一化 (Pre-Norm) 和后归一化 (Post-Norm)。
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        初始化编码器层。

        参数：
        - d_model: 输入和输出的特征维度。
        - nhead: 多头注意力机制中的注意力头数。
        - dim_feedforward: 前馈神经网络的隐藏层维度。
        - dropout: 用于注意力和前馈层的丢弃率。
        - activation: 前馈神经网络中的激活函数，默认为 ReLU。
        - normalize_before: 是否在注意力和前馈层前应用归一化（前归一化）。
        """
        super().__init__()

        # 多头自注意力模块
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数（可以是ReLU或GELU等）
        self.activation = _get_activation_fn(activation)

        # 是否使用前归一化
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        如果提供了位置编码，将其添加到输入张量中。

        参数：
        - tensor: 输入张量。
        - pos: 位置编码张量（可选）。

        返回：
        - 添加位置编码后的张量。
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        后归一化 (Post-Norm) 的前向传播流程。

        参数：
        - src: 输入序列特征张量，形状为 (S, N, E)，
          其中 S 是序列长度，N 是批次大小，E 是特征维度。
        - src_mask: 注意力掩码，形状为 (S, S)（可选）。
        - src_key_padding_mask: 填充掩码，形状为 (N, S)（可选）。
        - pos: 位置编码张量，形状为 (S, N, E)（可选）。

        返回：
        - 编码后的输出张量，形状为 (S, N, E)。
        """
        # 多头自注意力：结合位置编码
        # print(f"src shape: {src.shape},pos shape: {pos.shape}")
        q = k = self.with_pos_embed(src, pos)
        # print(f"encoder{q,k}")
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # 残差连接 + Dropout + 归一化
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络 + 残差连接 + Dropout + 归一化
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """
        前归一化 (Pre-Norm) 的前向传播流程。

        参数：
        - src: 输入序列特征张量，形状为 (S, N, E)。
        - src_mask: 注意力掩码，形状为 (S, S)（可选）。
        - src_key_padding_mask: 填充掩码，形状为 (N, S)（可选）。
        - pos: 位置编码张量，形状为 (S, N, E)（可选）。

        返回：
        - 编码后的输出张量，形状为 (S, N, E)。
        """
        # 先进行归一化
        src2 = self.norm1(src)

        # 多头自注意力：结合位置编码
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # 残差连接 + Dropout
        src = src + self.dropout1(src2)

        # 前馈网络 + 归一化
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        前向传播，根据 `normalize_before` 选择使用前归一化还是后归一化。

        参数：
        - src: 输入序列特征张量。
        - src_mask: 注意力掩码（可选）。
        - src_key_padding_mask: 填充掩码（可选）。
        - pos: 位置编码张量（可选）。

        返回：
        - 编码后的输出张量。
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    """
    根据输入参数构建 Transformer 模型。

    参数：
    - args: 包含 Transformer 配置参数的对象，通常是命令行参数解析的结果。

    返回：
    - 一个 Transformer 实例。
    """
    return Transformer(
        d_model=args.hidden_dim,                 # 输入特征维度（Transformer 的特征维度）。
        dropout=args.dropout,                   # 丢弃率，用于防止过拟合。
        nhead=args.nheads,                      # 多头注意力机制中的注意力头数。
        dim_feedforward=args.dim_feedforward,   # 前馈网络的隐藏层维度。
        num_encoder_layers=args.enc_layers,     # 编码器的层数。
        num_decoder_layers=args.dec_layers,     # 解码器的层数。
        normalize_before=args.pre_norm,         # 是否在多头注意力和前馈网络之前进行归一化。
        return_intermediate_dec=True,           # 是否返回解码器中间层的结果。
    )



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
