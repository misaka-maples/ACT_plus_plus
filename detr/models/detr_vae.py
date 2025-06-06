# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer, build_transformer_decoder
import time

import numpy as np

import IPython
e = IPython.embed

import matplotlib.pyplot as plt

# mask: shape [B, 1, H, W]
def visualize_mask(mask, index=0):
    """可视化第 index 个样本的 mask"""
    m = mask[index, 0].detach().cpu().numpy()  # shape: [H, W]
    plt.imshow(m, cmap='gray')
    plt.title(f"Mask #{index}")
    plt.colorbar()
    plt.show()

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE_Decoder(nn.Module):
    """ This is the decoder only transformer """
    def __init__(self, backbones, transformer_decoder, state_dim, num_queries, camera_names, action_dim,
                 feature_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.cam_num = len(camera_names)
        self.transformer_decoder = transformer_decoder
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer_decoder.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.proprio_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
        # encoder extra parameters
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq
        self.additional_pos_embed = nn.Embedding(1, hidden_dim) # learned position embedding for proprio and latent
        self.feature_loss = feature_loss
          
    def forward(self, qpos, image):
        if self.feature_loss:
            # bs,_,_,h,w = image.shape
            image_future = image[:,len(self.camera_names):].clone()
            image = image[:,:len(self.camera_names)].clone()

        if len(self.backbones)>1:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_features_future = []
            all_cam_pos = []
            for cam_id, _ in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
        else:
            all_cam_features = []
            all_cam_pos = []
            if self.feature_loss and self.training:
                all_cam_features_future = []
                bs,_,_,h,w = image.shape
                image_total = torch.cat([image, image_future], axis=0) #cat along the batch dimension
                bs_t,_,_,h_t,w_t = image_total.shape
                features, pos = self.backbones[0](image_total.reshape([-1,3,image_total.shape[-2],image_total.shape[-1]]))
                project_feature = self.input_proj(features[0])
                project_feature = project_feature.reshape([bs_t, self.cam_num,project_feature.shape[1],project_feature.shape[2],project_feature.shape[3]])
                for i in range(self.cam_num):
                    all_cam_features.append(project_feature[:bs,i,:])
                    all_cam_pos.append(pos[0])
                    all_cam_features_future.append(project_feature[bs:,i,:])
            else:
                bs,_,_,h,w = image.shape
                features, pos = self.backbones[0](image.reshape([-1,3,image.shape[-2],image.shape[-1]]))
                project_feature = self.input_proj(features[0]) 
                project_feature = project_feature.reshape([bs, self.cam_num,project_feature.shape[1],project_feature.shape[2],project_feature.shape[3]])
                for i in range(self.cam_num):
                    all_cam_features.append(project_feature[:,i,:])
                    all_cam_pos.append(pos[0])
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos) #B, 512
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3) #B, 512,12,26
        pos = torch.cat(all_cam_pos, axis=3) #B, 512,12,26
        hs = self.transformer_decoder(src, self.query_embed.weight, proprio_input=proprio_input, pos_embed=pos, additional_pos_embed=self.additional_pos_embed.weight) #B, chunk_size, 512
        # a_hat = self.action_head(hs) #B, chunk_size, action_dim
        hs_action = hs[:,-1*self.num_queries:,:].clone() #B, action_dim, 512
        hs_img = hs[:,1:-1*self.num_queries,:].clone() #B, image_feature_dim, 512 #final image feature
        hs_proprio = hs[:,[0],:].clone() #B, proprio_feature_dim, 512
        a_hat = self.action_head(hs_action)
        a_proprio = self.proprio_head(hs_proprio) #proprio head
        if self.feature_loss and self.training:
            # proprioception features
            src_future = torch.cat(all_cam_features_future, axis=3) #B, 512,12,26
            src_future = src_future.flatten(2).permute(2, 0, 1).transpose(1, 0) # B, 12*26, 512
            hs_img = {'hs_img': hs_img, 'src_future': src_future}
            
        return a_hat, a_proprio, hs_img
    
class RegionEnhancer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask  # 可选，如果你有 mask
        return self.relu(self.conv(x))
def visualize_feature_with_mask(feature, mask, index=0):
    """
    可视化特征图（取平均），并叠加 mask
    feature: [B, C, H, W]
    mask: [B, 1, H, W]
    """
    fmap = feature[index].mean(dim=0).detach().cpu().numpy()
    msk = mask[index, 0].detach().cpu().numpy()
    plt.imshow(fmap, cmap='viridis')
    plt.imshow(msk, cmap='Reds', alpha=0.4)
    plt.title(f"Feature + Mask #{index}")
    plt.axis("off")
    plt.show()

def resize_mask_to_image(mask, target_size=(480, 640), mode='nearest'):
    """
    将 mask resize 到目标图像大小（默认 480×640）
    
    Args:
        mask: Tensor，形状 [B, 1, H, W]（与 features 对应）
        target_size: tuple，(H_img, W_img)
        mode: 上采样方式，推荐用 'nearest' 或 'bilinear'
    
    Returns:
        resized_mask: Tensor [B, 1, H_img, W_img]
    """
    resized_mask = F.interpolate(mask, size=target_size, mode=mode)
    return resized_mask
class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim,features_region_enhancer):
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
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.features_region_enhancer = features_region_enhancer
        if features_region_enhancer:
            self.region_enhancer = RegionEnhancer(in_channels=512)
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent


    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar
    def make_center_mask(self,features, mask_size=(15, 15), center=None):
        """
        构造一个二值mask，中心在center处，大小为mask_size。
        
        Args:
            features: Tensor, shape [B, C, H, W]
            mask_size: tuple(int, int), 形如 (crop_h, crop_w)
            center: tuple(int, int) or None，形如 (center_h, center_w)
                    若为 None，则默认居中
            
        Returns:
            mask: Tensor, shape [B, 1, H, W]
        """
        B, _, H, W = features.shape
        crop_h, crop_w = mask_size

        if center is None:
            center_h, center_w = H // 2, W // 2
        else:
            center_h, center_w = center

        start_h = max(center_h - crop_h // 2, 0)
        start_w = max(center_w - crop_w // 2, 0)
        end_h = min(start_h + crop_h, H)
        end_w = min(start_w + crop_w, W)

        # 修正 start_h/w 以确保 mask 是指定大小（处理边缘越界）
        start_h = end_h - crop_h
        start_w = end_w - crop_w

        mask = torch.zeros((B, 1, H, W), dtype=features.dtype, device=features.device)
        mask[:, :, start_h:end_h, start_w:end_w] = 1.0
        return mask

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # cvae decoder
        # t = time.time()
        if self.backbones is not None:
            if len(self.backbones)>1:
                # Image observation features and position embeddings
                all_cam_features = []
                all_cam_pos = []
                for cam_id, cam_name in enumerate(self.camera_names):
                    features, pos = self.backbones[cam_id](image[:, cam_id])#torch.Size([8, 512, 15, 20])torch.Size([1, 512, 15, 20])
                    features = features[0] # take the last layer feature
                    pos = pos[0]
                    if self.features_region_enhancer:
                        mask = self.make_center_mask(features,(4,2),(15,10))
                        resized_mask = resize_mask_to_image(mask, target_size=(480, 640))

                        # 可视化
                        # visualize_mask(resized_mask)  # 上面已经定义好的函数
                        # visualize_feature_with_mask(features, mask)
                        features = self.region_enhancer(features,mask)  # 增强前特征 shape: [B, C, H, W]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)
            else:
                all_cam_features = []
                all_cam_pos = []
                bs,_,_,h,w = image.shape
                features, pos = self.backbones[0](image.reshape([-1,3,image.shape[-2],image.shape[-1]]))
                if self.features_region_enhancer:
                    mask = self.make_center_mask(features)
                    enhanced_feature = self.region_enhancer(features[0],mask)  # 增强前特征 shape: [B*2, C, H, W]
                    project_feature = self.input_proj(enhanced_feature)
                else:
                    project_feature = self.input_proj(features[0]) 
                 
                project_feature = project_feature.reshape([bs, 2,project_feature.shape[1],project_feature.shape[2],project_feature.shape[3]])
                all_cam_features.append(project_feature[:,0,:])
                all_cam_features.append(project_feature[:,1,:])
                all_cam_pos.append(pos[0])
                all_cam_pos.append(pos[0])
            # print(f'backbone time: {time.time()-t}')
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
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
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
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
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
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
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    if args.same_backbones:
        backbone = build_backbone(args)
        backbones = [backbone]
    else:
        for _ in args.camera_names:
            backbone = build_backbone(args)
            backbones.append(backbone)
        
    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    if args.model_type=="ACT":
        transformer = build_transformer(args)
        model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim= args.state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            vq=args.vq,
            vq_class=args.vq_class,
            vq_dim=args.vq_dim,
            action_dim=args.action_dim,
            features_region_enhancer=args.features_region_enhancer,
        )
    elif args.model_type=="HIT":
        transformer_decoder = build_transformer_decoder(args)

        model = DETRVAE_Decoder(
            backbones,
            transformer_decoder,
            state_dim= args.state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            action_dim=args.action_dim,
            feature_loss= args.feature_loss if hasattr(args, 'feature_loss') else False,
        )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

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
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

