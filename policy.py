import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import numpy as np
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

from collections import OrderedDict
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
import math
class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        self.camera_names = args_override['camera_names']
        self.observation_horizon = args_override['observation_horizon'] ### TODO TODO TODO DO THIS
        self.action_horizon = args_override['action_horizon'] # apply chunk size
        self.prediction_horizon = args_override['prediction_horizon'] # chunk size
        self.num_inference_timesteps = args_override['num_inference_timesteps']
        self.ema_power = args_override['ema_power']
        self.lr = args_override['lr']
        self.weight_decay = 0

        self.num_kp = 32
        self.feature_dimension = 64
        self.ac_dim = args_override['action_dim'] # 14 + 2
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14 # camera features and proprio

        backbones = []
        pools = []
        linears = []
        for _ in self.camera_names:
            backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            pools.append(SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
        backbones = nn.ModuleList(backbones)
        pools = nn.ModuleList(pools)
        linears = nn.ModuleList(linears)
        
        backbones = replace_bn_with_gn(backbones) # TODO


        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim*self.observation_horizon
        )

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })

        nets = nets.float().cuda()
        ENABLE_EMA = True
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)
        else:
            ema = None
        self.nets = nets
        self.ema = ema

        # setup noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        n_parameters = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_parameters/1e6,))


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]
        if actions is not None: # training time
            nets = self.nets
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)
            
            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss

            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss_dict
        else: # inference time
            To = self.observation_horizon
            Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim
            
            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model
            
            all_features = []
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]
                cam_features = nets['policy']['backbones'][cam_id](cam_image)
                pool_features = nets['policy']['pools'][cam_id](cam_features)
                pool_features = torch.flatten(pool_features, start_dim=1)
                out_features = nets['policy']['linears'][cam_id](pool_features)
                all_features.append(out_features)

            obs_cond = torch.cat(all_features + [qpos], dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action
            
            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])
        print('Loaded model')
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
            status = [status, status_ema]
        return status

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        """
        初始化 ACTPolicy。

        参数：
        - args_override: 字典，包含模型和优化器的构建参数。
        """
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)  # 调用辅助函数构建模型和优化器
        self.model = model  # CVAE 解码器部分
        self.optimizer = optimizer  # 优化器
        self.args_override = args_override
        if args_override['eval'] == False:
            self.kl_weight = args_override['kl_weight']  # KL 散度权重
            self.vq = args_override['vq']  # 是否启用 VQ（Vector Quantization）
            self.qpos_noise_std = args_override['qpos_noise_std']
            self.test_loss = args_override['new_loss']
        # print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        """
        在训练或推理模式下调用模型。

        参数：
        - qpos: 机器人状态向量。
        - image: 图像输入，形状为 (batch, num_cam, channel, height, width)。
        - actions: 动作序列 (batch, seq_len, action_dim)，用于训练。
        - is_pad: 动作序列中填充标志，用于掩码 (batch, seq_len)。
        - vq_sample: 在推理中从 VQ 离散化编码中采样的值。

        返回：
        - 训练模式: 损失字典。
        - 推理模式: 预测动作。
        """
        env_state = None
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # image = normalize(image)  # 归一化图像输入
        patch_h = 16
        patch_w = 22
        if self.args_override['backbone'] == 'dino_v2':
            if actions is not None:  # training time
                transform = transforms.Compose([
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    transforms.RandomPerspective(distortion_scale=0.5),
                    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)),
                    transforms.Resize((patch_h * 14, patch_w * 14)),
                    # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                qpos += (self.qpos_noise_std ** 0.5) * torch.randn_like(qpos)
            else:  # inference time
                transform = transforms.Compose([
                    transforms.Resize((patch_h * 14, patch_w * 14)),
                    # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])

            # 调整输入图像的形状
            batch_size, num_cam, channel, height, width = image.shape
            image = image.view(-1, channel, height, width)  # 合并 batch 和 camera
            image = transform(image)  # 应用变换
            image = image.view(batch_size, num_cam, channel, patch_h * 14, patch_w * 14)  # 恢复形状
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)  # 归一化图像输入


        if actions is not None:  # 训练模式
            actions = actions[:, :self.model.num_queries]  # 裁剪动作序列
            is_pad = is_pad[:, :self.model.num_queries]  # 裁剪填充标志

            loss_dict = dict()
            # 前向传播，获取模型输出
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, env_state, actions, is_pad, vq_sample)

            # KL 散度计算
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            # VQ 误差
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')

            # L1 损失（动作预测误差）
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')  # 每个元素的误差
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()  # 只考虑非填充部分
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            # 计算动作幅度奖励
            # action_amplitude_reward = torch.mean(torch.abs(a_hat))
            # loss_dict['amplitude_reward'] = -self.amplitude_weight * action_amplitude_reward
            if self.test_loss:
                # 调整 L1 损失
                # weight = torch.abs(torch.abs(a_hat)-torch.abs(actions))
                all_l1 = F.l1_loss(actions, a_hat, reduction='none')
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()

                loss_dict['l1'] = l1
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
                joint_limits = {
                    'min': [-1.5, 0, 0, 0.5, 0.25, -1.2, 0, 0, 0],
                    # Minimum joint values for each joint
                    'max': [-0.3, 1.5, 1.5, 1.8, 1.25, 0, 3.2, 0, 0]  # Maximum joint values for each joint
                }

                loss_fn = JointControlLoss(joint_limits=joint_limits, smoothness_weight=0.5, constraint_weight=2.0, )
                loss_dict['loss']=loss_dict['loss']+loss_fn(actions, a_hat)
            return loss_dict
        else:  # 推理模式
            a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample)  # 采样自先验
            return a_hat

    def configure_optimizers(self):
        """
        返回模型的优化器。
        """
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        """
        执行 VQ 编码，将连续动作表示离散化为 VQ 表示。

        参数：
        - qpos: 机器人状态向量。
        - actions: 动作序列。
        - is_pad: 填充标志。

        返回：
        - binaries: 离散化后的 VQ 表示。
        """
        actions = actions[:, :self.model.num_queries]
        is_pad = is_pad[:, :self.model.num_queries]
        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)
        return binaries

    def serialize(self):
        """
        序列化模型的状态字典。

        返回：
        - state_dict: 模型的状态字典。
        """
        return self.state_dict()

    def deserialize(self, model_dict):
        """
        加载给定的模型状态字典。

        参数：
        - model_dict: 包含模型权重的状态字典。

        返回：
        - 加载状态的结果。
        """
        return self.load_state_dict(model_dict)


class JointControlLoss(nn.Module):
    def __init__(self, joint_limits=None, smoothness_weight=1.0, constraint_weight=1.0):
        """
        :param joint_limits: Dictionary with 'min' and 'max' joint limits, e.g., {'min': q_min, 'max': q_max}.
        :param smoothness_weight: Weight for smoothness loss.
        :param constraint_weight: Weight for joint limit constraint loss.
        """
        super(JointControlLoss, self).__init__()
        self.joint_limits = joint_limits
        self.smoothness_weight = smoothness_weight
        self.constraint_weight = constraint_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_joints, target_joints):
        """
        :param pred_joints: Predicted joint values, shape (batch_size, time_steps, num_joints).
        :param target_joints: Target joint values, shape (batch_size, time_steps, num_joints).
        :return: Total loss value (on cuda device).
        """
        # Ensure inputs are tensors and move them to the same device (cuda)
        device = pred_joints.device  # Get the device of pred_joints
        pred_joints = pred_joints.to(device)
        target_joints = target_joints.to(device)

        # 1. Mean Squared Error (MSE) Loss for trajectory tracking
        mse_loss = self.mse_loss(pred_joints, target_joints)

        # 2. Smoothness Loss (L2 norm of differences between consecutive time steps)
        smoothness_loss = 0.0
        if pred_joints.shape[1] > 1:  # Time steps > 1
            smoothness_loss = torch.mean(
                torch.sum((pred_joints[:, 1:] - pred_joints[:, :-1]) ** 2, dim=-1)
            )

        # 3. Joint Constraint Loss
        constraint_loss = 0.0
        if self.joint_limits:
            q_min = torch.tensor(self.joint_limits['min'], dtype=torch.float32).to(device)
            q_max = torch.tensor(self.joint_limits['max'], dtype=torch.float32).to(device)

            lower_violation = torch.clamp(q_min - pred_joints, min=0)  # Violation below minimum
            upper_violation = torch.clamp(pred_joints - q_max, min=0)  # Violation above maximum
            constraint_loss = torch.mean(lower_violation + upper_violation)

        # Combine losses with weights
        total_loss = (
            mse_loss
            + self.smoothness_weight * smoothness_loss
            + self.constraint_weight * constraint_loss
        )
        # print(total_loss)
        return total_loss


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ImitationNetwork(nn.Module):
    def __init__(self, image_feature_dim=128, state_dim=7, action_dim=7):
        super(ImitationNetwork, self).__init__()
        # 图像特征提取
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 112, 112]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 56, 56]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 120 * 160, image_feature_dim),
            nn.ReLU()
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(image_feature_dim+state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, image, state):
        img_feat = self.cnn(image)
        combined = torch.cat([img_feat, state], dim=1)  # 拼接图像特征和状态
        action = self.fc(combined)
        return action