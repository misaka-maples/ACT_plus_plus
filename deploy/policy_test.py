import os,sys
from pathlib import Path
def get_env():
    root_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_path))
    # print(root_path)

    detr_path = os.path.join(root_path, 'detr')
    sys.path.append(str(detr_path))
    # print(detr_path)

    robomimic_path = os.path.join(root_path, 'robomimic', 'robomimic')
    sys.path.append(str(robomimic_path))
get_env()

from imitate_episodes import make_policy, set_seed
from constants import SIM_TASK_CONFIGS
from detr.models.latent_model import Latent_Model_Transformer
import os
import torch
import pickle
import numpy as np
import argparse
from einops import rearrange
import time
import torch
import os
import pickle
import numpy as np
import time


class ActionGenerator:
    def __init__(self, args):
        """
        初始化 ActionGenerator 类，加载配置和策略模型。

        参数：
        - args: 配置字典，包含必要的参数。
        """
        # 设置配置
        self.policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': ['top', 'right_wrist'],
            'vq': args['use_vq'],
            'vq_class': args['vq_class'],
            'vq_dim': args['vq_dim'],
            'action_dim': 9,
            'no_encoder': args['no_encoder'],
            'state_dim': 7,
        }
        self.config = {
            'ckpt_dir': args['ckpt_dir'],
            'policy_class': args['policy_class'],
            'policy_config': self.policy_config,
            'episode_len': 400,
            'temporal_agg': False,
            'state_dim': 7,
        }
        if args['qpos_list'] is None:
            raise "qpos_list in policy test is None"
        else:
            self.qpos_list = args['qpos_list']
        if args['image_dict'] is None:
            raise "image_list in policy test is None"
        else:
            self.image_dict = args['image_dict']
        # 载入策略模型
        self.ckpt_dir = args['ckpt_dir']
        self.policy_class = args['policy_class']
        self.policy = self._load_policy()

    def _load_policy(self):
        """
        加载策略模型。

        返回：
        - policy: 加载后的策略模型。
        """
        ckpt_path = os.path.join(self.ckpt_dir, 'policy_best.ckpt')
        policy = make_policy(self.policy_class, self.policy_config)
        policy.deserialize(torch.load(ckpt_path, weights_only=True))
        policy.cuda()
        policy.eval()
        return policy

    def _get_image(self, ts, image_dict):
        """
        获取图像并进行预处理，包括随机裁剪和调整大小。

        返回：
        - curr_image: 预处理后的图像张量。
        """
        curr_images = []
        for cam_name in self.policy_config['camera_names']:
            # 获取并调整图像维度
            curr_image = image_dict[cam_name]
            curr_image = torch.from_numpy(curr_image).permute(2, 0, 1).float()
            curr_images.append(curr_image)

        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  # 归一化并转为Tensor
        return curr_image

    def _eval_bc_with_external_inputs(self, qpos, image_dict):
        """
        实时生成单步动作。

        参数：
        - qpos: 当前时间步的状态向量（numpy 数组）。
        - image_dict: 当前时间步的图像数据（字典，包含多个相机视角的图像）。

        返回：
        - target_qpos: 当前时间步的目标关节位置。
        - base_action: 当前时间步的底座动作。
        """
        # 加载状态统计数据（只加载一次）
        stats_path = os.path.join(self.ckpt_dir, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        # 定义预处理和后处理函数
        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        # 预处理输入状态
        qpos = pre_process(qpos[0])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        # 获取图像并转换格式
        curr_image = self._get_image(ts=None, image_dict=image_dict)

        # 查询策略模型
        if self.policy_class == "ACT":
            all_actions = self.policy(qpos, curr_image)
            raw_action = all_actions[:, 0]  # 单步数据不涉及时间步索引
        elif self.policy_class == "Diffusion":
            all_actions = self.policy(qpos, curr_image)
            raw_action = all_actions[:, 0]
        elif self.policy_class == "CNNMLP":
            raw_action = self.policy(qpos, curr_image)
        else:
            raise NotImplementedError

        # 后处理动作
        raw_action = raw_action.squeeze(0).detach().cpu().numpy()
        action = post_process(raw_action)

        # 拆分动作为目标关节位置和底座动作
        target_qpos = action[:-2]
        base_action = action[-2:]

        return target_qpos

    def get_action(self):
        """
        获取当前时间步的动作。

        参数：
        - qpos_list: 当前时间步的状态向量列表（例如，机器人关节位置）。
        - image_dict: 当前时间步的图像数据字典，包含多个相机视角的图像。

        返回：
        - action: 当前时间步生成的动作向量。
        """
        # print(self.qpos_list)
        qpos_list=self.qpos_list
        image_dict=self.image_dict
        start_time = time.time()
        # print(start_time)
        # 调用评估函数生成动作
        action = self._eval_bc_with_external_inputs(qpos_list, image_dict)

        end_time = time.time()
        # print(f"calculate time: {end_time - start_time}")#===================================================

        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, default=512, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true',default=False)
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')
    ActionGenerator = ActionGenerator(vars(parser.parse_args()))
    ActionGenerator.get_action()
