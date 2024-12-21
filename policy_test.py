from imitate_episodes import make_policy, set_seed
from constants import SIM_TASK_CONFIGS
from detr.models.latent_model import Latent_Model_Transformer
import os
import torch
import pickle
import numpy as np
import argparse
import time


def get_action(args):
    policy_config = {'lr': args['lr'],
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
    config = {
        'ckpt_dir': args['ckpt_dir'],
        'policy_class': args['policy_class'],
        'policy_config': policy_config,
        'episode_len': 100,
        'temporal_agg': False,
        'state_dim': 7,
    }

    # 假设你提供了以下数据
    qpos_list = np.random.randn(1, 7)  # 100 个状态
    # 生成一个包含每个相机名称的图像字典
    image_dict = {}
    # image_dict = args['image_dict']
    # qpos_list = args['qpos_list']
    for cam_name in policy_config['camera_names']:
        # 生成随机 RGB 图像，形状为 (H, W, C)
        random_image = np.random.randn(480, 640, 3)  # 假设图像大小为 (480, 640, 3)
        # 转换为 Tensor，并调整通道顺序 (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(random_image).permute(2, 0, 1).float()
        image_dict[cam_name] = image_tensor
    # image_tensor = image_tensor.permute( 1, 2, 3, 0)
    start_time = time.time()
    actions = eval_bc_with_external_inputs(config, 'policy_last.ckpt', qpos_list, image_dict)
    end_time = time.time()

    print("Elapsed time: ", end_time - start_time)
    return actions


def get_image(ts, camera_names, image_dict, rand_crop_resize=False):
    """
    获取图像并进行预处理，包括随机裁剪和调整大小。
    """
    curr_images = []
    for cam_name in camera_names:
        # 获取并调整图像维度
        curr_image = image_dict[cam_name]
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  # 归一化并转为Tensor
    return curr_image


def eval_bc_with_external_inputs(config, ckpt_name, qpos_list, image_dict, save_episode=True):
    """
    使用外部提供的 qpos 和 image 数据进行评估。

    参数：
    - config: 配置字典。
    - ckpt_name: 模型检查点文件名。
    - qpos_list: 外部提供的 qpos 数据（列表或 NumPy 数组）。
    - image_list: 外部提供的图像数据（列表）。
    - save_episode: 是否保存回合数据，默认为 True。
    """
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']
    vq = policy_config['vq']

    # 加载策略模型
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    policy.deserialize(torch.load(ckpt_path))
    policy.cuda()
    policy.eval()

    # if vq:
    #     vq_dim = policy_config['vq_dim']
    #     vq_class = policy_config['vq_class']
    #     latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
    #     latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
    #     latent_model.deserialize(torch.load(latent_model_ckpt_path))
    #     latent_model.eval()
    #     latent_model.cuda()
    #     print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
    # else:
    #     print(f'Loaded: {ckpt_path}')

    # 加载状态统计数据
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1

    max_timesteps = int(min(max_timesteps, len(qpos_list)))  # 防止超出输入数据范围
    # 初始化 action 变量
    action = None  # 初始化 action 为 None
    # 开始评估
    episode_returns = []
    for rollout_id in range(1):
        rewards = []
        for t in range(max_timesteps):
            qpos = pre_process(qpos_list[t])
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            curr_image = get_image(ts=None,  # ts 不传入时使用外部提供的图像
                                   camera_names=config['policy_config']['camera_names'], image_dict=image_dict,
                                   rand_crop_resize=False)  # 这里示例使用了随机裁剪和调整大小

            # 查询策略
            if config['policy_class'] == "ACT":
                # if vq:
                #     vq_sample = latent_model.generate(1, temperature=1, x=None)
                #     all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                # else:
                all_actions = policy(qpos, curr_image)
                raw_action = all_actions[:, t % query_frequency]
            elif config['policy_class'] == "Diffusion":
                all_actions = policy(qpos, curr_image)
                raw_action = all_actions[:, t % query_frequency]
            elif config['policy_class'] == "CNNMLP":
                raw_action = policy(qpos, curr_image)
            else:
                raise NotImplementedError

            # 后处理动作
            raw_action = raw_action.squeeze(0).detach().cpu().numpy()
            # print(raw_action)
            action = post_process(raw_action)
            target_qpos = action[:-2]
            base_action = action[-2:]
            print(f"action{action.shape}, base_action{base_action.shape}, target_qpos{target_qpos.shape}")
    # print(action)
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
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')

    get_action(vars(parser.parse_args()))
