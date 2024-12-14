
def get_env_():
    import sys
    import os

    # 当前文件的目录
    current_dir = os.path.dirname(__file__)

    # 上一级目录
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

    # 要添加的目录
    dirs_to_add = [
        parent_dir,  # 项目根目录
        os.path.join(parent_dir, 'detr'),  # 上一级的
        os.path.join(parent_dir, 'robomimic'),
        os.path.join(parent_dir, 'robomimic', 'robomimic')
    ]

    # 动态添加目录到 sys.path
    for directory in dirs_to_add:
        if directory not in sys.path:  # 避免重复添加
            sys.path.append(directory)

get_env_()
import wandb
import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange

import time
from torchvision import transforms

from constants import FPS
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data  # data functions
from utils import sample_box_pose, sample_insertion_pose  # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action  # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
# from visualize_episodes import save_videos
from detr.models.latent_model import Latent_Model_Transformer
# from sim_env import BOX_POSE
import os
os.environ["WANDB_MODE"] = "disabled"  # 禁用wandb
# settings = wandb.Settings(
#     moitor_=False,       # 禁用 GPU 监控
#     monitor_cpu=False,        # 禁用 CPU 监控
#     monitor_network=False,    # 禁用网络监控
#     monitor_disk=False,       # 禁用磁盘监控
#     monitor_memory=False,     # 禁用内存监控（如果适用）
#     monitor_system=False      # 禁用系统监控（综合）
# )
wandb.init(
    project='act++',
    name='loss_log',
    # settings=settings
)
#63d8db0b669a1b1c712284f186093c19c278719c
def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def main(args):
    # set_seed(1)
    # command line parameters
    task_name = args['task_name']
    if task_name is None:
        raise "No task name specified"
    is_sim = task_name[:4] == 'sim_'
    if is_sim or task_name == 'all':
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    elif task_name == 'train' or task_name == 'train_test':
        from constants import RIGHT_ARM_TASK_CONFIGS
        task_config = RIGHT_ARM_TASK_CONFIGS[task_name]

    is_eval = args['eval']
    ckpt_dir = task_config['ckpt_dir']
    policy_class = task_config['policy_class']
    onscreen_render = args['onscreen_render']
    batch_size_train = task_config['batch_size']
    batch_size_val = task_config['batch_size']
    num_steps = task_config['num_steps']
    eval_every = task_config['eval_every']
    validate_every = task_config['validate_every']
    save_every = task_config['save_every']
    resume_ckpt_path = task_config['resume_ckpt_path']

    # get task parameters

    dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # fixed parameters
    state_dim = 7
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': task_config['lr'],
                         'num_queries': task_config['chunk_size'],
                         'kl_weight': task_config['kl_weight'],
                         'hidden_dim': task_config['hidden_dim'],
                         'dim_feedforward': task_config['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': args['use_vq'],
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         'action_dim': 9,
                         'no_encoder': args['no_encoder'],
                         'state_dim': 7
                         }
    elif policy_class == 'Diffusion':

        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'action_dim': 9,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,

                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1,
                         'camera_names': camera_names, }
    else:
        raise NotImplementedError

    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],
        'history_len': args['history_len'],
        'future_len': args['future_len'],
        'prediction_len': args['prediction_len'],
    }

    config = {
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': policy_config['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': task_config['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': task_config['load_pretrain'],
        'actuator_config': actuator_config,

    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = ckpt_dir.split('/')[-1]
    # if not is_eval:
    #     wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)
    #     wandb.config.update(config)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)
    if is_eval:
        ckpt_names = [f'policy_last.ckpt']
        results = []
        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, task_config['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    # wandb.finish()


def make_policy(policy_class, policy_config):
    """
    根据指定的策略类别创建对应的策略实例。

    参数：
    - policy_class (str): 策略类别的名称，可选值为 'ACT', 'CNNMLP', 'Diffusion'。
    - policy_config (dict): 策略配置字典，包含初始化策略所需的参数。

    返回：
    - policy: 创建的策略实例。

    异常：
    - NotImplementedError: 如果指定的策略类别未实现，则抛出该异常。
    """
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError(f"Policy class '{policy_class}' is not implemented.")
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image




def forward_pass(data, policy):
    # 解包输入数据，其中包含图像数据、qpos 数据、动作数据和填充标记
    image_data, qpos_data, action_data, is_pad = data

    # 将所有输入数据移动到 GPU 上（使用 .cuda()）
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()

    # 使用 policy 执行前向传播，传入相应的输入数据
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        loading_status = policy.deserialize(torch.load(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'policy_step_50000_seed_0.ckpt')))
        print(f'loaded! {loading_status}')
    if os.path.exists(config['resume_ckpt_path']) is True:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = np.inf
    best_ckpt_info = None

    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps + 1)):
        # validation
        if step % validate_every == 0 and step != 0:
            print(f'\nValidating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)
                print(f"validation_dicts:{validation_dicts}\nvalidation_summary:{validation_summary}")
                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            # print(f"1111111111111111111111111111111111111111111111")
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)
                wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # evaluation
        if (step > 0) and (step % eval_every == 0):
            # first save then eval
            ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)
            # success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=1)
            # wandb.log({'success': success}, step=step)

        # training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        wandb.log(forward_dict, step=step) # not great, make training 1.md-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)
    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')
    return best_ckpt_info


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=False)
    parser.add_argument('--load_pretrain', action='store_true', default=False)
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False, default=None)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')
    # with PyCallGraph(output=GraphvizOutput()):
    a = main(vars(parser.parse_args()))
