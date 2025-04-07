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
from utils import load_data  # data functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action  # helper functions
if False:
    from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy,HITPolicy
from policy_origin import ACTPolicy,CNNMLPPolicy,DiffusionPolicy
from visualize_episodes import save_videos
from detr.models.latent_model import Latent_Model_Transformer
import wandb
# 限制 PyTorch 只能看到 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 重新获取 GPU 设备索引
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"当前使用的 GPU: {device}")
print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")
server = "192.168.2.120"
class Train:
    def __init__(self):
        self.args = {
            'eval': False,
            'onscreen_render': False,
            'ckpt_dir': "/workspace/exchange/4_3_hdf5_file/act",#ckpt保存路径
            'dataset_dir':"/workspace/exchange/4_3_hdf5_file",#数据集路径
            'policy_class': 'ACT',
            'task_name': 'train',
            'batch_size': 4,
            'seed': 0,
            'num_steps': 20000,
            'lr': 1e-5,
            'kl_weight': 10,
            'load_pretrain': False,
            'eval_every': 1000,
            'validate_every': 1000,
            'save_every': 1000,
            'camera_names': ['top', 'right_wrist','left_wrist'],
            'resume_ckpt_path': 'ckpts/act/policy_best.ckpt',
            'skip_mirrored_data': False,
            'actuator_network_dir': None,
            'history_len': 10,
            'future_len': 10,
            'prediction_len': 10,
            'temporal_agg': False,
            'use_vq': False,
            'vq_class': None,
            'vq_dim': None,
            'vq': False,
            'no_encoder': False,
            'worker_num': 4,
            'chunk_size': 120,
            'num_queries':120,
            'hidden_dim': 512,
            'state_dim': 9,
            'action_dim': 11,
            'dim_feedforward': 3200,
            'num_heads': 8,
            'backbone': 'resnet50',
            'same_backbones':False,
            # 'lr_backbone': 1e-5,
            'feature_loss':False,
            'episode_len': 400,
            'drop_out':0.3,
            'wandb_project': 'ACT_Training',  # Project name for wandb
            'enc_layers': 4, 
            'dec_layers': 7, 
            'qpos_noise_std': 0,
            'train_ratio':0.95,
            # 'num_queries': 15,
        }
 
    def main(self):
        wandb.init(project=self.args['wandb_project'],name="train", config=self.args,settings=wandb.Settings(_disable_stats=True))
        # 从 self.args 中提取相关参数
        dataset_dir = self.args['dataset_dir']  # 假设使用 ckpt_dir 作为数据集目录
        batch_size_train = self.args['batch_size']
        batch_size_val = self.args['batch_size']
        chunk_size = self.args['chunk_size']  # 假设 'history_len' 对应数据块大小
        load_pretrain = self.args['load_pretrain']
        policy_class = self.args['policy_class']
        stats_dir = None
        sample_weights = None  # 如果需要，可以传递样本权重
        train_ratio = self.args['train_ratio']  # 默认训练集比例
        worker_num = self.args['worker_num']
        camera_names = self.args['camera_names']  # 摄像头名称列表
        # 调用 load_data 函数加载数据
        train_dataloader, val_dataloader, stats, _ = load_data(
            dataset_dir, 
            name_filter=self.args.get('name_filter', lambda n: True),  # 根据实际情况传入 name_filter 函数
            camera_names=camera_names,  # 根据实际情况传入摄像头名称列表
            batch_size_train=batch_size_train, 
            batch_size_val=batch_size_val, 
            chunk_size=chunk_size,
            skip_mirrored_data=self.args['skip_mirrored_data'],
            load_pretrain=load_pretrain, 
            policy_class=policy_class, 
            stats_dir_l=stats_dir, 
            sample_weights=sample_weights, 
            train_ratio=train_ratio, 
            worker_num=worker_num
        )
        # 创建检查点目录（如果不存在）
        os.makedirs(self.args['ckpt_dir'], exist_ok=True)
        # 设置配置文件路径并保存
        config_path = os.path.join(self.args['ckpt_dir'], 'config.pkl')
        expr_name = self.args['ckpt_dir'].split('/')[-1]
        with open(config_path, 'wb') as f:
            pickle.dump(self.args, f)
        # 保存数据集统计信息
        stats_path = os.path.join(self.args['ckpt_dir'], 'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        best_ckpt_info = self.train(train_dataloader, val_dataloader)
        best_step, min_val_loss, best_state_dict = best_ckpt_info
        # save best checkpoint
        ckpt_path = os.path.join(self.args['ckpt_dir'], f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    def train(self, train_dataloader, val_dataloader):
        # 从 self.args 获取配置信息
        num_steps = self.args['num_steps']
        ckpt_dir = self.args['ckpt_dir']
        seed = self.args['seed']
        policy_class = self.args['policy_class']
        eval_every = self.args['eval_every']
        validate_every = self.args['validate_every']
        save_every = self.args['save_every']

        set_seed(seed)

        # 创建策略
        policy = self.make_policy()
        if self.args['load_pretrain']:
            loading_status = policy.deserialize(torch.load(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'policy_step_50000_seed_0.ckpt')))
            print(f'loaded! {loading_status}')
        if os.path.exists(self.args['resume_ckpt_path']) is True:
            loading_status = policy.deserialize(torch.load(self.args['resume_ckpt_path']))
            print(f'Resume policy from: {self.args["resume_ckpt_path"]}, Status: {loading_status}')
        policy.cuda()
        optimizer = self.make_optimizer(policy)

        min_val_loss = np.inf
        best_ckpt_info = None

        train_dataloader = self.repeater(train_dataloader)
        for step in tqdm(range(num_steps + 1)):
            # validation
            if step % validate_every == 0 and step != 0:
                print(f'\nValidating')

                with torch.inference_mode():
                    policy.eval()
                    validation_dicts = []
                    for batch_idx, data in enumerate(val_dataloader):
                        forward_dict = self.forward_pass(data, policy)
                        # print(forward_dict)
                        validation_dicts.append(forward_dict)
                        if batch_idx > 50:
                            break
                    validation_summary = compute_dict_mean(validation_dicts)
                    epoch_val_loss = validation_summary['loss']
                    # Log validation loss to wandb
                    wandb.log({"验证_loss": epoch_val_loss, "step": step})

                    if epoch_val_loss < min_val_loss:
                        min_val_loss = epoch_val_loss
                        best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
                # Log additional metrics if necessary
                for k, v in validation_summary.items():
                    wandb.log({f"val_{k}": v.item(), "step": step})

                for k in list(validation_summary.keys()):
                    validation_summary[f'val_{k}'] = validation_summary.pop(k)
                summary_string = ''
                for k, v in validation_summary.items():
                    summary_string += f'{k}: {v.item():.3f} '
                print(summary_string)

            # evaluation
            if (step > 0) and (step % save_every == 0):
                # first save then eval
                ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                torch.save(policy.serialize(), ckpt_path)

            # training
            policy.train()
            optimizer.zero_grad()
            data = next(train_dataloader)
            forward_dict = self.forward_pass(data, policy)
             # Log training loss to wandb
            loss = forward_dict['loss']
            wandb.log({"training_loss": loss.item(), "step": step})

            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()

            # if step % save_every == 0:
            #     ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            #     torch.save(policy.serialize(), ckpt_path)

        # 最后保存模型
        ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
        torch.save(policy.serialize(), ckpt_path)
        best_step, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
        torch.save(best_state_dict, ckpt_path)
        wandb.finish()  # Close the wandb run
        print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')
        return best_ckpt_info
    def eval(self,ckpt_name):
        ckpt_dir = self.args['ckpt_dir']
        state_dim = self.args['state_dim']
        real_robot = self.args['real_robot']
        policy_class = self.args['policy_class']
        onscreen_render = self.args['onscreen_render']
        policy_config = self.args['policy_config']
        camera_names = self.args['camera_names']
        max_timesteps = self.args['episode_len']
        task_name = self.args['task_name']
        temporal_agg = self.args['temporal_agg']
        onscreen_cam = 'angle'
        vq = self.args['policy_config']['vq']
        actuator_config = self.args['actuator_config']
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        policy = self.make_policy(policy_class, policy_config)
        loading_status = policy.deserialize(torch.load(ckpt_path, weights_only=True))
        print(loading_status)
        policy.cuda()
        policy.eval()
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
        query_frequency = self.args['chunk_size']
        if temporal_agg:
            query_frequency = 1
        max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    def data_loader(self):
        pass
    def make_policy(self):
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
        if self.args['policy_class'] == 'ACT':
            policy = ACTPolicy(self.args)
        elif self.args['policy_class'] == 'CNNMLP':
            policy = CNNMLPPolicy(self.args)
        # elif self.args['policy_class'] == 'HIT':
        #     policy = HITPolicy(self.args)
        elif self.args['policy_class'] == 'Diffusion':
            policy = DiffusionPolicy(self.args)
        else:
            raise NotImplementedError(f"Policy class '{self.args['policy_class']}' is not implemented.")
        return policy
    def make_optimizer(self,policy):
        if self.args['policy_class'] == 'ACT':
            optimizer = policy.configure_optimizers()
        elif self.args['policy_class'] == 'HIT':
            optimizer = policy.configure_optimizers()
        elif self.args['policy_class'] == 'CNNMLP':
            optimizer = policy.configure_optimizers()
        elif self.args['policy_class'] == 'Diffusion':
            optimizer = policy.configure_optimizers()
        else:
            raise NotImplementedError
        return optimizer
    def forward_pass(self,data,policy):
         # 解包输入数据，其中包含图像数据、qpos 数据、动作数据和填充标记
        image_data, qpos_data, action_data, is_pad = data

        # 将所有输入数据移动到 GPU 上（使用 .cuda()）
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()

        # # 使用 policy 执行前向传播，传入相应的输入数据
        # print(f"qpos_data: {qpos_data.shape}")
        # print(f"image_data: {image_data.shape}")
        # print(f"action_data: {action_data.shape}")
        return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None
    def repeater(self,data_loader):
        epoch = 0
        for loader in repeat(data_loader):
            for data in loader:
                yield data
            print(f'Epoch {epoch} done')
            epoch += 1
if __name__ == '__main__':
    train = Train()
    train.main()