from policy_test import ActionGenerator
import torch
import numpy as np
import argparse
image_dict={}
for cam_name in ['top', 'left_wrist', 'right_wrist']:
    # 生成随机 RGB 图像，形状为 (H, W, C)
    random_image = np.random.randn(480, 640, 3)  # 假设图像大小为 (480, 640, 3)
    # 转换为 Tensor，并调整通道顺序 (H, W, C) -> (C, H, W)
    # image_tensor = torch.from_numpy(random_image).permute(2, 0, 1).float()
    # image_dict[cam_name] = image_tensor
    image_dict[cam_name]=random_image
qpos_list = np.random.randn(1,7)  # 100 个状态
config = {
    'image_dict': image_dict,
    'qpos_list': qpos_list,
    'eval': True,  # 表示启用了 eval 模式（如需要布尔类型，直接写 True/False）
    'task_name': 'sim_transfer_cube_scripted',
    'ckpt_dir': '/home/zhnh/Documents/xzx_projects/aloha_deploy/act-plus-plus/results',
    'policy_class': 'ACT',
    'kl_weight': 10,
    'chunk_size': 30,
    'hidden_dim': 512,
    'batch_size': 8,
    'dim_feedforward': 3200,
    'num_steps': 2000,
    'lr': 1e-5,
    'seed': 0,
    'use_vq': False,
    'vq_class': 1,
    'vq_dim': 7,
    'no_encoder': False,
    'num_queries': 100,
}
ActionGenerator=ActionGenerator(config)
action = ActionGenerator.get_action()
print(action)