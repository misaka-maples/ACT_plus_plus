# from rem_test import get_env
# get_env()
from datetime import datetime
import os, datetime
# from deploy.rem_test import current_time
from hdf5_edit import get_state
from policy_test import ActionGenerator
import numpy as np
import math
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualize_episodes import visualize_joints
from constants import HDF5_DIR, DATA_DIR
current_time = datetime.datetime.now()
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]


# def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
#     if label_overwrite:
#         label1, label2 = label_overwrite
#     else:
#         label1, label2 = 'State', 'Command'
#     qpos = np.array(qpos_list)  # 转为 NumPy 数组
#     command = np.array(command_list)
#     num_ts, num_dim = qpos.shape  # 时间步和维度数
#     h, w = 2, num_dim
#     num_figs = num_dim
#     fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))
#
#     # 绘制关节状态
#     all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
#     for dim_idx in range(num_dim):
#         ax = axs[dim_idx]
#         ax.plot(qpos[:, dim_idx], label=label1)
#         ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
#         ax.legend()
#
#     # 绘制动作指令
#     for dim_idx in range(num_dim):
#         ax = axs[dim_idx]
#         ax.plot(command[:, dim_idx], label=label2)
#         ax.legend()
#     if ylim:
#         for dim_idx in range(num_dim):
#             ax = axs[dim_idx]
#             ax.set_ylim(ylim)
#     plt.tight_layout()
#     plt.savefig(plot_path)
#     print(f'Saved qpos plot to: {plot_path}')
#     plt.close()


def main(args):
    actions = ActionGenerator(args)
    a = actions.get_action()

    return a


def rand_action():
    qpos_list = np.random.randn(1, 7)  # 100 个状态
    # 生成一个包含每个相机名称的图像字典
    image_dict = {}
    # image_dict = args['image_dict']
    # qpos_list = args['qpos_list']
    for cam_name in ['top', 'right_wrist']:
        # 生成随机 RGB 图像，形状为 (H, W, C)
        random_image = np.random.randn(480, 640, 3)  # 假设图像大小为 (480, 640, 3)
        # 转换为 Tensor，并调整通道顺序 (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(random_image).permute(2, 0, 1).float()
        image_dict[cam_name] = image_tensor
    return qpos_list, image_dict


if __name__ == '__main__':
    camera_top_data, camera_right_data, qpos_list = get_state(HDF5_DIR+
        '\episode_0.hdf5')
    actions_list = []
    loop_len = len(camera_right_data)-100
    for i in tqdm(range(loop_len)):
        # print(f"roll:{i}")
        image_dict = {
            'top': camera_top_data[i],
            'right_wrist': camera_right_data[i],
        }
        qpos = qpos_list[i]
        config = {
            'image_dict': image_dict,
            'qpos_list': qpos,
            'eval': True,  # 表示启用了 eval 模式（如需要布尔类型，直接写 True/False）
            'task_name': 'sim_transfer_cube_scripted',
            'ckpt_dir': DATA_DIR,
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
            'vq_class': None,
            'vq_dim': None,
            'no_encoder': False,
            'num_queries': 100,
        }
        # print(actions_list)

        actions = main(config)
        # actions=qpos
        actions = [i - 2 for i in actions]
        actions[2] = -actions[2]
        print(f"actions", actions)
        actions_list.append(actions)
        power = actions[6]
        actions = [i * 180.0 / math.pi for i in actions[:6]]

        # aa = QposRecorder.real_right_arm.Movej_CANFD(actions[:6], False)
        # QposRecorder.real_right_arm.Movej_Cmd(actions[:6],80,0,0,True)
        # if power > 2:
        #     QposRecorder.real_right_arm.Set_Tool_Voltage(3, True)
        # else:
        #     QposRecorder.real_right_arm.Set_Tool_Voltage(0, True)
        #
        # time.sleep(0.05)

    path_save_image = os.path.join(DATA_DIR, "deploy_image")
    if os.path.exists(path_save_image) is False:
        os.mkdir(path_save_image)
    image_path = os.path.join(path_save_image, current_time.strftime("%m-%d-%H-%M") + ".png")
    visualize_joints(qpos_list, actions_list, image_path)

    # print(actions_list)
