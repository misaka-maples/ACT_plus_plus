# from rem_test import get_env
# get_env()
from datetime import datetime
import os, datetime, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
        os.path.join(parent_dir, 'robomimic')
    ]

    # 动态添加目录到 sys.path
    for directory in dirs_to_add:
        if directory not in sys.path:  # 避免重复添加
            sys.path.append(directory)

get_env_()
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
from hdf5_edit import get_image_from_folder

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
    camera_top_data, camera_right_data, qpos_list, action_ = get_state(HDF5_DIR +
                                                                       '\episode_4.hdf5')
    actions_list = []
    loss = []
    loop_len = len(camera_right_data)
    image_directory = r"D:\aloha\ACT_plus_plus\hdf5_file\04"  # 图像文件夹路径
    right_image = "camera_right_wrist"  # 图像文件名前缀
    top_image = "camera_top"
    image_extension = ".jpg"  # 图像扩展名
    num_images = 137  # 图像数量
    top__ = get_image_from_folder(image_directory, top_image, image_extension)
    right__ = get_image_from_folder(image_directory, right_image, image_extension)
    for i in tqdm(range(loop_len)):
        # print(f"roll:{i}")
        image_dict = {
            'top': camera_top_data[i],
            'right_wrist': camera_right_data[i],
        }
        # image_dict = {
        #     'top': top__[i],
        #     'right_wrist': right__[i],
        # }
        # print(image_dict)
        qpos = qpos_list[i]
        config = {
            'image_dict': image_dict,
            'qpos_list': qpos,
            'eval': True,  # 表示启用了 eval 模式（如需要布尔类型，直接写 True/False）
            'task_name': 'train',
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
        loss.append((actions - action_[i]) ** 2)
        actions = [i - 2 for i in actions]
        actions[2] = -actions[2]
        # print(f"actions", actions)##====================================================================
        actions_list.append(actions)
        # print(f"actions: {actions[2]},action_:{-action_[i][2] + 2}")
        # loss.append((actions - action_[i]-2) ** 2)
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
    loss_apth = os.path.join(path_save_image, 'loss' + current_time.strftime("%m-%d-%H-%M") + ".png")
    visualize_joints(qpos_list, actions_list, image_path, label_overwrite={'qpos', 'action'})
    plt.figure
    plt.plot(loss)
    plt.legend(JOINT_NAMES)
    plt.savefig(loss_apth)
    plt.show()
    plt.close()
    # print(actions_list)
