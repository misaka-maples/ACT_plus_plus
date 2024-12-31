from datetime import datetime
import os, datetime, sys
import argparse

from networkx.readwrite.json_graph.tree import tree_data
from triton.language.semantic import store

# from deploy.rem_test import current_time
from hdf5_edit import get_state
from policy_action_generation import ActionGenerator
import numpy as np
import math
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualize_episodes import visualize_joints
# from constants import HDF5_DIR, DATA_DIR
from hdf5_edit import get_image_from_folder, get_top_right_image
current_time = datetime.datetime.now()
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]


class QposRecorder:
    def __init__(self):
        self.joint_state_right=None
        self.real_right_arm = (RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E))
        self.arm_ini = self.real_right_arm.rm_create_robot_arm("192.168.1.18",8080, level=3)
        # self.robot_controller = RoboticArm("192.168.1.18", 8080, 3)
    def get_state(self):
        self.joint_state_right = self.real_right_arm.rm_get_current_arm_state()
        # print(f"get state test", self.joint_state_right)
        return_action = self.joint_state_right[1]['joint']
        # print(return_action)
        return return_action

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

camera_names = ['top', 'right_wrist']
def main(args):
    camera_top_data, camera_right_data, qpos_list, action_ = get_state(
        r'/home/zhnh/Documents/project/act_arm_project/temp/episode_40.hdf5')
    # actions_list = []
    # qpos_list = []
    images_dict = {cam_name: [] for cam_name in camera_names}  # 用于存储每个相机的图片
    actions_list = []
    qpos_list_ = []
    loss = []
    loop_len = len(camera_right_data)
    if args['folder_get_image']:
        image_directory = r"D:\aloha\ACT_plus_plus\hdf5_file\04"  # 图像文件夹路径
        top_image, right_image = get_top_right_image(image_directory, '.jpg')
    config = {
        'eval': True,  # 表示启用了 eval 模式（如需要布尔类型，直接写 True/False）
        'task_name': 'train',
        'ckpt_dir': r'/home/zhnh/Documents/project/act_arm_project/temp',
        'policy_class': 'ACT',
        'chunk_size': 210,
        'backbone': 'resnet18',
        'temporal_agg':True,
        'max_timesteps': loop_len,
    }
    ActionGeneration = ActionGenerator(config)
    if args['joint_true'] is True:
        from Robotic_Arm.rm_robot_interface import RoboticArm
        posRecorder = QposRecorder()

    for i in tqdm(range(loop_len)):
        # print(f"roll:{i}")
        ActionGeneration.t = i
        image_dict = {
            'top': camera_top_data[i],
            'right_wrist': camera_right_data[i],
        }
        # print(image_dict)
        if args['joint_true'] is True:
            qpos = posRecorder.get_state()
        else:
            qpos = qpos_list[i]
        radius_qpos = [math.radians(j) for j in qpos]
        ActionGeneration.image_dict = image_dict
        ActionGeneration.qpos_list = radius_qpos
        actions = ActionGeneration.get_action()
        # print(qpos)
        # actions=qpos
        actions_list.append(actions)
        loss.append((actions - action_[i]) ** 2)
        if args['joint_true'] is True:
            power = actions[6]
            if power > 2.7:
                posRecorder.real_right_arm.rm_set_tool_voltage(3)
            else:
                posRecorder.real_right_arm.rm_set_tool_voltage(0)
            actions = [i * 180.0 / math.pi for i in actions[:6]]
            posRecorder.real_right_arm.rm_movej(actions, 50, 0, 0, 1)

    path_save_image = os.path.join(os.getcwd(), "deploy_image")
    if os.path.exists(path_save_image) is False:
        os.mkdir(path_save_image)
    image_path = os.path.join(path_save_image, current_time.strftime("%m-%d-%H-%M") + ".png")
    loss_apth = os.path.join(path_save_image, 'loss' + current_time.strftime("%m-%d-%H-%M") + ".png")
    visualize_joints(qpos_list, actions_list, image_path)
    plt.figure()
    plt.plot(loss)
    plt.legend(JOINT_NAMES)
    plt.savefig(loss_apth)
    plt.show()
    plt.close()
    # print(actions_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_get_image', default=False, type=float)
    parser.add_argument('--joint_true', default=False, required=False, action='store_true')

    a = main(vars(parser.parse_args()))
