import os
import h5py
import argparse
from collections import defaultdict
from sim_env import make_sim_env
from utils import sample_box_pose, sample_insertion_pose
from sim_env import BOX_POSE
from constants import DT
from visualize_episodes import save_videos

import IPython

e = IPython.embed  # 用于调试的 IPython 会话，代码暂停时进入调试模式


def main(args):
    # 获取传入的参数（命令行参数）
    dataset_path = args['dataset_path']

    # 检查指定的 dataset_path 文件是否存在
    if not os.path.isfile(dataset_path):
        print(f'数据集在 {dataset_path} 路径下不存在。')
        exit()  # 如果文件不存在，退出程序

    # 读取 HDF5 文件，获取动作数据
    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]  # 获取 '/action' 路径下的数据（即动作数据）

    # 初始化仿真环境
    env = make_sim_env('sim_transfer_cube')  # 创建一个名为 'sim_transfer_cube' 的仿真环境
    BOX_POSE[0] = sample_box_pose()  # 随机生成一个盒子的位置姿态，供仿真重置使用
    ts = env.reset()  # 重置仿真环境，获取初始状态
    episode_replay = [ts]  # 初始化回放列表，将初始状态添加进去

    # 对于每个动作，执行并记录状态
    for action in actions:
        ts = env.step(action)  # 在环境中执行当前动作，返回新的状态
        episode_replay.append(ts)  # 将新的状态添加到回放列表中

    # 准备保存视频
    image_dict = defaultdict(lambda: [])  # 使用 defaultdict 来存储各个相机的图像（默认值为空列表）

    # 遍历回放列表，提取每个时间步的图像
    while episode_replay:
        ts = episode_replay.pop(0)  # 弹出回放列表中的第一个状态
        # 遍历当前状态下每个相机的图像
        for cam_name, image in ts.observation['images'].items():
            image_dict[cam_name].append(image)  # 将图像按相机名称添加到字典中

    # 生成视频文件路径，并保存视频
    video_path = dataset_path.replace('episode_', 'replay_episode_').replace('hdf5', 'mp4')  # 替换文件名，生成视频路径
    save_videos(image_dict, DT, video_path=video_path)  # 使用 `save_videos` 保存回放视频


if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', action='store', type=str, help='数据集路径', required=True)  # 输入数据集路径
    main(vars(parser.parse_args()))  # 解析命令行参数并调用 main 函数
