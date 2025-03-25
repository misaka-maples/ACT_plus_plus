import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
# from constants import DT
DT=0.02
import subprocess
import IPython

e = IPython.embed

# 定义关节和状态名称
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]
# results = subprocess.run(['h5check', './results/episode_1.hdf5'], capture_output=True, text=True)
# if "File passed checksum test" in results.stdout:
#     print("File passed checksum test")
# 加载 HDF5 格式的数据集
def load_hdf5(dataset_dir, dataset_name):
    # 构建数据集的路径
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    try:
        # 打开 HDF5 文件并读取数据
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']  # 是否是模拟数据
            qpos = root['/observations/qpos'][()]  # 关节位置数据
            # qvel = root['/observations/qvel'][()]  # 关节速度数据
            action = root['/action'][()]  # 动作数据

            # 读取摄像头图像数据
            compressed = root.attrs.get('compress', False)
            image_dict_ = dict()  # 初始化一个空字典来存储解压后的图像

            # 初始化每个摄像头的键为一个空列表，用于存储解压后的图像数据
            for cam_name in ['top', 'right_wrist']:
                image_dict_[cam_name] = []

            # 处理图像数据
            for cam_name in ['top',  'right_wrist']:
                compressed_data = root[f'/observations/images/{cam_name}'][()]  # 获取压缩图像数据

                if compressed:
                    num_images = compressed_data.shape[0]
                    for i in range(num_images):
                        # 获取单个压缩图像并解压
                        compressed_image = compressed_data[i]
                        decompressed_image = cv2.imdecode(compressed_image, 1)  # 解压为彩色图像

                        # 将解压后的图像添加到对应的摄像头列表中
                        image_dict_[cam_name].append(decompressed_image)

            # 将每个摄像头的图像列表转换为四维数组 (height, width, channels, num_images)
            for cam_name in image_dict_:
                image_dict_[cam_name] = np.stack(image_dict_[cam_name], axis=0)

            # # 输出解压后的图像数据形状（可选）
            # for cam_name, images in image_dict_.items():
            #     print(f"{cam_name}: {images.shape}")  # 输出每个摄像头图像的维度

    except Exception as e:
        print(f'Dataset {dataset_name} does not exist at \n{dataset_path}\n {e}')

    return qpos, action, image_dict_

# 主程序
def main(args):
    dataset_dir = args['dataset_dir']  # 数据集目录
    episode_idx = args['episode_idx']  # 具体的episode索引
    ismirror = args['ismirror']  # 是否是镜像数据

    # 根据是否镜像命名数据集文件名
    if ismirror:
        dataset_name = f'mirror_episode_{episode_idx}'
    else:
        dataset_name = f'episode_{episode_idx}'

    # 加载数据
    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)

    # 保存视频
    save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    # 可视化关节数据
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    # visualize_timestamp(t_list, dataset_path) # TODO 添加时间戳的可视化


# 保存视频
def save_videos(video, dt, video_path=None):
    # 如果视频是列表（按时间戳组织）
    if isinstance(video, list):
        cam_names = list(video[0].keys())  # 获取所有摄像头的名称
        cam_names = sorted(cam_names)
        h, w, _ = video[0][cam_names[0]].shape  # 获取单帧的图像尺寸
        w = w * len(cam_names)  # 总宽度是所有摄像头宽度的总和
        fps = int(1 / dt)  # 帧率
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # 按帧写入视频
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # 交换图像的B和R通道
                images.append(image)
            images = np.concatenate(images, axis=1)  # 按宽度拼接
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')

    # 如果视频是字典（按摄像头组织）
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        cam_names = sorted(cam_names)
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # 按宽度拼接

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # 交换图像的B和R通道
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


# 可视化关节状态和动作
def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None,STATE_NAMES=STATE_NAMES):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list)  # 转为 NumPy 数组
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape  # 时间步和维度数
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # 绘制关节状态
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # 绘制动作指令
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()


# 可视化时间戳（未被调用）
def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h * 2))

    # 处理时间戳
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    # 绘制时间戳和时间间隔
    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float) - 1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()


# 程序入口
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--ismirror', action='store_true')
    main(vars(parser.parse_args()))
