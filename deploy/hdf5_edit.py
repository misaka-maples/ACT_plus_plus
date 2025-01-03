import copy
import random

import matplotlib.pyplot as plt

import h5py
import cv2
import os
import fnmatch
import numpy as np
from constants import RIGHT_ARM_TASK_CONFIGS, HDF5_DIR, DATA_DIR
camera_names = RIGHT_ARM_TASK_CONFIGS['train']['camera_names']
max_timesteps = 0


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def get_key(file_path):
    data = []
    with h5py.File(file_path, 'r') as hdf:
        # 定义递归函数来遍历所有组和数据集
        def visit_function(name, obj):
            # print(name)  # 打印路径
            data.append(name)
        # 遍历 HDF5 文件中的所有路径
        hdf.visititems(visit_function)
        # print(data)
    return data
# 获取hdf5数据
def get_state(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            compressed = f.attrs.get('compress', False)
            # print(compressed)
            data = get_key(file_path)
            # top_paths = [path for path in data if 'top' in path]            # 检查是否存在原始的路径
            top_paths = [path for path in data if 'top' in path or 'high' in path]
            right_paths = [path for path in data if 'right' in path]
            qpos_paths = [path for path in data if 'qpos' in path]
            action_paths = [path for path in data if 'action' in path]
            # print(top)
            # 确保每个路径至少有一个匹配项
            if not top_paths or not right_paths or not qpos_paths or not action_paths:
                raise KeyError(f"Paths not found in the HDF5 file.")

            # 获取路径
            top = top_paths[0]  # 如果有多个 'top'，只取第一个
            right = right_paths[0]  # 如果有多个 'right'，只取第一个
            qpos = qpos_paths[0]  # 如果有多个 'qpos'，只取第一个
            action = action_paths[0]  # 如果有多个 'action'，只取第一个

            if top not in f:
                raise KeyError(f"Path '{top}' not found in the HDF5 file.")
            if right not in f:
                raise KeyError(f"Path '{right}' not found in the HDF5 file.")
            if qpos not in f:
                raise KeyError(f"Path '{qpos}' not found in the HDF5 file.")
            if action not in f:
                raise KeyError(f"Path '{action}' not found in the HDF5 file.")
            camera_top_data = f[top][:]
            camera_right_data = f[right][:]
            qpos = f[qpos][:]
            action = f[action][:]
            camera_top_data_list = []  # 用于存储解压后的帧
            camera_right_data_list = []
            if compressed:
                num_images = camera_top_data.shape[0]

                for i in range(num_images):
                    compressed_image = camera_top_data[i]
                    compressed_image_ = camera_right_data[i]
                    # 解压为彩色图像
                    decompressed_image = cv2.imdecode(compressed_image, 1)
                    decompressed_image_ = cv2.imdecode(compressed_image_, 1)
                    # 确保通道顺序是 BGR
                    # decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_RGB2BGR)
                    # image_list.append(decompressed_image)
                    camera_top_data_list.append(decompressed_image)
                    camera_right_data_list.append(decompressed_image_)
            else:
                camera_top_data_list = camera_top_data
                camera_right_data_list = camera_right_data
    except Exception as e:
        print(f"Error load hdf5 file:\n {e}")
    return camera_top_data_list, camera_right_data_list, qpos, action


#用于修改hdf5文件，再写入文件
def modify_hdf5(file_path, compress=None, truncate_ranges=None, edit=False):
    """
    修改 HDF5 文件中的摄像头数据，并在指定位置截断数据。

    将 `observations/images/camera_top` 修改为：
    - `camera_top`
    - `left_wrist`
    - `right_wrist`

    参数:
        file_path (str): HDF5 文件的路径。
        compress (bool): 是否设置压缩标志。
        truncate_ranges (dict): 各数据的截断范围。格式为 {'camera_top': (start, end), 'actions': (start, end)}。
    """
    try:
        with h5py.File(file_path, 'r+') as f:
            path_collection = []
            # 检查是否存在原始的路径
            data = get_key(file_path)
            for p in data:
                print(f"已经存在的路径",p)
            # top_paths = [path for path in data if 'top' in path]            # 检查是否存在原始的路径
            top_paths = [path for path in data if 'top' in path or 'high' in path]
            right_paths = [path for path in data if 'right' in path]
            left_paths = [path for path in data if 'left' in path]
            qpos_paths = [path for path in data if 'qpos' in path]
            action_paths = [path for path in data if 'action' in path]
            # print(top)
            # 确保每个路径至少有一个匹配项
            if not top_paths or not right_paths or not qpos_paths or not action_paths:
                raise KeyError(f"Paths not found in the HDF5 file.")

            # 获取路径
            top = top_paths[0]  # 如果有多个 'top'，只取第一个
            right = right_paths[0]  # 如果有多个 'right'，只取第一个
            qpos = qpos_paths[0]  # 如果有多个 'qpos'，只取第一个
            action = action_paths[0]  # 如果有多个 'action'，只取第一个
            left = left_paths[0]
            if top not in f:
                raise KeyError(f"Path '{top}' not found in the HDF5 file.")
            if right not in f:
                raise KeyError(f"Path '{right}' not found in the HDF5 file.")
            if left not in f:
                raise KeyError(f"Path '{left}' not found in the HDF5 file.")
            if qpos not in f:
                raise KeyError(f"Path '{qpos}' not found in the HDF5 file.")
            if action not in f:
                raise KeyError(f"Path '{action}' not found in the HDF5 file.")

            # 获取原始数据
            camera_top_data = f[top][:]
            camera_right_data = f[right][:]
            camera_left_data = f[left][:]
            qpos = f[qpos][:]
            actions = f[action][:]
            # print(qpos[0])
            # print(f"camera_top_data.shape: {camera_top_data.shape}, camera_right_data.shape: {camera_right_data.shape}")
            if edit:
                # 解压逻辑
                def decompress_images(compressed_data):
                    """解压图像数据"""
                    image_list = []
                    num_images = compressed_data.shape[0]
                    for i in range(num_images):
                        compressed_image = compressed_data[i]
                        # 解压为彩色图像
                        decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
                        if decompressed_image is None:
                            raise ValueError("Failed to decompress image. Data might not be valid compressed format.")
                        image_list.append(decompressed_image)

                    return np.array(image_list)

                    # 判断是否需要解压，如果图像维度是 (num_images, 480, 640, 3)，则不进行解压

                if camera_right_data.shape[1:] != (480, 640, 3):
                    if compress:
                        camera_right_data = decompress_images(camera_right_data)
                        print(camera_right_data.shape)
                else:
                    print(
                        f"\ncamera_right_data is already in the correct shape, skipping decompression.shape{camera_right_data.shape}")
                if camera_top_data.shape[1:] != (480, 640, 3):
                    if compress:
                        camera_top_data = decompress_images(camera_top_data)
                        print(camera_top_data.shape)
                else:
                    print(
                        f"\ncamera_top_data is already in the correct shape, skipping decompression.shape{camera_top_data.shape}")

                if camera_top_data.shape[1:] == (480, 640, 3) and camera_right_data.shape[1:] == (480, 640, 3):
                    f.attrs['compress'] = False
                # camera_top_data = decompress_images(camera_top_data)
                qpos = qpos[:, :7]
                actions = actions[:, :7]

                # 截断数据，如果指定了截断范围
                def truncate_data(data, key):
                    """根据指定的截断范围进行截断"""
                    if truncate_ranges and key in truncate_ranges:
                        start, end = truncate_ranges[key]
                        print(f"Truncating {key} from {start} to {end}")
                        return data[start:end]
                    return data

                camera_top_data = truncate_data(camera_top_data, 'top')
                camera_right_data = truncate_data(camera_right_data, 'right_wrist')
                camera_left_data = truncate_data(camera_left_data, 'left_wrist')
                qpos = truncate_data(qpos, 'qpos')
                actions = truncate_data(actions, 'action')

                # 创建新的路径并写入数据
                new_paths_top = ['observations/images/top']
                new_paths_right = ['observations/images/right_wrist']
                new_paths_left = ['observations/images/left_wrist']
                new_qpos_path = ['observations/qpos']
                new_actions_path = ['action']

                for path in new_actions_path:
                    if path in f:
                        del f[path]
                    f.create_dataset(path, data=actions)

                for path in new_qpos_path:
                    if path in f:
                        del f[path]
                    f.create_dataset(path, data=qpos)

                for path in new_paths_top:
                    if path in f:
                        del f[path]
                    f.create_dataset(path, data=camera_top_data)

                for path in new_paths_right:
                    if path in f:
                        del f[path]
                    f.create_dataset(path, data=camera_right_data)
                for path in new_paths_left:
                    if path in f:
                        del f[path]
                    f.create_dataset(path, data=camera_left_data)

                print("Modification complete. Paths updated:")
                for path in new_paths_top:
                    print(f"  - {path}")
                for path in new_paths_right:
                    print(f"  - {path}")
                for path in new_paths_left:
                    print(f"  - {path}")
                for path in new_qpos_path:
                    print(f"  - {path}")
                for path in new_actions_path:
                    print(f"  - {path}")

            print(f'compress:', f.attrs.get('compress'))
    except Exception as e:
        print(f"Error modifying HDF5 file:\n {e}")


def batch_modify_hdf5(dataset_dir, output_dir=None, skip_mirrored_data=True):
    """
    批量修改指定目录中的 HDF5 文件的 actions 数据。

    参数:
        dataset_dir (str): 数据集目录路径。
        output_dir (str): 修改后的文件保存路径。如果为 None，则覆盖原文件。
        skip_mirrored_data (bool): 是否跳过包含 "mirror" 的文件。
    """
    hdf5_files = find_all_hdf5(dataset_dir, skip_mirrored_data)

    for file_path in hdf5_files:
        if output_dir:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            # 构造新文件路径
            output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        else:
            output_file_path = None

        modify_hdf5(file_path, output_file_path)


rand = random.random()


def save_video(file_path, fps=10, i=0, arm='right_wrist'):
    dataset_path = os.path.join(file_path, f'episode_{i}' + '.hdf5')
    try:
        with h5py.File(dataset_path, 'r') as f:
            # print(f.keys())
            right_image_path = 'observations/images/right_wrist'
            if right_image_path not in f:
                raise KeyError(f"Path '{right_image_path}' not found in the HDF5 file.")
            top_image_path = 'observations/images/top'
            if top_image_path not in f:
                raise KeyError(f"Path '{top_image_path}' not found in the HDF5 file.")
            left_image_path='observations/images/left_wrist'
            if top_image_path not in f:
                raise KeyError(f"Path '{left_image_path}' not found in the HDF5 file.")
            right_wrist_data = f[right_image_path][()]  # 读取图像数据
            top_data = f[top_image_path][()]  # 读取图像数据
            left_data = f[left_image_path][()]
            compressed = f.attrs.get('compress', False)
            if 'right' in arm:
                arm_data = right_wrist_data
            elif 'top' in arm:
                arm_data = top_data
            elif 'left' in arm:
                arm_data = left_data
            image_list = []  # 用于存储解压后的帧
            if compressed:
                num_images = arm_data.shape[0]
                for i in range(num_images):
                    compressed_image = arm_data[i]
                    # 解压为彩色图像
                    decompressed_image = cv2.imdecode(compressed_image, 1)
                    # 确保通道顺序是 BGR
                    # decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_RGB2BGR)
                    # image_list.append(decompressed_image)
                    image_list.append(decompressed_image)
            else:
                # 假设数据直接是未压缩图像数组
                image_list = [frame for frame in arm_data]
            print(os.path.splitext(os.path.basename(dataset_path))[0])
            output_path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}_"+arm+".jpg")

            # for i in range(len(image_list)):
            # print(f"image_list_len:{len(image_list)}")
            # output_path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}-{i}.jpg")
            cv2.imwrite(output_path, image_list[0][:, :, [0, 1, 2]])
            # 如果图像列表为空，抛出错误
            if not image_list:
                raise ValueError("No images found to save as video.")

            # 获取帧的宽度和高度
            frame_height, frame_width, _ = image_list[0].shape

            # 定义视频写入器

            path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}_"+arm+".mp4")
            video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            # 将每一帧写入视频
            for frame in image_list:
                # image_list = image_list[:, :, [2, 1, 0]]  # 交换图像的B和R通道
                frame = frame[:, :, [0, 1, 2]]
                video_writer.write(frame)

            # 释放视频写入器
            video_writer.release()
            if os.path.exists(path):
                print(f"\nVideo saved successfully at {file_path}")
            else:
                raise "error to save video"
    except Exception as e:

        print(f"Error saving video:\n {e}")


def get_image_paths(directory, prefix, extension):
    """
    获取指定目录下所有图像的路径。

    参数:
        directory (str): 图像文件夹路径。
        prefix (str): 图像文件名前缀。
        extension (str): 图像文件扩展名，如 '.jpg'。

    返回:
        list: 包含所有图像文件路径的列表。
    """
    image_paths = []

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 仅选择符合前缀和扩展名的文件
        if filename.startswith(prefix) and filename.endswith(extension):
            file_path = os.path.join(directory, filename)
            image_paths.append(file_path)

    if not image_paths:
        print(f"Warning: No images found with prefix '{prefix}' and extension '{extension}' in {directory}.")

    return image_paths


def read_image(image_path, decompress=False):
    """
    读取图像，并根据需要进行解压操作。

    参数:
        image_path (str): 图像文件的路径。
        decompress (bool): 是否解压图像（适用于压缩格式）。

    返回:
        np.ndarray: 读取后的图像数据。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image at {image_path} not found.")

    # 读取压缩格式的图像 (如 JPEG, PNG)
    if decompress:
        # 解压图像文件
        image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读取为彩色图像
        if image_data is None:
            raise ValueError(f"Failed to decompress the image at {image_path}.")
        return image_data
    else:
        # 假设图像已经是正确的格式，直接读取
        image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_data is None:
            raise ValueError(f"Failed to read the image at {image_path}.")
        return image_data

def get_top_right_image(file_path, image_extension):
    image_directory = file_path  # 图像文件夹路径
    right_image = "camera_right_wrist"  # 图像文件名前缀
    top_image = "camera_top"
    # image_extension = ".jpg"  # 图像扩展名
    # num_images = 137  # 图像数量
    top__ = get_image_from_folder(image_directory, top_image, image_extension)
    right__ = get_image_from_folder(image_directory, right_image, image_extension)
    return top__, right__
def save_hdf5(file_path, joints_nums, episode_idx, data_dict, reshape_hdf5_path):
    # 获取最大时间步数

    max_timesteps = min(len(get_image_paths(file_path, "camera_right_wrist", ".jpg")), len(get_image_paths(file_path, "camera_top", ".jpg")), len(data_dict['/observations/qpos']), len(data_dict['/action']))
    # print(max_timesteps)
    # 确保目录存在
    # reshape_hdf5_path = os.path.join(DATA_DIR, "reshape_hdf5")
    os.makedirs(reshape_hdf5_path, exist_ok=True)

    # 设置 HDF5 文件路径
    dataset_path = os.path.join(reshape_hdf5_path, f'episode_{episode_idx}')

    # 保存 HDF5 文件
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True

        # 创建 observations group
        obs = root.create_group('observations')
        image = obs.create_group('images')

        # 创建各个相机数据集
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3), compression='gzip', compression_opts=2)

        # 创建 qpos 和 action 数据集
        qpos = obs.create_dataset('qpos', (max_timesteps, joints_nums), dtype='float32')
        action = root.create_dataset('action', (max_timesteps, joints_nums), dtype='float32')

        # 保存 data_dict 中的数据
        for name, array in data_dict.items():
            # print(data_dict.items).
            if name not in root:
                root.create_dataset(name, data=array)
            else:
                # print(f"{name} already exists.\n{array}\n")
                root[name][...] = array  # 更新数据

        print(f"Data saved to {dataset_path}.hdf5")


def get_image_from_folder(image_directory, image_prefix, image_extension):
    # 示例：获取 137 张图像的路径

    return_image_path = []
    # 获取所有图像路径
    image_paths = get_image_paths(image_directory, image_prefix, image_extension)
    for image_path in image_paths:
        i = read_image(image_path)
        return_image_path.append(i)

    return return_image_path


def batch_save_hdf5():
    for i in range(20):
        if i > 8:
            image_directory = f"F:\\origin_data\\11_27\\{i + 1}"  # 图像文件夹路径
        else:
            image_directory = f"F:\\origin_data\\11_27\\0{i + 1}"  # 图像文件夹路径
        right_image = "camera_right_wrist"  # 图像文件名前缀
        top_image = "camera_top"
        image_extension = ".jpg"  # 图像扩展名
        # num_images = 137  # 图像数量
        # max_timesteps = min(len(get_image_paths(image_directory, "camera_right_wrist", ".jpg")), len(get_image_paths(image_directory, "camera_top", ".jpg")))

        top__ = get_image_from_folder(image_directory, top_image, image_extension)
        # print(len(top__[:max_timesteps]))
        right__ = get_image_from_folder(image_directory, right_image, image_extension)
        camera_top_data, camera_right_data, qpos_list, action_ = get_state(f"F:\\hdf5_file\\save_dir\\origin" + f'\\episode_{i}.hdf5')
        # print(action_.shape)

        qpos_list = np.vstack([np.zeros((2, 7)), qpos_list])
        action_ = np.vstack([action_, np.zeros((1, 7))])
        max_timesteps = min(len(get_image_paths(image_directory, "camera_right_wrist", ".jpg")), len(get_image_paths(image_directory, "camera_top", ".jpg")), len(qpos_list), len(action_))
        # print(max_timesteps, action_.shape)
        # 获取所有图像路径
        data_dict = {
            '/observations/qpos': qpos_list[:max_timesteps],
            # 'observations/qvel':[],
            '/observations/images/top': top__[:max_timesteps],
            '/observations/images/right_wrist': right__[:max_timesteps],
            '/action': action_[:max_timesteps],
        }

        print(f"qpos_shape: {len(data_dict['/observations/qpos'])}\ntop_shape: {len(data_dict['/observations/images/top'])}\nright_shape: {len(data_dict['/observations/images/right_wrist'])}\naction_shape: {len(data_dict['/action'])}")
        save_hdf5(image_directory, 7, i, data_dict, reshape_hdf5_path=DATA_DIR+'\\reshape_hdf5_qpos_2')


if __name__ == '__main__':
    # truncate_ranges = {
    #     'top': (45, 100),
    #     'action': (45, 100),
    #     'right_wrist': (45, 100),
    #     'qpos': (45, 100),
    # }
    modify_hdf5('/home/zhnh/Documents/project/act_arm_project/3_cam_1.2/episode_0.hdf5', compress=False)
    # batch_modify_hdf5(dataset_dir, output_dir, skip_mirrored_data=True)
    # 保存视频
    # for i in range(32,53):
    save_video(r'/home/zhnh/Documents/project/act_arm_project/3_cam_1.2', fps=30, i=0,arm='right')
    #
    # image_directory = r"F:\origin_data\\11_27\\01"  # 图像文件夹路径
    # right_image = "camera_right_wrist"  # 图像文件名前缀
    # top_image = "camera_top"
    # image_extension = ".jpg"  # 图像扩展名
    # # num_images = 137  # 图像数量
    # max_timesteps = min(len(get_image_paths(image_directory, "camera_right_wrist", ".jpg")), len(get_image_paths(image_directory, "camera_top", ".jpg")))
    #
    # top__ = get_image_from_folder(image_directory, top_image, image_extension)
    # # print(len(top__[:max_timesteps]))
    # right__ = get_image_from_folder(image_directory, right_image, image_extension)
    # camera_top_data, camera_right_data, qpos_list, action_ = get_state(f"F:\hdf5_file\save_dir\origin" + '\\episode_0.hdf5')
    # # 获取所有图像路径
    # data_dict = {
    #     '/observations/qpos': qpos_list,
    #     '/observations/images/top': top__[:max_timesteps],
    #     '/observations/images/right_wrist': right__[:max_timesteps],
    #     '/action': action_,
    # }
    # save_hdf5(image_directory, 7, 0, data_dict)
    # print(i.shape)
    # print(image_paths)
    # batch_save_hdf5()

    # _, _, qpos, actions = get_state(DATA_DIR + "/save_dir/episode_8.hdf5")
    # # 假设 actions 是一个二维列表或数组
    # actions = [i - 2 for i in actions]
    #
    # # 打印修改前的第三列数据
    # # print(len([i[2] for i in actions]))
    #
    # # 对每个数据的第三位取反
    # actions = [[*i[:2], -i[2], *i[3:]] for i in actions]
    #
    # # 打印修改后的第三列数据
    # # print([i[2] for i in actions])
    # # 将数据传入 visualize 函数
    # qpos, actions =  save_hdf5_content()
    # visualize_episodes.visualize_joints(qpos, actions, DATA_DIR + '/temp.png',)
