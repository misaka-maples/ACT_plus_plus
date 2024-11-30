import random

import h5py
import cv2
import os
import fnmatch
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
def get_state(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            compressed = f.attrs.get('compress', False)
            # 检查是否存在原始的路径
            top = 'observations/images/top'
            right = 'observations/images/right_wrist'
            qpos = 'observations/qpos'
            if top not in f:
                raise KeyError(f"Path '{top}' not found in the HDF5 file.")
            if right not in f:
                raise KeyError(f"Path '{right}' not found in the HDF5 file.")
            if qpos not in f:
                raise KeyError(f"Path '{qpos}' not found in the HDF5 file.")
            camera_top_data = f[top][:]
            camera_right_data = f[right][:]
            qpos = f[qpos][:]
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
    except Exception as e:
        print(f"Error saving video:\n {e}")
    return camera_top_data_list, camera_right_data_list, qpos


def modify_hdf5(file_path):
    """
    修改 HDF5 文件中的摄像头数据。

    将 `observations/images/camera_top` 修改为：
    - `camera_top`
    - `left_wrist`
    - `right_wrist`

    参数:
        file_path (str): HDF5 文件的路径。
    """
    try:
        with h5py.File(file_path, 'r+') as f:
            # 检查是否存在原始的路径
            top = 'observations/images/top'
            right = 'observations/images/right_wrist'
            if top not in f:
                raise KeyError(f"Path '{top}' not found in the HDF5 file.")
            original_path_pos = 'observations/qpos'
            if original_path_pos not in f:
                raise KeyError(f"Path '{original_path_pos}' not found in the HDF5 file.")
            original_path_actions = 'action'
            if original_path_pos not in f:
                raise KeyError(f"Path '{original_path_actions}' not found in the HDF5 file.")
            # 获取原始数据
            camera_top_data = f[top][:]
            camera_right_data = f[right][:]
            qpos = f[original_path_pos][:]
            actions = f[original_path_actions][:]
            print(f"camera_top_data.shape{camera_top_data.shape},camera_right_data.shape{camera_right_data.shape}")
            print(f'hdf5_edit_qpos: {qpos.shape}')
            qpos = qpos[:, :7]
            actions = actions[:, :7]-2
            last_elements = [row[-1] for row in actions]
            print(f'hdf5_edit_action: {last_elements}')
            # 创建新的路径并写入数据
            new_paths = [
                'observations/images/top',
                'observations/images/right_wrist',
                # 'observations/images/right_wrist'
            ]
            new_qpos_path = [
                'observations/qpos',
            ]
            new_actions_path = [
                'action'
            ]
            for path in new_actions_path:
                # 如果路径已存在，删除旧的路径
                if path in f:
                    del f[path]
                # 写入新的数据
                f.create_dataset(path, data=actions)
            for path in new_qpos_path:
                # 如果路径已存在，删除旧的路径
                if path in f:
                    del f[path]
                # 写入新的数据
                f.create_dataset(path, data=qpos)
            for path in new_paths:
                # 如果路径已存在，删除旧的路径
                if path in f:
                    del f[path]
                # 写入新的数据
                f.create_dataset(path, data=camera_top_data)

            print("Modification complete. Paths updated:")
            for path in new_paths:
                print(f"  - {path}")
            for path in new_qpos_path:
                print(f"  - {path}")
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


def save_video(file_path, fps=10, i=0):
    dataset_path = os.path.join(file_path, f'episode_{i}' + '.hdf5')
    try:
        with h5py.File(dataset_path, 'r') as f:
            original_path = 'observations/images/right_wrist'
            if original_path not in f:
                raise KeyError(f"Path '{original_path}' not found in the HDF5 file.")
            original_path_top = 'observations/images/top'
            if original_path_top not in f:
                raise KeyError(f"Path '{original_path_top}' not found in the HDF5 file.")

            right_wrist_data = f[original_path][()]  # 读取图像数据
            top_data = f[original_path_top][()]  # 读取图像数据
            compressed = f.attrs.get('compress', False)

            image_list = []  # 用于存储解压后的帧
            if compressed:
                num_images = top_data.shape[0]
                for i in range(num_images):
                    compressed_image = top_data[i]
                    # 解压为彩色图像
                    decompressed_image = cv2.imdecode(compressed_image, 1)
                    # 确保通道顺序是 BGR
                    # decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_RGB2BGR)
                    # image_list.append(decompressed_image)
                    image_list.append(decompressed_image)

            else:
                # 假设数据直接是未压缩图像数组
                image_list = [frame for frame in right_wrist_data]
            print(os.path.splitext(os.path.basename(dataset_path))[0])
            output_path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}.jpg")
            cv2.imwrite(output_path, image_list[0][:, :, [2, 1, 0]])
            # 如果图像列表为空，抛出错误
            if not image_list:
                raise ValueError("No images found to save as video.")

            # 获取帧的宽度和高度
            frame_height, frame_width, _ = image_list[0].shape

            # 定义视频写入器

            path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}.mp4")
            video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            # 将每一帧写入视频
            for frame in image_list:
                # image_list = image_list[:, :, [2, 1, 0]]  # 交换图像的B和R通道
                frame = frame[:, :, [2, 1, 0]]
                video_writer.write(frame)

            # 释放视频写入器
            video_writer.release()
            print(f"\nVideo saved successfully at {file_path}")

    except Exception as e:
        print(f"Error saving video:\n {e}")


if __name__ == '__main__':
    modify_hdf5('D:\\aloha\qpos_7_image_2\\act++\is_sim_0_compress_1_real\episode_0.hdf5')
    # batch_modify_hdf5(dataset_dir, output_dir, skip_mirrored_data=True)
    # 保存视频
    # save_video('D:\\aloha\qpos_7_image_2\\act++\is_sim_0_compress_1_real', fps=10, i=19)
