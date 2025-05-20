import copy
import random
from visualize_episodes import visualize_joints
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import h5py
import shutil
import os
import datetime
import cv2
import os
import fnmatch
import numpy as np
import traceback
from constants import RIGHT_ARM_TASK_CONFIGS, HDF5_DIR, DATA_DIR
camera_names = RIGHT_ARM_TASK_CONFIGS['train']['camera_names']
max_timesteps = 0
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
STATE_NAMES = JOINT_NAMES + ["gripper_pos"]+ ["gripper_force"]
atrrbut = [ 'top', 'right_wrist', 'left_wrist', 'qpos', 'action']
class Modify_hdf5:
    def __init__(self, compress=None, truncate_ranges=None, edit=False):
        self.attributes = []
        self.data_dict = {}
    def find_all_hdf5(self, dataset_dir, skip_mirrored_data):
        hdf5_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for filename in fnmatch.filter(files, '*.hdf5'):
                if 'features' in filename: continue
                if skip_mirrored_data and 'mirror' in filename:
                    continue
                hdf5_files.append(os.path.join(root, filename))
        print(f'Found {len(hdf5_files)} hdf5 files')
        return hdf5_files
    def check_hdf5(self, file_path):
        if not os.path.exists(file_path):
            print("文件不存在")
        try:
            with h5py.File(file_path, 'r') as f:
                compressed = f.attrs.get('compress', False)
                # print(compressed)
                
                data = self.get_key(file_path)
                for attr in atrrbut:
                    matched_key = next((key for key in data if attr in key), None)
                    if matched_key:
                        self.data_dict[attr] = f[matched_key][:]
                    else:
                        print(f"Warning: {attr} not in HDF5 file")
        except Exception as e:
            print(f"Error load hdf5 file:\n {e}")
            traceback.print_exc()
        return self.data_dict
    def get_key(self,file_path):
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
    def get_state(self,file_path):
        try:
            with h5py.File(file_path, 'r') as f:
                compressed = f.attrs.get('compress', False)
                # print(compressed)
                data = self.get_key(file_path)
                # top_paths = [path for path in data if 'top' in path]            # 检查是否存在原始的路径
                top_paths = [path for path in data if 'top' in path or 'high' in path]
                right_paths = [path for path in data if 'right_wrist' in path]
                left_paths = [path for path in data if 'left_wrist' in path]
                print(f"left_paths:{left_paths}")
                qpos_paths = [path for path in data if 'qpos' in path]
                action_paths = [path for path in data if 'action' in path]
                # print(top)
                # 确保每个路径至少有一个匹配项
                if not top_paths or not right_paths or not qpos_paths or not action_paths:
                    raise KeyError(f"Paths not found in the HDF5 file.")

                # 获取路径
                if len(top_paths) >= 1:
                    top_paths = top_paths[0]  # 如果有多个 'top'，只取第一个
                if len(right_paths) >= 1:
                    right_paths = right_paths[0]  # 如果有多个 'right'，只取第一个
                if len(left_paths) >= 1:
                    left_paths = left_paths[0]  # 如果有多个 'qpos'，只取第一个
                if len(action_paths) >= 1:
                    action_paths = action_paths[0]  # 如果有多个 'action'，只取第一个
                if len(qpos_paths) >= 1:
                    qpos_paths = qpos_paths[0]
                
                camera_top_data = f[top_paths][:]
                camera_right_data = f[right_paths][:]
                camera_left_data = f[left_paths][:]
                qpos = f[qpos_paths][:]
                action = f[action_paths][:]
                camera_top_data_list = []  # 用于存储解压后的帧
                camera_right_data_list = []
                camera_left_data_list = []
                if compressed:
                    num_images = camera_top_data.shape[0]

                    for i in range(num_images):
                        compressed_image = camera_top_data[i]
                        compressed_image_ = camera_right_data[i]
                        compressed_image__ = camera_left_data[i]

                        # 解压为彩色图像
                        decompressed_image = cv2.imdecode(compressed_image, 1)
                        decompressed_image_ = cv2.imdecode(compressed_image_, 1)
                        decompressed_image__ = cv2.imdecode(compressed_image__, 1)

                        # 确保通道顺序是 BGR
                        # decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_RGB2BGR)
                        # image_list.append(decompressed_image)
                        camera_top_data_list.append(decompressed_image)
                        camera_right_data_list.append(decompressed_image_)
                        camera_left_data_list.append(decompressed_image__)

                else:
                    camera_top_data_list = camera_top_data
                    camera_right_data_list = camera_right_data
                    camera_left_data_list = camera_left_data

        except Exception as e:
            print(f"Error load hdf5 file:\n {e}")
            traceback.print_exc()
        return camera_top_data_list, camera_right_data_list, camera_left_data_list, qpos, action


    #用于修改hdf5文件，再写入文件
    # def modify_hdf5(self,file_path, compress=None, truncate_ranges=None, edit=False,exposure_factor=1):
    #     """
    #     修改 HDF5 文件中的摄像头数据，并在指定位置截断数据。

    #     将 `observations/images/camera_top` 修改为：
    #     - `camera_top`
    #     - `left_wrist`
    #     - `right_wrist`

    #     参数:
    #         file_path (str): HDF5 文件的路径。
    #         compress (bool): 是否设置压缩标志。
    #         truncate_ranges (dict): 各数据的截断范围。格式为 {'camera_top': (start, end), 'actions': (start, end)}。
    #     """
    #     try:
    #         with h5py.File(file_path, 'r+') as f:
    #             path_collection = []
    #             # 检查是否存在原始的路径
    #             data = self.get_key(file_path)
    #             for p in data:
    #                 print(f"已经存在的路径",p)
    #             # top_paths = [path for path in data if 'top' in path]            # 检查是否存在原始的路径
    #             top_paths = [path for path in data if 'top' in path or 'high' in path]
    #             right_paths = [path for path in data if 'right' in path]
    #             left_paths = [path for path in data if 'left' in path]
    #             qpos_paths = [path for path in data if 'qpos' in path]
    #             action_paths = [path for path in data if 'action' in path]
    #             # print(top)
    #             # 确保每个路径至少有一个匹配项
    #             if not top_paths or not right_paths or not qpos_paths or not action_paths:
    #                 raise KeyError(f"Paths not found in the HDF5 file.")

    #             # 获取路径
    #             top = top_paths[0]  # 如果有多个 'top'，只取第一个
    #             right = right_paths[0]  # 如果有多个 'right'，只取第一个
    #             qpos = qpos_paths[0]  # 如果有多个 'qpos'，只取第一个
    #             action = action_paths[0]  # 如果有多个 'action'，只取第一个
    #             left = left_paths[0]
    #             if top not in f:
    #                 raise KeyError(f"Path '{top}' not found in the HDF5 file.")
    #             if right not in f:
    #                 raise KeyError(f"Path '{right}' not found in the HDF5 file.")
    #             if left not in f:
    #                 raise KeyError(f"Path '{left}' not found in the HDF5 file.")
    #             if qpos not in f:
    #                 raise KeyError(f"Path '{qpos}' not found in the HDF5 file.")
    #             if action not in f:
    #                 raise KeyError(f"Path '{action}' not found in the HDF5 file.")

    #             # 获取原始数据
    #             camera_top_data = f[top][:]
    #             camera_right_data = f[right][:]
    #             camera_left_data = f[left][:]
    #             qpos = f[qpos][:]
    #             actions = f[action][:]
    #             # print(len(qpos[0]))
    #             # print(qpos[0])
    #             # print(f"camera_top_data.shape: {camera_top_data.shape}, camera_right_data.shape: {camera_right_data.shape}")
    #             if edit:
    #                 # 解压逻辑
    #                 def decompress_images(compressed_data):
    #                     """解压图像数据"""
    #                     image_list = []
    #                     num_images = compressed_data.shape[0]
    #                     for i in range(num_images):
    #                         compressed_image = compressed_data[i]
    #                         # 解压为彩色图像
    #                         decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    #                         if decompressed_image is None:
    #                             raise ValueError("Failed to decompress image. Data might not be valid compressed format.")
    #                         image_list.append(decompressed_image)

    #                     return np.array(image_list)

    #                     # 判断是否需要解压，如果图像维度是 (num_images, 480, 640, 3)，则不进行解压

    #                 if camera_right_data.shape[1:] != (480, 640, 3):
    #                     if compress:
    #                         camera_right_data = decompress_images(camera_right_data)
    #                         print(camera_right_data.shape)
    #                 else:
    #                     print(
    #                         f"\ncamera_right_data is already in the correct shape, skipping decompression.shape{camera_right_data.shape}")
    #                 if camera_top_data.shape[1:] != (480, 640, 3):
    #                     if compress:
    #                         camera_top_data = decompress_images(camera_top_data)
    #                         print(camera_top_data.shape)
    #                 else:
    #                     print(
    #                         f"\ncamera_top_data is already in the correct shape, skipping decompression.shape{camera_top_data.shape}")
    #                 if camera_left_data.shape[1:] != (480, 640, 3):
    #                     if compress:
    #                         camera_left_data = decompress_images(camera_left_data)
    #                         print(camera_left_data.shape)
    #                 else:
    #                     print(
    #                         f"\ncamera_left_data is already in the correct shape, skipping decompression.shape{camera_left_data.shape}")

    #                 if camera_top_data.shape[1:] == (480, 640, 3) and camera_right_data.shape[1:] == (480, 640, 3):
    #                     f.attrs['compress'] = False
    #                 # camera_top_data = decompress_images(camera_top_data)
    #                 # qpos = qpos[:, :7]
    #                 # actions = actions[:, :7]

    #                 # 截断数据，如果指定了截断范围
    #                 def truncate_data(data, key):
    #                     """根据指定的截断范围进行截断"""
    #                     if truncate_ranges and key in truncate_ranges:
    #                         start, end = truncate_ranges[key]
    #                         print(f"Truncating {key} from {start} to {end}")
    #                         return data[start:end]
    #                     return data

    #                 camera_top_data = truncate_data(camera_top_data, 'top')
    #                 camera_right_data = truncate_data(camera_right_data, 'right_wrist')
    #                 camera_left_data = truncate_data(camera_left_data, 'left_wrist')
    #                 qpos = truncate_data(qpos, 'qpos')
    #                 actions = truncate_data(actions, 'action')

    #                 # 创建新的路径并写入数据
    #                 new_paths_top = ['observations/images/top']
    #                 new_paths_right = ['observations/images/right_wrist']
    #                 new_paths_left = ['observations/images/left_wrist']
    #                 new_qpos_path = ['observations/qpos']
    #                 new_actions_path = ['action']

    #                 for path in new_actions_path:
    #                     if path in f:
    #                         del f[path]
    #                     f.create_dataset(path, data=actions)

    #                 for path in new_qpos_path:
    #                     if path in f:
    #                         del f[path]
    #                     f.create_dataset(path, data=qpos)

    #                 for path in new_paths_top:
    #                     if path in f:
    #                         del f[path]
    #                     f.create_dataset(path, data=camera_top_data)

    #                 for path in new_paths_right:
    #                     if path in f:
    #                         del f[path]
    #                     f.create_dataset(path, data=camera_right_data)
    #                 for path in new_paths_left:
    #                     if path in f:
    #                         del f[path]
    #                     f.create_dataset(path, data=camera_left_data)

    #                 print("Modification complete. Paths updated:")
    #                 for path in new_paths_top:
    #                     print(f"  - {path}")
    #                 for path in new_paths_right:
    #                     print(f"  - {path}")
    #                 for path in new_paths_left:
    #                     print(f"  - {path}")
    #                 for path in new_qpos_path:
    #                     print(f"  - {path}")
    #                 for path in new_actions_path:
    #                     print(f"  - {path}")
   
    #             if exposure_factor:
    #                 # 遍历每组相机数据
    #                 for cam_data in [camera_left_data, camera_top_data, camera_right_data]:
    #                     for i, img in enumerate(cam_data):
    #                         # 曝光调整
    #                         enhanced = np.clip(img * exposure_factor, 0, 255).astype(np.uint8)
    #                         # 替换原列表中的图像
    #                         cam_data[i] = enhanced
    #             # 创建新的路径并写入数据
    #             new_paths_top = ['observations/images/top']
    #             new_paths_right = ['observations/images/right_wrist']
    #             new_paths_left = ['observations/images/left_wrist']
    #             new_qpos_path = ['observations/qpos']
    #             new_actions_path = ['action']

    #             for path in new_actions_path:
    #                 if path in f:
    #                     del f[path]
    #                 f.create_dataset(path, data=actions)

    #             for path in new_qpos_path:
    #                 if path in f:
    #                     del f[path]
    #                 f.create_dataset(path, data=qpos)

    #             for path in new_paths_top:
    #                 if path in f:
    #                     del f[path]
    #                 f.create_dataset(path, data=camera_top_data)

    #             for path in new_paths_right:
    #                 if path in f:
    #                     del f[path]
    #                 f.create_dataset(path, data=camera_right_data)
    #             for path in new_paths_left:
    #                 if path in f:
    #                     del f[path]
    #                 f.create_dataset(path, data=camera_left_data)

    #             print("Modification complete. Paths updated:")
    #             for path in new_paths_top:
    #                 print(f"  - {path}")
    #             for path in new_paths_right:
    #                 print(f"  - {path}")
    #             for path in new_paths_left:
    #                 print(f"  - {path}")
    #             for path in new_qpos_path:
    #                 print(f"  - {path}")
    #             for path in new_actions_path:
    #                 print(f"  - {path}")
    #             print(f'compress:', f.attrs.get('compress'))
    #     except Exception as e:
    #         print(f"Error modifying HDF5 file:\n {e}")


    def rename_modified_hdf5_files(self, directory, start_index=51):
        # 获取目录下所有带 "_modified_" 的 .hdf5 文件
        files = [f for f in os.listdir(directory) if f.endswith('.hdf5') and '_modified_' in f]
        files.sort()  # 排序以保证顺序一致

        index = start_index
        for filename in files:
            new_name = f'episode_{index}.hdf5'
            new_path = os.path.join(directory, new_name)

            # 如果新文件名已存在，跳过
            if os.path.exists(new_path):
                print(f'文件 {new_name} 已存在，跳过重命名 {filename}')
            else:
                old_path = os.path.join(directory, filename)
                os.rename(old_path, new_path)
                print(f'✅ 已将 {filename} 重命名为 {new_name}')
                index += 1
    def gray_image(self,color_images):
        '''Step 2：扩展成 (95, 480, 640, 3)，复制三次'''
        gray_images = (
                    0.299 * color_images[:, :, :, 0] +
                    0.587 * color_images[:, :, :, 1] +
                    0.114 * color_images[:, :, :, 2]
                ).astype(np.uint8)  # (95, 480, 640)

                
        gray_images_3ch = np.stack([gray_images]*3, axis=-1)
        return gray_images_3ch
    def overwrite_img(self,img,top, bottom, left, right):
        # 参数：四边的黑边宽度（单位：像素）

        # 获取图像尺寸
        h, w = img.shape[:2]

        # 涂黑四边
        img[0:top, :] = 0                     # 上
        img[h-bottom:h, :] = 0               # 下
        img[:, 0:left] = 0                   # 左
        img[:, w-right:w] = 0                # 右
        return img
    def modify_hdf5(self, file_path, compress=None, truncate_ranges=None, edit=False, exposure_factor=1, save_as_new_file=False,arm = 'arm',gray = False,one_arm=False,save_img=False):
        """
        修改 HDF5 文件中的摄像头数据，可进行解压、曝光调整、截断，并保存为新文件或覆盖原文件。

        参数:
            file_path (str): HDF5 文件的路径。
            compress (bool): 是否进行图像解压（仅当图像是压缩格式时有效）。
            truncate_ranges (dict): 各数据的截断范围，如 {'camera_top': (start, end)}。
            edit (bool): 是否执行编辑操作。
            exposure_factor (float): 曝光因子，默认 1 表示不调整。
            save_as_new_file (bool): 是否保存为新的 HDF5 文件，避免修改原始文件。
        """
        try:
            # 若保存为新文件，先复制原始文件

            if save_as_new_file:
                # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

                # 取上一级目录
                parent_dir = os.path.dirname(file_path)
                print(parent_dir)
                # 在上一级目录下新建文件夹
                new_dir = os.path.join(parent_dir, f'{parent_dir}_overwrited')
                os.makedirs(new_dir, exist_ok=True)

                # 文件名不变，复制到新目录
                new_file_path = os.path.join(new_dir, os.path.basename(file_path))
                shutil.copy(file_path, new_file_path)

                print(f"[INFO] 原始文件复制到上一级目录的新文件夹: {new_file_path}")

                file_path = new_file_path  # 更新 file_path 指向新的位置
            with h5py.File(file_path, 'r+') as f:
                data = self.get_key(file_path)
                print(data)
                arm_path = [path for path in data if arm in path ]
                # 获取路径
                top_paths = [path for path in data if 'top' in path or 'high' in path]
                right_paths = [path for path in data if 'right' in path]
                left_paths = [path for path in data if 'left' in path]
                qpos_paths = [path for path in data if 'qpos' in path]
                action_paths = [path for path in data if 'action' in path]

                if not top_paths or not right_paths or not qpos_paths or not action_paths:
                    raise KeyError("缺少关键路径。")

                top = top_paths[0]
                right = right_paths[0]
                left = left_paths[0]
                qpos_key = qpos_paths[0]
                action_key = action_paths[0]
                # print(qpos_key)
                # 加载数据（可选择 copy）
                if save_as_new_file:
                    camera_top_data = f[top][:].copy()
                    camera_right_data = f[right][:].copy()
                    camera_left_data = f[left][:].copy()
                    qpos = f[qpos_key][:].copy()
                    actions = f[action_key][:].copy()
                else:
                    camera_top_data = f[top][:]
                    camera_right_data = f[right][:]
                    camera_left_data = f[left][:]
                    if one_arm:
                        qpos = f[qpos_key][:,:8]
                        actions = f[action_key][:,:8]
                    else:
                        qpos = f[qpos_key][:]
                        actions = f[action_key][:]
                # qpos[:,:6] = np.degrees(qpos[:,:6])
                # actions[:,:6] = np.degrees(actions[:,:6])
                # qpos[:,9:14] = np.degrees(qpos[:,8:14])
                # actions[:,9:14] = np.degrees(actions[:,8:14])
                # qpos[:,8] = np.deg2rad(qpos[:,8])
                # actions[:,8] = np.deg2rad(actions[:,8])
                # print(qpos)
                camera_top_data_np = np.array(camera_top_data)
                camera_right_data_np = np.array(camera_right_data)
                camera_left_data_np = np.array(camera_left_data)
                qpos_np = np.array(qpos)
                actions_np = np.array(actions)
                len_ = camera_top_data_np.shape[0]

                # print(qpos_np.shape,qpos_np[:3,:])
                if gray:
                    camera_top_data_np_gray = self.gray_image(camera_top_data_np)
                    camera_right_data_np_gray = self.gray_image(camera_right_data_np)
                    camera_left_data_np_gray = self.gray_image(camera_left_data_np)
                    camera_top_data = camera_top_data_np_gray.copy()
                    camera_right_data = camera_right_data_np_gray.copy()
                    camera_left_data = camera_left_data_np_gray.copy()
                # print(camera_top_data_np_gray.shape,camera_right_data_np_gray.shape,camera_left_data_np_gray.shape,qpos_np.shape,actions_np.shape)
                if False:
                    save_dir = 'test_images_png'
                    os.makedirs(save_dir, exist_ok=True)  # 自动创建保存目录

                    # for i in range(gray_images_3ch.shape[0]):
                    img = camera_top_data_np[160].copy()  # 取出第i张 (480, 640, 3)
                    # img = self.overwrite_img(img,200,0,240,130)
                    # img = self.overwrite_img(img,240,0,0,0)
                    # 计算图片中点高度
                    # h, w = img.shape[:2]
                    # img[:h//2, :] = 0  # 将上半部分设为黑色

                    save_path = os.path.join(save_dir, f'image.png')  # 文件名自动编号
                    cv2.imwrite(save_path, img)
                    print(save_path)
                if True:
                    for camera_name, cam_data in zip(camera_names,[camera_left_data, camera_top_data, camera_right_data]):
                        print('camera_name,',camera_name)
                        for i, img in enumerate(cam_data):
                            if camera_name == "left_wrist":
                                img = self.overwrite_img(img,200,0,240,130)
                                # cv2.imwrite('/workspace/left_image.png', img)
                                # print(cam_data.shape,img.shape,i)
                            elif camera_name == "top":
                                img = self.overwrite_img(img,240,0,0,0)
                                # cv2.imwrite('/workspace/top_image.png', img)
                if edit:
                    def decompress_images(compressed_data):
                        image_list = []
                        for i in range(compressed_data.shape[0]):
                            decompressed_image = cv2.imdecode(compressed_data[i], cv2.IMREAD_COLOR)
                            if decompressed_image is None:
                                raise ValueError("图像解压失败。")
                            image_list.append(decompressed_image)
                        return np.array(image_list)

                    if compress:
                        if camera_right_data.shape[1:] != (480, 640, 3):
                            camera_right_data = decompress_images(camera_right_data)
                        if camera_top_data.shape[1:] != (480, 640, 3):
                            camera_top_data = decompress_images(camera_top_data)
                        if camera_left_data.shape[1:] != (480, 640, 3):
                            camera_left_data = decompress_images(camera_left_data)

                    if camera_top_data.shape[1:] == (480, 640, 3) and camera_right_data.shape[1:] == (480, 640, 3):
                        f.attrs['compress'] = False

                    def truncate_data(data, key):
                        if truncate_ranges and key in truncate_ranges:
                            start, end = truncate_ranges[key]
                            print(f"[INFO] Truncating {key} from {start} to {end}")
                            return data[start:end]
                        return data

                    camera_top_data = truncate_data(camera_top_data, 'top')
                    camera_right_data = truncate_data(camera_right_data, 'right_wrist')
                    camera_left_data = truncate_data(camera_left_data, 'left_wrist')
                    qpos = truncate_data(qpos, 'qpos')
                    actions = truncate_data(actions, 'action')

                # 曝光调整
                if exposure_factor and exposure_factor != 1:
                    for cam_data in [camera_left_data, camera_top_data, camera_right_data]:
                        for i, img in enumerate(cam_data):
                            enhanced = np.clip(img * exposure_factor, 0, 255).astype(np.uint8)
                            cam_data[i] = enhanced

                # 写入路径
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

                print("\n[✅] 修改完成，以下数据路径已更新：")
                for path in new_paths_top + new_paths_right + new_paths_left + new_qpos_path + new_actions_path:
                    print(f"  - {path}")
                print(f"[INFO] compress 属性为: {f.attrs.get('compress')}")

        except Exception as e:
            print(f"[❌] 修改 HDF5 文件出错: {e}")


    def batch_modify_hdf5(self, dataset_dir, output_dir=None, skip_mirrored_data=True):
        """
        批量修改指定目录中的 HDF5 文件的 actions 数据。

        参数:
            dataset_dir (str): 数据集目录路径。
            output_dir (str): 修改后的文件保存路径。如果为 None，则覆盖原文件。
            skip_mirrored_data (bool): 是否跳过包含 "mirror" 的文件。
        """
        hdf5_files = self.find_all_hdf5(dataset_dir, skip_mirrored_data)
        # hdf5_files_ = self.filter_episodes_by_index(
        #     file_paths=hdf5_files,
        #     start=60,
        #     end=80
        #     )
        # print(hdf5_file_)
        for file_path in hdf5_files:
            print(file_path)
            # if output_dir:
            #     # 确保输出目录存在
            #     os.makedirs(output_dir, exist_ok=True)
            #     # 构造新文件路径
            #     output_file_path = os.path.join(output_dir, os.path.basename(file_path))
            # else:
            #     output_file_path = None

            self.modify_hdf5(
                file_path=file_path, 
                compress=False,
                edit=False,
                exposure_factor=1,
                save_as_new_file=True,  # 不影响原始文件
                gray=False,
                one_arm=False
                )
        # self.rename_modified_hdf5_files(dataset_dir,42)

    # rand = random.random()

    def filter_episodes_by_index(self,file_paths, start=10, end=30):
        filtered = []
        for path in file_paths:
            # 提取 episode 编号
            filename = os.path.basename(path)
            if filename.startswith("episode_") and filename.endswith(".hdf5"):
                try:
                    index = int(filename.split("_")[1].split(".")[0])
                    if start <= index <= end:
                        filtered.append(path)
                except ValueError:
                    pass
        return sorted(filtered)



    def save_video(self,file_path, fps=10, i=0, arm='right_wrist',exposure_factor = 0.5):
        dataset_path = os.path.join(file_path, f'episode_{i}' + '.hdf5')
        adjusted_images = []
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
                  # >1 提亮，<1 变暗
                for i, img in enumerate(image_list):
                    enhanced = np.clip(img * exposure_factor, 0, 255).astype(np.uint8)
                    adjusted_images.append(enhanced)
                print(os.path.splitext(os.path.basename(dataset_path))[0])
                output_path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}_"+arm+".jpg")

                # for i in range(len(image_list)):
                # print(f"image_list_len:{len(image_list)}")
                # output_path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}-{i}.jpg")
                cv2.imwrite(output_path, adjusted_images[0][:, :, [0, 1, 2]])
                # 如果图像列表为空，抛出错误
                if not adjusted_images:
                    raise ValueError("No images found to save as video.")

                # 获取帧的宽度和高度
                frame_height, frame_width, _ = adjusted_images[0].shape

                # 定义视频写入器

                path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}_"+arm+".mp4")
                video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

                # 将每一帧写入视频
                for frame in adjusted_images:
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
    def save_arm_video(self,file_path, fps=10, i=0, arm_path:str = None,exposure_factor = 0.5):
        dataset_path = os.path.join(file_path, f'episode_{i}' + '.hdf5')
        adjusted_images = []
       
        with h5py.File(dataset_path, 'r') as f:
            # print(f.keys())
            key = self.get_key(dataset_path)
            # arm_path = 'observations/images/right_wrist'
            if arm_path not in key:
                raise KeyError(f"Path '{arm_path}' not found in the HDF5 file.\nvalue key{key}")
        
            arm_path_data = f[arm_path][()]  # 读取图像数据

            compressed = f.attrs.get('compress', False)
            image_list = []  # 用于存储解压后的帧
            if compressed:
                num_images = arm_path_data.shape[0]
                for i in range(num_images):
                    compressed_image = arm_path_data[i]
                    # 解压为彩色图像
                    decompressed_image = cv2.imdecode(compressed_image, 1)
                    # 确保通道顺序是 BGR
                    # decompressed_image = cv2.cvtColor(decompressed_image, cv2.COLOR_RGB2BGR)
                    # image_list.append(decompressed_image)
                    image_list.append(decompressed_image)
            else:
                # 假设数据直接是未压缩图像数组
                image_list = [frame for frame in arm_path_data]
                # >1 提亮，<1 变暗
            for i, img in enumerate(image_list):
                enhanced = np.clip(img * exposure_factor, 0, 255).astype(np.uint8)
                adjusted_images.append(enhanced)
            print(os.path.splitext(os.path.basename(dataset_path))[0])
            arm_path_new = arm_path.replace('/', '_')
            output_path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}_"+arm_path_new+".jpg")
            # print(output_path)
            # for i in range(len(image_list)):
            # print(f"image_list_len:{len(image_list)}")
            # output_path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}-{i}.jpg")
            cv2.imwrite(output_path, adjusted_images[0][:, :, [0, 1, 2]])
            # 如果图像列表为空，抛出错误
            if not adjusted_images:
                raise ValueError("No images found to save as video.")

            # 获取帧的宽度和高度
            frame_height, frame_width, _ = adjusted_images[0].shape

            # 定义视频写入器

            path = os.path.join(file_path, f"frame_{os.path.splitext(os.path.basename(dataset_path))[0]}_"+arm_path_new+".mp4")
            video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            print(path)
            # 将每一帧写入视频
            for frame in adjusted_images:
                # image_list = image_list[:, :, [2, 1, 0]]  # 交换图像的B和R通道
                frame = frame[:, :, [0, 1, 2]]
                video_writer.write(frame)

            # 释放视频写入器
            video_writer.release()
            if os.path.exists(path):
                print(f"\nVideo saved successfully at {file_path}")
            else:
                raise "error to save video"

    def visual_qpos_action(self,file_path):

        with h5py.File(file_path, 'r') as f:
            key = self.get_key(file_path)
            
            # 数据路径
            qpos_path = 'observations/qpos'
            action_path = 'action'
            
            # 提取数据
            qpos_path_data = f[qpos_path][()]  
            action_path_data = f[action_path][()]
            
            # 获取文件所在的根目录
            root_dir = os.path.dirname(file_path)
            
            # 获取文件名（去除扩展名）
            file_stem = os.path.splitext(os.path.basename(file_path))[0]
            
            # 构造图片保存路径
            image_path = os.path.join(root_dir, file_stem + ".png")
            
            print(image_path)
            print(qpos_path_data[70])
            visualize_joints(qpos_path_data, action_path_data, image_path, STATE_NAMES=STATE_NAMES)


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
    #     'top': (0, 558),
    #     'action': (0, 558),
    #     'right_wrist': (0, 558),
    #     'qpos': (0, 558),
    # }
    test = Modify_hdf5()
    # test.batch_modify_hdf5('/workspace/exchange/5-9/duikong')
    # test.modify_hdf5(
    #     file_path='/workspace/exchange/episode_0.hdf5', 
    #     compress=False,
    #     edit=False,
    #     exposure_factor=1,
    #     save_as_new_file=False,  # 不影响原始文件
    #     gray=False,
    #     one_arm = False,
    #     # truncate_ranges=truncate_ranges
    #     )
    # batch_modify_hdf5(dataset_dir, output_dir, skip_mirrored_data=True)
    # 保存视频
    # for i in range(32,53):
    test.save_arm_video('/workspace/exchange/5-9/duikong_overwrited', fps=10, i=1,arm_path='observations/images/left_wrist',exposure_factor = 1)
    # test.save_video('/workspace/exchange/4-24', fps=10, i=1,arm='left_wrist',exposure_factor = 1)
    # test.visual_qpos_action('/workspace/exchange/5-9/exchange/episode_15.hdf5')
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
