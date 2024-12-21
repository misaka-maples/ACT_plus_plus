import json
import os
import time
from queue import Queue
from threading import Lock
from typing import List
from Robotic_Arm.rm_robot_interface import *
import math, h5py
import cv2
import numpy as np
from pyorbbecsdk import *
from utils import frame_to_bgr_image
from pynput.keyboard import Listener, Key
from policy_action_generation import ActionGenerator
frames_queue_lock = Lock()

# Configuration settings
MAX_DEVICES = 2
MAX_QUEUE_SIZE = 2
ESC_KEY = 27
save_points_dir = os.path.join(os.getcwd(), "point_clouds")
save_depth_image_dir = os.path.join(os.getcwd(), "depth_images")
save_color_image_dir = os.path.join(os.getcwd(), "color_images")

frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
stop_processing = False
curr_device_cnt = 0

# Load config file for multiple devices
config_file_path = os.path.join(os.path.dirname(__file__), "/home/zhnh/Documents/xzx_projects/GraspNet_Pointnet2_PyTorch1.13.1/pyorbbecsdk/config/multi_device_sync_config.json")
multi_device_sync_config = {}
camera_names = ['top', 'right_wrist']


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
posRecorder = QposRecorder()

def read_config(config_file: str):
    global multi_device_sync_config
    with open(config_file, "r") as f:
        config = json.load(f)
    for device in config["devices"]:
        multi_device_sync_config[device["serial_number"]] = device
        print(f"Device {device['serial_number']}: {device['config']['mode']}")

# Frame processing and saving
def process_frames(pipelines):
    global frames_queue
    global stop_processing
    global curr_device_cnt, save_points_dir, save_depth_image_dir, save_color_image_dir
    all_frames_processed = False  # 添加一个标志来指示是否所有帧都已处理
    images = {}
    while not stop_processing and not all_frames_processed:

        now = time.time()
        for device_index in range(curr_device_cnt):
            with frames_queue_lock:
                # 尝试从队列中获取帧，如果队列为空则返回None
                frames = frames_queue[device_index].get() if not frames_queue[device_index].empty() else None
            if frames is None:
                continue  # 如果没有帧，跳过当前循环

            color_frame = frames.get_color_frame() if frames else None
            depth_frame = frames.get_depth_frame() if frames else None

            if color_frame:
                # 将彩色帧转换为BGR图像
                color_image = frame_to_bgr_image(color_frame)
                color_filename = os.path.join(save_color_image_dir,
                                              f"color_{device_index}_{color_frame.get_timestamp()}.png")
                # print(f"Saving {color_filename}")  # 打印保存的文件名
                x=cv2.resize(color_image, (640, 480))
                # cv2.imwrite(color_filename, x)  # 保存彩色图像
                images[device_index] = color_image
                # 将图像添加到列表中
                # images.append(color_image)
                # print(f"Image in loop for device {device_index}: {len(images)}")  # 打印图像的形状
            # print(len(images))
            # 检查当前设备是否还有帧未处理
            # if frames_queue[device_index].empty() and len(images)==2:
            if  len(images) == 2:
                all_frames_processed = True  # 如果队列为空，设置标志为True
                # print(f"quit_process_frames")
        # 打印处理帧所花费的时间
        # print(f"Processing time: {time.time() - now:.3f}s")

    # 函数返回所有设备的所有彩色图像列表
    return images
def sync_mode_from_str(sync_mode_str: str) -> OBMultiDeviceSyncMode:
    sync_mode_str = sync_mode_str.upper()
    if sync_mode_str == "FREE_RUN":
        return OBMultiDeviceSyncMode.FREE_RUN
    elif sync_mode_str == "STANDALONE":
        return OBMultiDeviceSyncMode.STANDALONE
    elif sync_mode_str == "PRIMARY":
        return OBMultiDeviceSyncMode.PRIMARY
    elif sync_mode_str == "SECONDARY":
        return OBMultiDeviceSyncMode.SECONDARY
    elif sync_mode_str == "SECONDARY_SYNCED":
        return OBMultiDeviceSyncMode.SECONDARY_SYNCED
    elif sync_mode_str == "SOFTWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.SOFTWARE_TRIGGERING
    elif sync_mode_str == "HARDWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.HARDWARE_TRIGGERING
    else:
        raise ValueError(f"Invalid sync mode: {sync_mode_str}")


# def on_press(key):
#     """
#     当键盘被按下时调用。
#     :param key: 被按下的键。
#     """
#     if hasattr(key, 'char') and key.char == 'k':
#         # x=QposRecorder()
#         posRecorder.real_right_arm.rm_set_tool_voltage(3)
#         print(posRecorder.real_right_arm.rm_get_tool_voltage()[1])
#         # print(f'Key "K" pressed')
#
# def on_release(key):
#     """
#     当键盘被释放时调用。
#     :param key: 被释放的键。
#     """
#     if hasattr(key, 'char') and key.char == 'k':
#         # x = QposRecorder()
#         posRecorder.real_right_arm.rm_set_tool_voltage(0)
#         print(posRecorder.real_right_arm.rm_get_tool_voltage()[1])
#         print(f'Key "K" released')
#     if key == Key.esc:
#         # 停止监听器
#         return False
#
# def start_listener():
#     """
#     启动键盘监听器的函数。
#     """
#     with Listener(on_press=on_press, on_release=on_release) as listener:
#         listener.join()
#
# # 创建一个线程来运行监听器
# listener_thread = threading.Thread(target=start_listener)
# listener_thread.start()


def start_streams(pipelines: List[Pipeline], configs: List[Config]):
    index = 0
    for pipeline, config in zip(pipelines, configs):
        print(f"Starting device {index}")
        pipeline.start(
            config,
            lambda frame_set, curr_index=index: on_new_frame_callback(
                frame_set, curr_index
            ),
        )
        pipeline.enable_frame_sync()
        index += 1
def stop_streams(pipelines: List[Pipeline]):
    index = 0
    for pipeline in pipelines:
        print(f"Stopping device {index}")
        pipeline.stop()
        index += 1


# def save_hdf5(max_timesteps, joints_nums, episode_idx, data_dict, reshape_hdf5_path):
#     os.makedirs(reshape_hdf5_path, exist_ok=True)
#     dataset_path = os.path.join(reshape_hdf5_path, f'episode_{episode_idx}.hdf5')
#
#     with h5py.File(dataset_path, 'w') as root:
#         root.attrs['sim'] = True
#         obs = root.create_group('observations')
#         images_group = obs.create_group('images')
#
#         # 创建每个相机的数据集并写入数据
#         for cam_name in ['images/top', 'images/right_wrist']:  # 确保只处理相关键
#             if f'/observations/{cam_name}' in data_dict:
#                 images_group.create_dataset(
#                     cam_name.split('/')[-1],  # 使用相机名称作为数据集名称
#                     data=np.array(data_dict[f'/observations/{cam_name}']),
#                     dtype='uint8',
#                     # compression='gzip',
#                     # compression_opts=4
#                 )
#
#         # 写入 qpos 数据
#         if '/observations/qpos' in data_dict:
#             if 'qpos' in obs:
#                 print("Dataset 'qpos' already exists. Updating it.")
#                 del obs['qpos']
#             obs.create_dataset(
#                 'qpos',
#                 data=np.array(data_dict['/observations/qpos']),
#                 dtype='float32'
#             )
#
#         # 写入 action 数据
#         if '/action' in data_dict:
#             if 'action' in root:
#                 print("Dataset 'action' already exists. Updating it.")
#                 del root['action']
#             root.create_dataset(
#                 'action',
#                 data=np.array(data_dict['/action']),
#                 dtype='float32'
#             )
#
#     print(f"Data saved to {dataset_path}")

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
def on_new_frame_callback(frames: FrameSet, index: int):
    global frames_queue
    global MAX_QUEUE_SIZE
    assert index < MAX_DEVICES
    with frames_queue_lock:
        if frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            frames_queue[index].get()
        frames_queue[index].put(frames)


# def save_images_from_dict(data_dict, output_dir):
#     """
#     保存字典中的图像到指定文件夹。
#
#     参数：
#         data_dict (dict): 包含图像数据的字典，键为相机路径，值为图像列表。
#         output_dir (str): 保存图像的根目录。
#     """
#     os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
#
#     # 遍历图像键
#     for key in ['/observations/images/top', '/observations/images/right_wrist']:
#         if key not in data_dict:
#             print(f"Key {key} not found in data_dict.")
#             continue
#
#         cam_name = key.split('/')[-1]  # 提取相机名称
#         cam_dir = os.path.join(output_dir, cam_name)  # 创建相机子目录
#         os.makedirs(cam_dir, exist_ok=True)
#
#         for i, image in enumerate(data_dict[key]):
#             filename = os.path.join(cam_dir, f"color_image_{cam_name}_{i}.png")
#             resized_image = cv2.resize(image, (640, 480))  # 调整到 640x480
#             cv2.imwrite(filename, resized_image)
#             print(f"Saved: {filename}")
def main():
    global curr_device_cnt, max_timesteps, qpos_list, images_dict, QposRecorder
    read_config(config_file_path)
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("No device connected")
        return
    pipelines = []
    configs = []
    curr_device_cnt = device_list.get_count()
    for i in range(min(device_list.get_count(), MAX_DEVICES)):
        device = device_list.get_device_by_index(i)
        pipeline = Pipeline(device)
        config = Config()
        serial_number = device.get_device_info().get_serial_number()
        sync_config_json = multi_device_sync_config[serial_number]
        sync_config = device.get_multi_device_sync_config()
        sync_config.mode = sync_mode_from_str(sync_config_json["config"]["mode"])
        sync_config.color_delay_us = sync_config_json["config"]["color_delay_us"]
        sync_config.depth_delay_us = sync_config_json["config"]["depth_delay_us"]
        sync_config.trigger_out_enable = sync_config_json["config"]["trigger_out_enable"]
        sync_config.trigger_out_delay_us = sync_config_json["config"]["trigger_out_delay_us"]
        sync_config.frames_per_trigger = sync_config_json["config"]["frames_per_trigger"]
        device.set_multi_device_sync_config(sync_config)
        print(f"Device {serial_number} sync config: {sync_config}")

        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        print(f"Device {serial_number} color profile: {color_profile}")

        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = profile_list.get_default_video_stream_profile()
        print(f"Device {serial_number} depth profile: {depth_profile}")
        config.enable_stream(depth_profile)
        pipelines.append(pipeline)
        configs.append(config)
        #可以放外面吗？
        # max_timesteps = 300
        # qpos_list = []
        # images_dict = {cam_name: [] for cam_name in camera_names}  # 用于存储每个相机的图片
        # save_image_dir = os.path.join(os.getcwd(), "color_images")
        # if not os.path.exists(save_image_dir):
        #     os.mkdir(save_image_dir)
        # save_image_dir = os.path.join(os.getcwd(), "color_images_")
        # if not os.path.exists(save_image_dir):
        #     os.mkdir(save_image_dir)
    start_streams(pipelines, configs)
    max_timesteps = 300
    qpos_list = []
    images_dict = {cam_name: [] for cam_name in camera_names}  # 用于存储每个相机的图片
    save_image_dir = os.path.join(os.getcwd(), "color_images")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    save_image_dir = os.path.join(os.getcwd(), "color_images_")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    image = process_frames(pipelines)
    images_dict1 = {cam_name: [] for cam_name in camera_names}
    images_dict1['right_wrist'].append(cv2.resize(image[0], (640, 480)))
    images_dict1['top'].append(cv2.resize(image[1], (640, 480)))
    angle_qpos = posRecorder.get_state()
    radius_qpos = [math.radians(j) for j in angle_qpos]
    radius_qpos[6] = posRecorder.real_right_arm.rm_get_tool_voltage()[1]
    qpos_list = radius_qpos
    actions_list = []
    qpos_list_ = []
    config = {
        'image_dict': images_dict1,
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
        'vq_class': None,
        'vq_dim': None,
        'no_encoder': False,
    }
    ActionGenerator1= ActionGenerator(config)
    global stop_processing
    try:
        # now = time.time()
        for i in range(max_timesteps):
            # 创建并启动监听器
            print(f"episode{i}\n"
                  f"--------------------------------------------------------------------------------------")
            image = process_frames(pipelines)
            # if image == []:
            #     image = process_frames(pipelines)
            # print(image)
            angle_qpos = posRecorder.get_state()
            radius_qpos = [math.radians(j) for j in angle_qpos]
            radius_qpos[6] = posRecorder.real_right_arm.rm_get_tool_voltage()[1]
            print(radius_qpos[6])
            qpos_list.append(radius_qpos)
            images_dict['right_wrist']=(cv2.resize(image[0], (640, 480)))
            images_dict['top']=(cv2.resize(image[1], (640, 480)))
            #filename_ = os.path.join(os.getcwd(), "color_images_", f"color_image_top_{i}.png")
            # cv2.imwrite(filename_,cv2.resize(image[0], (640, 480)))
            # print(f"7 joint :{posRecorder.get_state()[-1]}")
            # print(angle_qpos[1])
            # if angle_qpos[1]>60:
            #     # print(qpos)
            #     posRecorder.real_right_arm.rm_set_tool_voltage(0)
            # else:
            #     posRecorder.real_right_arm.rm_set_tool_voltage(3)
            ActionGenerator.image_dict = images_dict
            ActionGenerator.qpos_list = radius_qpos
            actions = ActionGenerator.get_action()
            actions = [i - 2 for i in actions]
            actions[2] = -actions[2]
            print(f":-----------------------------actions------------------------------------:\n{actions}")
            qpos_list_.append(ActionGenerator.qpos_list)
            actions_list.append(actions)
            power = actions[6]
            actions = [math.degrees(i) for i in actions[:6]]
            posRecorder.real_right_arm.Movej_Cmd(actions, 10, 0, 0, True)
            if power > 2:
                posRecorder.real_right_arm.rm_set_tool_voltage(3)
            else:
                posRecorder.real_right_arm.rm_set_tool_voltage(0)
            # print(len(qpos_list),len(actions_list))
    except KeyboardInterrupt:
        print("Interrupted by user")
        stop_processing = True
    finally:
        # 将图像列表转换为 NumPy 数组
        for cam_name in camera_names:
            images_dict[cam_name] = np.array(images_dict[cam_name])

        # 创建动作列表
        action_list = np.vstack([qpos_list[0], qpos_list])

        # 构建数据字典
        data_dict = {
            '/observations/qpos': qpos_list[:max_timesteps],
            '/action': action_list[:max_timesteps],
        }
        # print(data_dict)
        # 添加图像数据到字典
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = images_dict[cam_name]

        # 确保监听器线程被正确关闭
        # listener_thread.join()
        print("===============Stopping pipelines====")
        stop_streams(pipelines)
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"total time in 1 roll:{end-start}")