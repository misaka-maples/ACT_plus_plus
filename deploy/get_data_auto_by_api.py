import json
import os
import threading
import time
from queue import Queue
from threading import Lock
from typing import List
from Robotic_Arm.rm_robot_interface import *
import math, h5py
import cv2
import numpy as np
from tqdm import tqdm
from pyorbbecsdk import *

from deploy.remote_control import posRecorder
# from utils import frame_to_bgr_image
# from pynput.keyboard import Listener, Key
from pynput import keyboard
import random
frames_queue_lock = Lock()

# Configuration settings
MAX_DEVICES = 3
MAX_QUEUE_SIZE = 2
ESC_KEY = 27
save_points_dir = os.path.join(os.getcwd(), "point_clouds")
save_depth_image_dir = os.path.join(os.getcwd(), "depth_images")
save_color_image_dir = os.path.join(os.getcwd(), "color_images")

frames_queue: List[Queue] = [Queue() for _ in range(MAX_DEVICES)]
stop_processing = False
curr_device_cnt = 0

# Load config file for multiple devices
config_file_path = os.path.join(os.path.dirname(__file__), "/home/zhnh/Documents/project/act_arm_project/pyorbbecsdk-main/config/multi_device_sync_config.json")
multi_device_sync_config = {}
camera_names = ['top', 'right_wrist','left_wrist']
#是位置姿态不是关节值
zero_pos = [0, 0, 0.8505, 0, 0, 3.14]
original_pos = [-0.344476, 0.146943, 0.619547, 1.722, 1.129, -1.437]
final_pos = [-0.344879, 0.342856, 0.24035, 2.725, 0.781, -0.658]
standard_final_pos = [-0.269835, 0.342173, 0.464594, 2.105, 1.046, -1.163]

class QposRecorder:
    def __init__(self):
        self.joint_state_right=None
        self.real_right_arm = (RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E))
        self.arm_ini = self.real_right_arm.rm_create_robot_arm("192.168.1.18",8080, level=3)
        # self.robot_controller = RoboticArm("192.168.1.18", 8080, 3)
    def get_state(self, model='joint'):
        self.joint_state_right = self.real_right_arm.rm_get_current_arm_state()
        # print(f"get state test", self.joint_state_right)
        return_action = self.joint_state_right[1][model]
        # print(return_action)
        return return_action

#
# key_state = False  # 按键状态：True 表示按下，False 表示松开
#
# # 按下按键时的回调
# def on_press(key):
#     global key_state
#     # if hasattr(key, 'char') and key.char == 'w':
#     #     print("Pressed 'w'")
#     try:
#         if key.char:  # 普通按键
#             if not key_state and key.char == 'w' and hasattr(key, 'char'):  # 避免重复触发
#                 # print("按键被按下，开始执行任务...")
#                 posRecorder.real_right_arm.rm_set_tool_voltage(3)
#                 print(posRecorder.real_right_arm.rm_get_tool_voltage())
#                 key_state = True
#     except AttributeError:
#         # 忽略特殊按键
#         pass
# # 松开按键时的回调
# def on_release(key):
#     global key_state
#     if key_state:  # 避免重复触发
#         # print("\n\r\033[10G按键已松开，重置状态...", end="", flush=True)
#         posRecorder.real_right_arm.rm_set_tool_voltage(0)
#         print(posRecorder.real_right_arm.rm_get_tool_voltage())
#         key_state = False
#
#     # 如果按下 ESC 键，退出监听
#     if key == keyboard.Key.esc:
#         # print("退出程序...")
#         return False
#
# def wait_for_key(target_key):
#     """
#     等待用户按下指定按键。
#     :param target_key: 要监听的目标按键（字符，如 's' 或特殊键 keyboard.Key.esc）
#     """
#     print(f"等待按下 '{target_key}' 键...")
#
#     def on_press(key):
#         try:
#             if hasattr(key, 'char') and key.char == target_key:  # 字符按键
#                 print(f"检测到 '{target_key}' 键，继续程序...")
#                 return False  # 停止监听
#             elif key == target_key:  # 特殊按键
#                 print(f"检测到特殊按键 '{target_key}'，继续程序...")
#                 return False
#         except AttributeError:
#             pass  # 忽略其他按键
#
#     with keyboard.Listener(on_press=on_press) as listener:
#         listener.join()  # 阻塞程序直到监听器结束
# # 在后台启动键盘监听线程
# listener = keyboard.Listener(on_press=on_press, on_release=on_release)
# listener.start()
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
            if  len(images) == curr_device_cnt:
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


def save_hdf5(max_timesteps, joints_nums, episode_idx, data_dict, reshape_hdf5_path):
    os.makedirs(reshape_hdf5_path, exist_ok=True)
    dataset_path = os.path.join(reshape_hdf5_path, f'episode_{episode_idx}.hdf5')

    with h5py.File(dataset_path, 'w') as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        images_group = obs.create_group('images')

        # 创建每个相机的数据集并写入数据
        for cam_name in camera_names:  # 确保只处理相关键
            if f'/observations/images/{cam_name}' in data_dict:
                images_group.create_dataset(
                    cam_name.split('/')[-1],  # 使用相机名称作为数据集名称
                    data=np.array(data_dict[f'/observations/images/{cam_name}']),
                    dtype='uint8',
                    # compression='gzip',
                    # compression_opts=4
                )

        # 写入 qpos 数据
        if '/observations/qpos' in data_dict:
            if 'qpos' in obs:
                print("Dataset 'qpos' already exists. Updating it.")
                del obs['qpos']
            obs.create_dataset(
                'qpos',
                data=np.array(data_dict['/observations/qpos']),
                dtype='float32'
            )

        # 写入 action 数据
        if '/action' in data_dict:
            if 'action' in root:
                print("Dataset 'action' already exists. Updating it.")
                del root['action']
            root.create_dataset(
                'action',
                data=np.array(data_dict['/action']),
                dtype='float32'
            )

    print(f"Data saved to {dataset_path}")

def on_new_frame_callback(frames: FrameSet, index: int):
    global frames_queue
    global MAX_QUEUE_SIZE
    assert index < MAX_DEVICES
    with frames_queue_lock:
        if frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            frames_queue[index].get()
        frames_queue[index].put(frames)

def rand_pos(standard_final_pos, b):
    random_number = random.uniform(-b, b)
    random_pos = standard_final_pos.copy()
    random_pos[0] += random_number
    random_pos[2] += random_number
    # print(random_pos)
    return random_pos
def save_images_from_dict(data_dict, output_dir):
    """
    保存字典中的图像到指定文件夹。

    参数：
        data_dict (dict): 包含图像数据的字典，键为相机路径，值为图像列表。
        output_dir (str): 保存图像的根目录。
    """
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    # 遍历图像键
    for key in ['/observations/images/top', '/observations/images/right_wrist']:
        if key not in data_dict:
            print(f"Key {key} not found in data_dict.")
            continue

        cam_name = key.split('/')[-1]  # 提取相机名称
        cam_dir = os.path.join(output_dir, cam_name)  # 创建相机子目录
        os.makedirs(cam_dir, exist_ok=True)

        for i, image in enumerate(data_dict[key]):
            filename = os.path.join(cam_dir, f"color_image_{cam_name}_{i}.png")
            resized_image = cv2.resize(image, (640, 480))  # 调整到 640x480
            cv2.imwrite(filename, resized_image)
            print(f"Saved: {filename}")
def button():
    import tkinter as tk
    QposRecorder0 = QposRecorder()

    # 假设电磁铁控制程序
    # real_right_arm = Arm(RM65, "192.168.1.18")

    # 全局变量，用于记录按钮状态
    # button_state = {"power_on": False, "power_off": False}

    def power_on_magnet():
        """
        按下“上电”按钮，设置状态为 True。
        """
        QposRecorder0.real_right_arm.rm_set_tool_voltage(3)
        print("电磁铁已上电")

    def power_off_magnet():
        """
        按下“下电”按钮，设置状态为 True。
        """
        QposRecorder0.real_right_arm.rm_set_tool_voltage(0)  # 下电
        print("电磁铁已下电")

    root = tk.Tk()
    root.title("电磁铁控制")
    root.geometry("300x200")  # 设置窗口大小

    # 上电按钮
    button_on = tk.Button(root, text="上电", command=power_on_magnet, width=15, height=2)
    button_on.pack(pady=20)

    # 下电按钮
    button_off = tk.Button(root, text="下电", command=power_off_magnet, width=15, height=2)
    button_off.pack(pady=20)

    # 主循环
    root.mainloop()
def move_j(posRecorder,rand_pos):
    posRecorder.real_right_arm.rm_set_tool_voltage(3)

    queue_action=[original_pos, final_pos, rand_pos]
    temp_ = queue_action[2].copy()
    temp_[1] = temp_[1] - 0.1
    posRecorder.real_right_arm.rm_movej_p(temp_, 14, 0, 1, 0)
    posRecorder.real_right_arm.rm_movej_p(queue_action[2], 14, 0, 1, 0)
    temp_ = queue_action[2].copy()
    temp_[1] = temp_[1]-0.1
    # print(temp_)
    posRecorder.real_right_arm.rm_movel(temp_, 14, 0, 1, 0)
    temp_ = queue_action[1].copy()
    temp_[1] = temp_[1]-0.05
    posRecorder.real_right_arm.rm_movel(temp_, 14, 0, 1, 0)
    posRecorder.real_right_arm.rm_movel(queue_action[1], 14, 0, 1, 0)
    temp_ = queue_action[1].copy()
    temp_[1] = temp_[1]-0.1
    # print(temp_)
    posRecorder.real_right_arm.rm_movej_p(temp_, 14, 0, 1, 0)
    posRecorder.real_right_arm.rm_movej_p(queue_action[0],14, 0, 0, 0)
def move_back(posRecorder,rand_):
    print("move_back_to_origin_position")
    rand_position = rand_pos(rand_,0.05)
    # print(final_pos)
    posRecorder.real_right_arm.rm_set_tool_voltage(3)
    v=20
    temp_ = final_pos.copy()
    temp_[1] = temp_[1] - 0.1
    posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 0, 1)
    posRecorder.real_right_arm.rm_movej_p(final_pos, v, 0, 0, 1)

    posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 0, 1)
    posRecorder.real_right_arm.rm_movej_p(original_pos, v, 0, 0, 1)
    while True:
        temp_ = rand_position.copy()
        temp_[1] = temp_[1] - 0.1
        posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 0, 1)
        rand_pos_back = posRecorder.real_right_arm.rm_movej_p(rand_position, v, 0, 0, 1)
        posRecorder.real_right_arm.rm_set_tool_voltage(0)
        posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 0, 1)
        if rand_pos_back==1:
            print(f"error pos:{rand_position} rand again")
        elif rand_pos_back==0:
            break
    # posRecorder.real_right_arm.rm_set_tool_voltage(0)
    posRecorder.real_right_arm.rm_movej_p(original_pos, v, 0, 0, 1)
    return rand_position

def frame_to_bgr_image(frame):

    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    else:
        print("Unsupported color format: {}".format(color_format))
        return None

    return image
class CamDisplay:
    def __init__(self):
        self._image_queue = Queue()
        self._display_width = 1860
        self._display_height = 480

        print("Cam display init success!")

    def enqueue_image(self, img):
        img_list = []
        for key, value in img.items():
            img_list.append(value)
        img_queue_item = np.hstack(img_list)
        self._image_queue.put(img_queue_item)

    def display_all(self):
        while True:
            if not self._image_queue.empty():
                hstack_img = self._image_queue.get()
                resized_img = cv2.resize(hstack_img, (self._display_width, self._display_height))
                cv2.imshow("cam_view", resized_img)
                cv2.waitKey(1)

    def start_all_display(self):
        self._cam_view_thread = threading.Thread(target=self.display_all)
        self._cam_view_thread.start()

    def stop_all_display(self):
        self._cam_view_thread.join()

def main(rand_pos, indx, pipelines,posRecorder, cam_display):
    qpos_list = []
    get_image_number = 0
    images_dict = {cam_name: [] for cam_name in camera_names}  # 用于存储每个相机的图片
    max_len = 0
    global stop_processing,serial_number_list
    try:
        now = time.time()
        for i in range(30):
            image = process_frames(pipelines)
        print(f"预热结束")
        drop_time  = time.time()
        j=0
        first_trajectory = False
        for i in tqdm(range(max_timesteps)):

            if i==10:
                print("开始运动")
                move_j(posRecorder,rand_pos)
            now = time.time()
            if posRecorder.real_right_arm.rm_get_arm_current_trajectory()['trajectory_type'] == 2 and j < 1 :
                posRecorder.real_right_arm.rm_set_tool_voltage(3)
                j += 1
                first_trajectory = True
                # print("in first traj ")
            elif posRecorder.real_right_arm.rm_get_arm_current_trajectory()['trajectory_type'] == 1 and first_trajectory:
                posRecorder.real_right_arm.rm_set_tool_voltage(0)
            elif posRecorder.real_right_arm.rm_get_arm_current_trajectory()['trajectory_type'] == 0 and first_trajectory:
                traj_time = time.time()
                print(f"traj_time - drop_time{traj_time - drop_time}")
                max_len=i-1
                break
            image = process_frames(pipelines)
            cam_display.enqueue_image(image)

            angle_qpos = posRecorder.get_state()
            radius_qpos = [math.radians(j) for j in angle_qpos]
            radius_qpos[6] = posRecorder.real_right_arm.rm_get_tool_voltage()[1]
            # print(radius_qpos[6])
            qpos_list.append(radius_qpos)
            if curr_device_cnt==1 :
                images_dict['top'].append(cv2.resize(image[0], (640, 480)))
            elif curr_device_cnt == 2 and 'CP1L44P0004Y' in serial_number_list and 'CP1E54200056' in serial_number_list:
                images_dict['top'].append(cv2.resize(image[0], (640, 480)))
                images_dict['right_wrist'].append(cv2.resize(image[1], (640, 480)))
            elif curr_device_cnt ==3 and 'CP1L44P0006E' in serial_number_list and 'CP1E54200056' in serial_number_list and 'CP1L44P0004Y' in serial_number_list:
                images_dict['top'].append(cv2.resize(image[2], (640, 480)))
                images_dict['right_wrist'].append(cv2.resize(image[0], (640, 480)))
                images_dict['left_wrist'].append(cv2.resize(image[1], (640, 480)))

            else:
                raise "device error"
            for name in camera_names:
                filename_ = os.path.join(os.getcwd(), "color_images", f"color_image_{name}_{i}.png")
                if name =='top' and get_image_number <=0 and 'CP1L44P0006E' in serial_number_list:
                    # print(get_image_number)
                    cv2.imwrite(filename_,cv2.resize(image[1], (640, 480)))
                if name == 'right_wrist' and 'CP1E54200056' in serial_number_list and get_image_number <=0:
                    cv2.imwrite(filename_,cv2.resize(image[0], (640, 480)))
                    get_image_number += 1

            # print(f"7 joint :{posRecorder.get_state()[-1]}")
            # print(angle_qpos[1])
            # if angle_qpos[1]>60:
            #     # print(qpos)
            #     posRecorder.real_right_arm.rm_set_tool_voltage(0)
            # else:
            #     posRecorder.real_right_arm.rm_set_tool_voltage(3)
            next_episode = time.time()
            # print(f"1 episode time : {next_episode - now}")

    except KeyboardInterrupt:
        print("Interrupted by user")
        stop_processing = True
    finally:
        # 将图像列表转换为 NumPy 数组
        for cam_name in camera_names:
            images_dict[cam_name] = np.array(images_dict[cam_name])

        # 创建动作列表
        action_list = qpos_list
        qpos_list = np.vstack([qpos_list[0], qpos_list])

        # 构建数据字典
        data_dict = {
            '/observations/qpos': qpos_list[:max_len],
            '/action': action_list[:max_len],
        }
        # print(data_dict)
        # 添加图像数据到字典
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = images_dict[cam_name][:max_len]
        # for i in len(data_dict['/observations/images/top']):
        #     filename_ = os.path.join(os.getcwd(), "color_images_", f"color_image_top_{i}.png")
        #     cv2.imwrite(filename_, cv2.resize(image[0], (640, 480)))
        # save_images_from_dict(data_dict, 'color_image_')
        # 保存到 HDF5 文件
        # print(data_dict['/observations/qpos'])
        save_hdf5(
            max_timesteps=max_timesteps,
            joints_nums=7,
            episode_idx=indx,
            data_dict=data_dict,
            reshape_hdf5_path='../3_cam_1.2'
        )
        # 确保监听器线程被正确关闭
        # listener_thread.join()
        print("===============Stopping pipelines====")
if __name__ == "__main__":
    start = time.time()
    max_timesteps=5000
    read_config(config_file_path)
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("No device connected")

    pipelines = []
    configs = []
    serial_number_list = []
    curr_device_cnt = device_list.get_count()
    for i in range(min(device_list.get_count(), MAX_DEVICES)):
        device = device_list.get_device_by_index(i)
        pipeline = Pipeline(device)
        config = Config()
        serial_number = device.get_device_info().get_serial_number()
        serial_number_list.append(serial_number)
        # sync_config_json = multi_device_sync_config[serial_number]
        # sync_config = device.get_multi_device_sync_config()
        # sync_config.mode = sync_mode_from_str(sync_config_json["config"]["mode"])
        # sync_config.color_delay_us = sync_config_json["config"]["color_delay_us"]
        # sync_config.depth_delay_us = sync_config_json["config"]["depth_delay_us"]
        # sync_config.trigger_out_enable = sync_config_json["config"]["trigger_out_enable"]
        # sync_config.trigger_out_delay_us = sync_config_json["config"]["trigger_out_delay_us"]
        # sync_config.frames_per_trigger = sync_config_json["config"]["frames_per_trigger"]
        # device.set_multi_device_sync_config(sync_config)
        # print(f"Device {serial_number} sync config: {sync_config}")

        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        # print(f"Device {serial_number} color profile: {color_profile}")
        #
        # profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        # depth_profile = profile_list.get_default_video_stream_profile()
        # print(f"Device {serial_number} depth profile: {depth_profile}")
        # config.enable_stream(depth_profile)
        pipelines.append(pipeline)
        configs.append(config)
    # max_timesteps = 400
    save_image_dir = os.path.join(os.getcwd(), "color_images")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    # save_image_dir = os.path.join(os.getcwd(), "color_images_")
    # if not os.path.exists(save_image_dir):
    #     os.mkdir(save_image_dir)
    start_streams(pipelines, configs)
    # threading.Event().wait(0.1)
    pre_time = time.time()
    # button()
    posRecorder = QposRecorder()
    posRecorder.real_right_arm.rm_set_arm_delete_trajectory()
    posRecorder.real_right_arm.rm_movej_p(original_pos, 14, 0, 0, 1)
    # time.sleep(1111)
    # s= time.time()

    cam_display = CamDisplay()
    cam_display.start_all_display()
    # main(standard_final_pos,0, pipelines)
    # move_back(standard_final_pos)
    # time.sleep(1111)

    for i in range(20):
        print(f"i:{i}")
        # print(posRecorder.real_right_arm.rm_get_tool_voltage())
        final = move_back(posRecorder,standard_final_pos)
        main(final,i,pipelines,posRecorder,cam_display)
    stop_streams(pipelines)
    cam_display.stop_all_display()

    end = time.time()
    print(f"total time in 1 roll:{end-start}")


