import json
import os
from pdb import run
import threading
import time
from queue import Queue
from threading import Lock
from tkinter import NO
from traceback import print_tb
from typing import List
from Robotic_Arm.rm_robot_interface import *
import math, h5py
import cv2
import numpy as np
from tqdm import tqdm
from pyorbbecsdk import *
# from deploy.remote_control import posRecorder
# from utils import frame_to_bgr_image
# from pynput.keyboard import Listener, Key
from pynput import keyboard
import random
import serial
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
config_file_path = os.path.join(os.path.dirname(__file__), "../pyorbbecsdk/config/multi_device_sync_config.json")
multi_device_sync_config = {}
camera_names = ['top', 'right_wrist']
#是位置姿态不是关节值
zero_pos = [0, 0, 0, 0, 0, 0]
original_pos = [-0.242293, 0.055747, 0.692225, -1.569, -0.597, 1.472]#空中点位
final_pos =  [-0.300054, 0.215523, 0.493377, -2.749, -0.706, 1.804]#桌面点位
standard_start_pos = [-0.19953, 0.169551, 0.685523, -1.605, -0.672, 0.916]#桌面点位
test_pos =[-0.106262, 0.165886, 0.68286, 1.667, 0.764, -2.07]
test_pos_2 = [-0.136529, 0.041631, 0.681072, 1.808, 0.971, -1.668]
# original_pos=test_pos_2
# final_pos=test_pos_2

class gpcontrol():
    def __init__(self):
        self.DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
        self.BAUD_RATE = 50000
        self.min_data = b'\x00\x00\xFF\xFF\xFF\xFF\x00\x00'
        self.max_data = b'\x00\xFF\xFF\xFF\xFF\xFF\x00\x00'
        self.ser = self.open_serial()
        self.is_sending = False
        self.is_configured = False  # 配置标志位
        set_can1 = b'\x49\x3B\x42\x57\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E'
        start_can1 = b'\x49\x3B\x44\x57\x01\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E'
        self.send_data(set_can1)  # 发送配置指令
        self.read_data()
        self.send_data(start_can1)  # 启动 CAN 通道
        self.read_data()    
    def open_serial(self):
        """打开串口"""
        port = self.DEFAULT_SERIAL_PORT
        baudrate = self.BAUD_RATE
        try:
            ser = serial.Serial(port, baudrate, timeout=1)
            print(f"串口 {port} 已打开，波特率 {baudrate}")
            return ser
        except Exception as e:
            print(f"无法打开串口 {port}: {e}")
            return None
    
    def send_data(self, data):
        """发送数据到串口"""
        ser=self.ser
        if ser and ser.is_open:
            ser.write(data)
            print(f"发送数据: {data.hex()}")
        else:
            print("串口未打开，无法发送数据")


    def filter_can_data(self, data):
        """根据头（0x5A）和尾（0xA5）过滤数据"""
        valid_frames = []

        # 查找所有以 0x5A 开头并以 0xA5 结尾的数据帧
        start_idx = 0
        while start_idx < len(data):
            # 查找下一个0x5A
            start_idx = data.find(b'\x5A', start_idx)
            if start_idx == -1:  # 如果找不到0x5A，退出循环
                break

            # 查找下一个0xA5
            end_idx = data.find(b'\xA5', start_idx)
            if end_idx == -1:  # 如果找不到0xA5，退出循环
                break

            # 提取有效数据帧（包括0x5A和0xA5）
            frame = data[start_idx:end_idx + 1]

            # 确保数据帧长度合理（至少 8 字节）
            if len(frame) >= 8:
                valid_frames.append(frame)

            # 设置起始索引，继续查找下一个帧
            start_idx = end_idx + 1

        return valid_frames


    def read_data(self):
        """读取串口返回数据并过滤符合头尾要求的数据"""
        ser = self.ser
        if ser and ser.is_open:
            data = ser.read(32)  # 读取最大 64 字节
            if data:
                valid_frames = self.filter_can_data(data)
                if valid_frames:
                    back_data=0
                    for frame in valid_frames:
                        if frame[:2].hex()=='5aff':
                            print("-----------")
                            continue
                        else:
                            print(f"接收符合条件的CAN数据: {frame.hex()}")
                            back_data=frame.hex()
                    
                    return valid_frames, back_data
                else:
                    print("未收到符合条件的数据帧")
            else:
                print("未收到数据")
        else:
            print("串口未打开，无法读取数据")
        return None


    def send_can_data(self, can_id, data, channel):
        """
        发送 CAN 数据帧
        :param ser: 串口对象
        :param can_id: 4字节 CAN ID
        :param data: 发送数据，最大 64 字节
        """
        can_id_bytes = can_id  # CAN ID 转换成 4字节

        data_length = len(data)
        if data_length > 64:
            data = data[:64]  # 限制数据长度为 64 字节

        frame_header = b'\x5A'  # 帧头
        frame_info_1 = (data_length | channel << 7).to_bytes(1, 'big')  # CAN通道0, DLC数据长度
        frame_info_2 = b'\x00'  # 发送类型: 正常发送, 标准帧, 数据帧, 不加速
        frame_data = data.ljust(64, b'\x00')  # 数据填充到 64 字节
        frame_end = b'\xA5'  # 帧尾

        send_frame = frame_header + frame_info_1 + frame_info_2 + can_id_bytes + frame_data[:data_length] + frame_end
        # print("发送 CAN 帧:", send_frame.hex())
        self.send_data(send_frame)
        # _,data = self.read_data()
        # return data
    def open_half_gp(self):
        half_open_gp = b'\x00\x7f\xFF\xFF\xFF\xFF\x00\x00'
        while 1:
            self.send_can_data(b'\x00\x00\x00\x01', half_open_gp, 0x01)
            data = self.read_data() 
            if data is not None:
                _, gpdata = data
                while gpdata == 0:
                    self.send_can_data(b'\x00\x00\x00\x01', half_open_gp, 0x01)
                    data = self.read_data()
                    if data is not None:
                        _, gpdata = data
                gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
                return [gpstate,gppos,gpforce]
        
    def open_all_gp(self):
        open_gp = b'\x00\xff\xFF\xFF\xFF\xFF\x00\x00'
        while 1:
            self.send_can_data(b'\x00\x00\x00\x01', open_gp, 0x01)
            data = self.read_data() 
            if data is not None:
                _, gpdata = data
                while gpdata == 0:
                    self.send_can_data(b'\x00\x00\x00\x01', open_gp, 0x01)
                    data = self.read_data()
                    if data is not None:
                        _, gpdata = data
                gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
                return [gpstate,gppos,gpforce]
        
    def close_gp(self):
        close_gp = b'\x00\x00\xFF\xFF\xFF\xFF\x00\x00'
        while 1:
            self.send_can_data(b'\x00\x00\x00\x01', close_gp, 0x01)
            data = self.read_data() 
            if data is not None:
                _, gpdata = data
                while gpdata == 0:
                    self.send_can_data(b'\x00\x00\x00\x01', close_gp, 0x01)
                    data = self.read_data()
                    if data is not None:
                        _, gpdata = data
                gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
                return [gpstate,gppos,gpforce]
        
    
    def close(self):
        if self.ser:
            self.ser.close()

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
                # print(f"frames is none")
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

    try:
        print(f"Saving data to: {dataset_path}")  # 输出保存路径
        with h5py.File(dataset_path, 'w') as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            images_group = obs.create_group('images')
            gp = root.create_group('gp')

            # 创建每个相机的数据集并写入数据
            for cam_name in camera_names:
                if f'/observations/images/{cam_name}' in data_dict:
                    try:
                        cam_data = np.array(data_dict[f'/observations/images/{cam_name}'])
                        print(f"Saving image for {cam_name}, shape: {cam_data.shape}")  # 打印图片数据的尺寸
                        images_group.create_dataset(
                            cam_name.split('/')[-1],
                            data=cam_data,
                            dtype='uint8',
                        )
                    except Exception as e:
                        print(f"Error saving image data for camera {cam_name}: {e}")

            # 写入 qpos 数据
            if '/observations/qpos' in data_dict:
                if 'qpos' in obs:
                    print("Dataset 'qpos' already exists. Updating it.")
                    del obs['qpos']
                qpos_data = np.array(data_dict['/observations/qpos'])
                print(f"Saving qpos, shape: {qpos_data.shape}")
                obs.create_dataset(
                    'qpos',
                    data=qpos_data,
                    dtype='float32'
                )

            # 写入 action 数据
            if '/action' in data_dict:
                if 'action' in root:
                    print("Dataset 'action' already exists. Updating it.")
                    del root['action']
                action_data = np.array(data_dict['/action'])
                print(f"Saving action, shape: {action_data.shape}")
                root.create_dataset(
                    'action',
                    data=action_data,
                    dtype='float32'
                )

            # 保存 gpstate, gppos, gpforce 数据
            if '/gp/gppos' in data_dict:
                if 'gppos' in gp:
                    print("Dataset 'gppos' already exists. Updating it.")
                    del gp['gppos']
                try:
                    gppos_data = np.array([int(x, 16) for x in data_dict['/gp/gppos']], dtype='int32')
                    print(f"Saving gppos, length: {len(gppos_data)}")
                    gp.create_dataset(
                        'gppos',
                        data=gppos_data
                    )
                except Exception as e:
                    print(f"Error saving gppos data: {e}")
                    return

            if '/gp/gpstate' in data_dict:
                if 'gpstate' in gp:
                    print("Dataset 'gpstate' already exists. Updating it.")
                    del gp['gpstate']
                try:
                    gpstate_data = np.array([int(x, 16) for x in data_dict['/gp/gpstate']], dtype='int32')
                    print(f"Saving gpstate, length: {len(gpstate_data)}")
                    gp.create_dataset(
                        'gpstate',
                        data=gpstate_data
                    )
                except Exception as e:
                    print(f"Error saving gpstate data: {e}")
                    return

            if '/gp/gpforce' in data_dict:
                if 'gpforce' in gp:
                    print("Dataset 'gpforce' already exists. Updating it.")
                    del gp['gpforce']
                try:
                    gpforce_data = np.array([int(x, 16) for x in data_dict['/gp/gpforce']], dtype='int32')
                    print(f"Saving gpforce, length: {len(gpforce_data)}")
                    gp.create_dataset(
                        'gpforce',
                        data=gpforce_data
                    )
                except Exception as e:
                    print(f"Error saving gpforce data: {e}")
                    return

            # 强制刷新文件，确保数据写入
            root.flush()

    except Exception as e:
        print(f"Error during saving hdf5 file: {e}")
    
    print(f"Data saved to {dataset_path}")
def on_new_frame_callback(frames: FrameSet, index: int):
    global frames_queue
    global MAX_QUEUE_SIZE
    assert index < MAX_DEVICES
    with frames_queue_lock:
        if frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            frames_queue[index].get()
        frames_queue[index].put(frames)

def rand_pos(standard_start_pos, b):
    random_number = random.uniform(-b, b)
    random_pos = standard_start_pos.copy()
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
def run_check_state(cmd):
    state = None
    while state[0]!='00':
        state = cmd
        print(f"state{state}")
    return state
def move_j(posRecorder,rand_pos):
    # posRecorder = QposRecorder()
    queue_action=[original_pos, final_pos, rand_pos]
    v=15
    low_v=3
    temp_ = queue_action[2].copy()
    temp_[1] = temp_[1] - 0.1
    posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 1, 0)
    # temp_[1] = temp_[1] + 0.09
    # posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 1, 0)
    posRecorder.real_right_arm.rm_movej_p(queue_action[2], v, 0, 1, 0)
    temp_[1] = temp_[1] + 0.09
    posRecorder.real_right_arm.rm_movel(temp_, low_v, 0, 1, 0)
    # temp_[1] = temp_[1]-0.09
    # posRecorder.real_right_arm.rm_movel(temp_, low_v, 0, 1, 0)

    temp_ = queue_action[1].copy()
    temp_[1] = temp_[1]-0.1
    posRecorder.real_right_arm.rm_movel(temp_, v, 0, 1, 0)
    temp_[1] = temp_[1]+0.09
    posRecorder.real_right_arm.rm_movel(temp_, v, 0, 1, 0)
    # posRecorder.real_right_arm.rm_movel(queue_action[1], v, 0, 1, 0)
    temp_[1] = temp_[1]
    posRecorder.real_right_arm.rm_movej_p(temp_, low_v, 0, 1, 0)
    temp_[1] = temp_[1]-0.09
    posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 1, 0)

    temp_ = queue_action[0].copy()
    temp_[1] = temp_[1]-0.1
    posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 1, 0)
    posRecorder.real_right_arm.rm_movej_p(queue_action[0],v, 0, 0, 0)

def move_back(gpcontrol1,posRecorder, rand_):
    print("move_back_to_origin_position")
    # posRecorder = QposRecorder()
    rand_position = rand_pos(rand_,0.04)
    rand_position=standard_start_pos
    # print(final_pos)
    v=20
    temp_ = final_pos.copy()
    temp_[1] = temp_[1] - 0.1
    state = gpcontrol1.open_all_gp()
    posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 0, 1)
    posRecorder.real_right_arm.rm_movej_p(final_pos, v, 0, 0, 1)
    state = gpcontrol1.close_gp()
 
    posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 0, 1)
    # posRecorder.real_right_arm.rm_movej_p(original_pos, v, 0, 0, 1)
    while True:
        temp_ = rand_position.copy()
        temp_[1] = temp_[1] - 0.1
        rand_pos_back1 = posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 0, 1)
        rand_pos_back = posRecorder.real_right_arm.rm_movej_p(rand_position, v, 0, 0, 1)
        print(rand_pos_back1,rand_pos_back)
        if rand_pos_back==1 or rand_pos_back1==1:
            print(f"error pos:{rand_position} rand again")
            posRecorder.real_right_arm.rm_set_delete_current_trajectory()
        elif rand_pos_back==0 and rand_pos_back1==0:
            state = gpcontrol1.open_half_gp()
            posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 0, 1)
            break
    temp_ = rand_position.copy()
    temp_[1] = temp_[1] - 0.1
    posRecorder.real_right_arm.rm_movej_p(temp_, v, 0, 0, 1)
    posRecorder.real_right_arm.rm_movej_p(original_pos, v, 0, 0, 1)
    print("quit back mode")
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

def main(rand_pos, indx, pipelines, posRecorder, gpcontrol1, cam_display):
    qpos_list = []
    
    gpstate_list=[]
    gppos_list=[]
    gpforce_list=[]
    # posRecorder = QposRecorder()
    get_image_number = 0
    images_dict = {cam_name: [] for cam_name in camera_names}  # 用于存储每个相机的图片
    max_len = max_timesteps
    global stop_processing,serial_number_list
    stop_sign=0
    try:
        now = time.time()
        print("开始预热")
        for i in range(30):
            
            image = process_frames(pipelines)
        print(f"预热结束")
        drop_time  = time.time()
        j=0
        first_trajectory = False
        for i in tqdm(range(max_timesteps)):
            # print(i)
            gpstate,gppos,gpforce=0,0,0
            time.sleep(0.1)
            if i==10:
                print("\n开始运动")
                # state=gpcontrol1.open_all_gp()
                # print(state)
                move_j(posRecorder,rand_pos)
            now = time.time()
            trajectory_type = posRecorder.real_right_arm.rm_get_arm_current_trajectory()['trajectory_type']
            print("trajectory_type:",trajectory_type)
            if trajectory_type == 2 :
                state=gpcontrol1.close_gp()
                if state is None:
                    raise ValueError(f"state is none {state}")
                gpstate,gppos,gpforce = state[0],state[1],state[2]
                gpstate_list.append(gpstate)
                gppos_list.append(gppos)
                gpforce_list.append(gpforce)
                # print(state)
                first_trajectory = True
                # print("in first traj ")
            elif trajectory_type == 1 :
                state=gpcontrol1.open_half_gp()
                if state is None:
                    raise ValueError(f"state is none {state}")
                gpstate,gppos,gpforce = state[0],state[1],state[2]
                gpstate_list.append(gpstate)
                gppos_list.append(gppos)
                gpforce_list.append(gpforce)
                # print(state)
            elif trajectory_type == 0 and first_trajectory :
                stop_sign+=1
                if stop_sign>5:
                    traj_time = time.time()
                    print(f"traj_time - drop_time{traj_time - drop_time}")
                    max_len=i-1
                    break
            else:
                state = gpcontrol1.open_half_gp()
                # print(state)
                if state is None:
                    raise ValueError(f"state is none {state}")
                gpstate,gppos,gpforce = state[0],state[1],state[2]
                gpstate_list.append(gpstate)
                gppos_list.append(gppos)
                gpforce_list.append(gpforce)
                
            image = process_frames(pipelines)
            if display:
                cam_display.enqueue_image(image)
            if not isinstance(gpstate, str):
                gpstate = str(gpstate)
            if not isinstance(gppos, str):
                gppos = str(gppos)
            if not isinstance(gpforce, str):
                gpforce = str(gpforce)
            angle_qpos = posRecorder.get_state()
            radius_qpos = [math.radians(j) for j in angle_qpos]
            radius_qpos.append(posRecorder.real_right_arm.rm_get_tool_voltage()[1])
            radius_qpos.append(np.array(int(gpstate, 16), dtype='int32'))
            radius_qpos.append(np.array(int(gppos, 16), dtype='int32'))
            print(gppos,radius_qpos[8])
            radius_qpos.append(np.array(int(gpforce, 16), dtype='int32'))
            qpos_list.append(radius_qpos)
            if curr_device_cnt==1 :
                images_dict['top'].append(cv2.resize(image[0], (640, 480)))
            elif curr_device_cnt == 2 and 'CP1L44P0006E' in serial_number_list and 'CP1E54200056' in serial_number_list:
                images_dict['top'].append(cv2.resize(image[1], (640, 480)))
                images_dict['right_wrist'].append(cv2.resize(image[0], (640, 480)))
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
            # can_data = b'\x00\x7f\xFF\xFF\xFF\xFF\x00\x00'
            # gpcontrol1.send_can_data(b'\x00\x00\x00\x01', can_data, 0x01)
            
            # print(gpstate,gppos,gpforce)
            next_episode = time.time()

    except KeyboardInterrupt:
        print("Interrupted by user")
        stop_processing = True
    finally:
        # 将图像列表转换为 NumPy 数组
        for cam_name in camera_names:
            images_dict[cam_name] = np.array(images_dict[cam_name])

        # 创建动作列表
        action_list = qpos_list
        if qpos_list is not None:
            qpos_list = np.vstack([qpos_list[0], qpos_list])
        else:
            raise "qpos is none"
        data_dict={}
        max_len=min(max_len,len(gpforce_list))
        # 构建数据字典
        data_dict = {
            '/observations/qpos': qpos_list[:max_len],
            '/action': action_list[:max_len],
            '/gp/gpstate': gpstate_list[:max_len],
            '/gp/gppos': gppos_list[:max_len],
            '/gp/gpforce': gpforce_list[:max_len],            


        }
        print(f"gpqpos len:{len(data_dict['/gp/gppos'])}")
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
            reshape_hdf5_path='../gp_episode_2_25_15'
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
    display=False
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
    start_streams(pipelines, configs)
    pre_time = time.time()

    print("初始化夹爪")
    gpcontrol1=gpcontrol()
    s= time.time()
    # cam_display = CamDisplay()
    if display:
        cam_display = CamDisplay()
        cam_display.start_all_display()
    for i in range(0,10):
        print(f"i:{i}")
        posRecorder = QposRecorder()
        posRecorder.real_right_arm.rm_set_arm_delete_trajectory()
        posRecorder.real_right_arm.rm_set_tool_voltage(0)
        time.sleep(1)
        posRecorder.real_right_arm.rm_set_tool_voltage(3)
        time.sleep(1)
        posRecorder.real_right_arm.rm_movej(zero_pos, 50, 0, 0, 1)
        posRecorder.real_right_arm.rm_movej_p(original_pos, 50, 0, 0, 1)
        
        final = move_back(gpcontrol1,posRecorder, standard_start_pos)
        posRecorder.real_right_arm.rm_set_arm_delete_trajectory()
        main(final,i,pipelines,posRecorder,gpcontrol1,0)
        posRecorder.real_right_arm.rm_delete_robot_arm()

    stop_streams(pipelines)
    if display:
        cam_display.stop_all_display()

    end = time.time()
    print(f"total time in 1 roll:{end-start}")
