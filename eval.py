# from datetime import datetime
import os, datetime, sys
import threading
import time
import json

import numpy as np
import cv2
from queue import Queue
from pyorbbecsdk import *
from utils import frame_to_bgr_image

MAX_DEVICES = 5  # 假设最多支持 5 台设备
MAX_QUEUE_SIZE = 10  # 最大帧队列长度
multi_device_sync_config = {}
config_file_path = os.path.join(os.path.dirname(__file__), "./config/multi_device_sync_config.json")
from networkx.readwrite.json_graph.tree import tree_data
from triton.language.semantic import store
from deploy.hdf5_edit_utils import Modify_hdf5
from deploy.policy_action_generation import ActionGenerator
import numpy as np
import math
from tqdm import tqdm
from deploy.visualize_episodes import visualize_joints
from tcp_tx import PersistentClient
import pytz
tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(tz)
# datetime.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
STATE_NAMES = JOINT_NAMES +["gripper_pos"]+ ["gripper_force"]
camera_names = ['top', 'right_wrist','left_wrist']

class CAMERA_HOT_PLUG:
    def __init__(self):
        self.mutex = threading.Lock()
        self.ctx = Context()
        self.device_list = self.ctx.query_devices()
        self.curr_device_cnt = self.device_list.get_count()
        self.pipelines: list[Pipeline] = []
        self.configs: list[Config] = []
        self.serial_number_list: list[str] = ["" for _ in range(self.curr_device_cnt)]
        self.color_frames_queue: dict[str, Queue] = {}
        self.depth_frames_queue: dict[str, Queue] = {}

        self.setup_cameras()
        self.start_streams()
        print("相机初始化完成")

        # 监控设备线程
        self.monitor_thread = threading.Thread(target=self.monitor_devices, daemon=True)
        self.monitor_thread.start()

    def monitor_devices(self):
        """定期检查相机连接状态"""
        while True:
            time.sleep(2)
            new_device_list = self.ctx.query_devices()
            new_device_cnt = new_device_list.get_count()

            if new_device_cnt != self.curr_device_cnt:
                print("设备变化检测到，重新初始化相机...")
                self.stop_streams()
                self.device_list = new_device_list
                self.curr_device_cnt = new_device_cnt
                self.setup_cameras()
                self.start_streams()

    def setup_cameras(self):
        """初始化相机设备"""
        self.read_config(config_file_path)

        if self.curr_device_cnt == 0:
            print("⚠️ No device connected")
            return
        if self.curr_device_cnt > MAX_DEVICES:
            print("⚠️ Too many devices connected")
            return

        for i in range(self.curr_device_cnt):
            device = self.device_list.get_device_by_index(i)
            serial_number = device.get_device_info().get_serial_number()

            self.color_frames_queue[serial_number] = Queue()
            self.depth_frames_queue[serial_number] = Queue()
            pipeline = Pipeline(device)
            config = Config()

            # 读取同步配置
            sync_config_json = multi_device_sync_config.get(serial_number, {})
            sync_config = device.get_multi_device_sync_config()
            sync_config.mode = self.sync_mode_from_str(sync_config_json["config"]["mode"])
            sync_config.color_delay_us = sync_config_json["config"]["color_delay_us"]
            sync_config.depth_delay_us = sync_config_json["config"]["depth_delay_us"]
            sync_config.trigger_out_enable = sync_config_json["config"]["trigger_out_enable"]
            sync_config.trigger_out_delay_us = sync_config_json["config"]["trigger_out_delay_us"]
            sync_config.frames_per_trigger = sync_config_json["config"]["frames_per_trigger"]
            device.set_multi_device_sync_config(sync_config)

            try:
                profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                color_profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)

                profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = profile_list.get_default_video_stream_profile()
                config.enable_stream(depth_profile)

                self.pipelines.append(pipeline)
                self.configs.append(config)
                self.serial_number_list[i] = serial_number
            except OBError as e:
                print(f"setup_cameras error: {e}")

    def start_streams(self):
        """启动相机流"""
        print(self.serial_number_list)
        for index, (pipeline, config, serial) in enumerate(zip(self.pipelines, self.configs, self.serial_number_list)):
            pipeline.start(
                config,
                lambda frame_set, curr_serial=serial: self.on_new_frame_callback(frame_set, curr_serial),
            )

    def stop_streams(self):
        """停止相机流"""
        with self.mutex:
            try:
                for pipeline in self.pipelines:
                    pipeline.stop()
                self.pipelines = []
                self.configs = []
                print("📷 Devices stopped")
            except Exception as e:
                print(f"⚠️ Error stopping streams: {e}")

    def on_new_frame_callback(self, frames: FrameSet, serial_number: str):
        """接收新帧并存入队列"""
        with self.mutex:
            if serial_number not in self.color_frames_queue:
                print(f"⚠️ WARN: 未识别的相机序列号 {serial_number}，跳过帧处理")
                return

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame:
                if self.color_frames_queue[serial_number].qsize() >= MAX_QUEUE_SIZE:
                    self.color_frames_queue[serial_number].get()
                self.color_frames_queue[serial_number].put(color_frame)

            if depth_frame:
                if self.depth_frames_queue[serial_number].qsize() >= MAX_QUEUE_SIZE:
                    self.depth_frames_queue[serial_number].get()
                self.depth_frames_queue[serial_number].put(depth_frame)

    def rendering_frame(self, max_wait=5):
        image_dict: dict[str, np.ndarray] = {}
        start_time = time.time()
        color_width, color_height = None, None
        while len(image_dict) != self.curr_device_cnt:
            if time.time() - start_time > max_wait:
                print("⚠️ WARN: 渲染超时，部分相机未收到帧数据")
                break
            for serial_number in self.color_frames_queue.keys():
                color_frame = None
                if not self.color_frames_queue[serial_number].empty():
                    color_frame = self.color_frames_queue[serial_number].get()
                if color_frame is None:
                    continue
                color_width, color_height = color_frame.get_width(), color_frame.get_height()
                color_image = frame_to_bgr_image(color_frame)
                image_dict[serial_number] = color_image

        return image_dict,color_width, color_height

    def sync_mode_from_str(self, sync_mode_str: str) -> OBMultiDeviceSyncMode:
        """将字符串转换为同步模式"""
        sync_mode_str = sync_mode_str.upper()
        sync_modes = {
            "FREE_RUN": OBMultiDeviceSyncMode.FREE_RUN,
            "STANDALONE": OBMultiDeviceSyncMode.STANDALONE,
            "PRIMARY": OBMultiDeviceSyncMode.PRIMARY,
            "SECONDARY": OBMultiDeviceSyncMode.SECONDARY,
            "SECONDARY_SYNCED": OBMultiDeviceSyncMode.SECONDARY_SYNCED,
            "SOFTWARE_TRIGGERING": OBMultiDeviceSyncMode.SOFTWARE_TRIGGERING,
            "HARDWARE_TRIGGERING": OBMultiDeviceSyncMode.HARDWARE_TRIGGERING,
        }
        return sync_modes.get(sync_mode_str, OBMultiDeviceSyncMode.FREE_RUN)

    def read_config(self, config_file: str):
        """读取配置文件"""
        global multi_device_sync_config
        with open(config_file, "r") as f:
            config = json.load(f)
        for device in config["devices"]:
            multi_device_sync_config[device["serial_number"]] = device
            print(f"📷 Device {device['serial_number']}: {device['config']['mode']}")

class eval:
    def __init__(self,real_robot=False,data_true=False):
        self.real_robot = real_robot
        self.data_true = data_true
        if self.real_robot:
            self.camera = CAMERA_HOT_PLUG()
            self.persistentClient = PersistentClient('192.168.2.14', 8001)

            self.image = {'top': [], 'right_wrist': [], 'left_wrist':[]}
            self.main()
        else:
            self.main()
    def updata_frame(self):
        """更新摄像头图像"""
        global multi_device_sync_config
        frame_data, color_width, color_height = self.camera.rendering_frame()
        serial_number_list = self.camera.serial_number_list
        camera_index_map = {device['config']['camera_name']: serial_number_list.index(device["serial_number"]) for device in multi_device_sync_config.values() if device["serial_number"] in serial_number_list}

        # print(f"frame_data: {type(frame_data)}")
        # print(frame_data[serial_number_list[0]].shape)
        # 判断 frame_data 的类型
        if isinstance(frame_data, dict):  # 多台摄像头返回字典 {str: np.ndarray}
            if not frame_data:  # 字典为空
                print("⚠️ WARN: 没有接收到任何摄像头图像")
                return
            if all(img.size == 0 for img in frame_data.values()):  # 所有相机的图像都是空的
                print("⚠️ WARN: 所有摄像头的图像数据为空")
                return
            # print(f"⚠️ WARN: 多台摄像头，序列号列表: {serial_number_list}")
        elif isinstance(frame_data, np.ndarray):  # 只有一台相机
            if frame_data.size == 0:
                print("⚠️ WARN: 没有接收到任何摄像头图像")
                return
            # 只有一个摄像头时，将其存入字典，模拟多摄像头格式
            frame_data = {"0": frame_data}  
            serial_number_list = ["0"]
            print(f"⚠️ WARN: 只有一台摄像头，序列号为 {serial_number_list[0]}")
        else:
            print(f"⚠️ ERROR: 无效的 frame_data 类型: {type(frame_data)}")
            return
        # 初始化结果图像
        num_images = len(frame_data)
        result_image = None
        for device in multi_device_sync_config.values():
            cam_name, sn = device['config']['camera_name'], device["serial_number"]
            if sn in frame_data:
                img = frame_data[sn]
                if result_image is None:
                    result_image = img  # 第一个摄像头的图像
                else:
                    result_image = np.hstack((result_image, img))  # 按水平方向拼接图像
            else:
                print(f"⚠️ WARN: 摄像头 {cam_name}（{sn}）的图像数据缺失")

        if result_image is not None:
            # 调整大小并显示图像
            result_image = cv2.resize(result_image, (color_width, color_height))
            # self.display_image(result_image)
            for camera_name in camera_names:
                self.image[camera_name] = frame_data.get(str(serial_number_list[camera_index_map[camera_name]]))
            # self.image['top'] = frame_data.get(str(serial_number_list[camera_index_map['top']]), None)
            # self.image['right_wrist'] = frame_data.get(str(serial_number_list[camera_index_map['right_wrist']]), None) if num_images > 1 else None
    def is_close(self, actual, target, tolerance=0.1):
        """
        判断两个列表的每个元素是否在允许误差范围内
        :param actual: 实际值列表（如当前机械臂状态）
        :param target: 目标值列表
        :param tolerance: 允许的最大误差（绝对值）
        :return: 所有元素均满足误差要求返回True，否则False
        """
        # 处理None和长度检查
        if actual is None or target is None:
            return False
        if len(actual) != len(target):
            return False
        
        # 逐个元素比较误差
        for a, t in zip(actual, target):
            if abs(a - t) > tolerance:
                return False
        return True
    def main(self):
        actions_list = []
        qpos_list_ = []
        loss = []
        self.radius_qpos_list = []

        if self.real_robot:
            loop_len = 200
            task_complete_step = None
            square_size = 100
        else:
            data_dict = Modify_hdf5()
            dict_ = data_dict.check_hdf5(r'/workspace/exchange/4-24/hdf5_file_duikong/episode_16.hdf5')
            # print(dict_["action"].shape)
            loop_len = len(dict_['top'])
        config = {
            'ckpt_dir': r'/workspace/exchange/4-24/hdf5_file_duikong/hit',
            'max_timesteps': loop_len,
            'ckpt_name': "policy_step_1000_seed_0.ckpt",
            'backbone': 'resnet18'
        }
        image_dict = {i:[] for i in camera_names}
        # print(image_dict)
        ActionGeneration = ActionGenerator(config)
        for i in tqdm(range(loop_len)):
            # print(f"roll:{i}")
            ActionGeneration.t = i
            if self.real_robot:
                if self.data_true:
                    self.updata_frame()
                    if self.is_close(self.persistentClient.get_arm_position_joint(1)[:3],[-121.42, -599.741, -209.687]) or i >30:
                        task_complete_step = 1
                    for camera_name in camera_names:
                        image_dict[camera_name] = self.image[camera_name]
                        qpos = self.persistentClient.get_arm_position_joint(1)
                        radius_qpos = [math.radians(j) for j in qpos]
                        img_copy = [row[:] for row in image_dict[camera_name]]  # 深拷贝，防止改到原图
                        height = len(img_copy)
                        width = len(img_copy[0])
                        print(height,width)
                        square_color = [0, 0, 255] if task_complete_step is  None else [0, 255, 0]  
                        if square_color == [0,0,255]:
                            print("红色")
                        elif  square_color == [0, 255, 0]:
                            print("绿色")
                        # 左下角：行范围 [height - square_size, height)
                        for row in range(height - square_size, height):
                            for col in range(square_size):
                                if 0 <= row < height and 0 <= col < width:
                                    img_copy[row][col] = square_color
                        image_dict[camera_name] = np.array(img_copy)
                else:
                    for camera_name in camera_names:
                        image_dict[camera_name]=np.array(dict_[camera_name][i])
                        qpos = self.persistentClient.get_arm_position_joint(1)
                        radius_qpos = [math.radians(j) for j in qpos]
            else:
                
                for camera_name in camera_names:
                    image_dict[camera_name] = np.array(dict_[camera_name][i])
                radius_qpos = dict_['qpos'][i]
           
            self.radius_qpos_list.append(radius_qpos)
            # print(image_dict)
            ActionGeneration.image_dict = image_dict
            ActionGeneration.qpos_list = radius_qpos
            actions = ActionGeneration.get_action()
            # print(qpos)
            # print(list(np.degrees(actions)))
            left_arm_action = np.rad2deg(actions[:6])
            right_arm_action= np.rad2deg(actions[8:14])
            left_gp = actions[6:8]
            right_gp = actions[14:16]
            # print(actions[6:8],actions[14:16])
            # print(left_arm_action,right_arm_action)
            if self.real_robot:

                self.persistentClient.set_arm_position(list(left_arm_action), "joint", 1)
                self.persistentClient.set_arm_position(list(right_arm_action), "joint", 2)
            actions_list.append(actions)
            # loss.append((actions - dict_['action'][i]) ** 2)
        today = current_time.strftime("%m-%d-%H-%M")
        path_save_image = os.path.join(os.getcwd(), "deploy_image", f"{today}")
        os.makedirs(path_save_image, exist_ok=True)
        image_path = os.path.join(path_save_image, config['backbone']+"_"+ os.path.splitext(config['ckpt_name'])[0]+ ".png")
        loss_apth = os.path.join(path_save_image, 'loss' + current_time.strftime("%m-%d+8-%H-%M") + ".png")
       
        visualize_joints(self.radius_qpos_list, actions_list, image_path, STATE_NAMES=STATE_NAMES)

if __name__ == '__main__':
    eval(real_robot=False,data_true=False)