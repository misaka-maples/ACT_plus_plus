import os, datetime, sys
import threading
import time
import json
from pyorbbecsdk import *
import numpy as np
from queue import Queue
import cv2
# 获取当前脚本的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# 添加到 sys.path
sys.path.append(parent_dir)
config_file_path = os.path.join(os.path.dirname(__file__), "../config/multi_device_sync_config.json")
from utils import frame_to_bgr_image

multi_device_sync_config = {}

MAX_DEVICES = 5  # 假设最多支持 5 台设备
MAX_QUEUE_SIZE = 10  # 最大帧队列长度

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
        # self.temporal_filter = TemporalFilter(alpha=0.5)  # Modify alpha based on desired smoothness

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

        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

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
            if not frames:
                return None
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                return None
            frames = self.align_filter.process(frames)
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
        color_image_dict: dict[str, np.ndarray] = {}
        depth_image_dict: dict[str, np.ndarray] = {}
        
        start_time = time.time()
        color_width, color_height = None, None
        # print(len(color_image_dict),self.curr_device_cnt)
        while len(color_image_dict) != self.curr_device_cnt:
            # print(len(color_image_dict))
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
                color_image_dict[serial_number] = color_image
            # for serial_number in self.depth_frames_queue.keys():
                depth_frame = None
                if not self.depth_frames_queue[serial_number].empty():
                    depth_frame = self.depth_frames_queue[serial_number].get()
                if depth_frame is None:
                    continue
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width()))
                depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()

                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
                # print("color_image.shape:", color_image.shape)
                # print("depth_image.shape:", depth_image.shape)
                depth_image_dict[serial_number] = depth_data
                # depth_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
                if serial_number == 'CP1L44P0004Y':
                    cv2.imwrite("color.png",color_image)
                    # cv2.imwrite("depth.png", depth_image)
                    np.save("depth.npy",depth_data)
                # image_dict[serial_number] = depth_image

        return color_image_dict,depth_image_dict,color_width, color_height
    def get_images(self):
        color_image_dict,depth_image_dict,color_width, color_height = self.rendering_frame()
        return color_image_dict,depth_image_dict,color_width, color_height
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


if __name__ == "__main__":
    camera = CAMERA_HOT_PLUG()
    right_camera_sn = 'CP1L44P0006E'
    while True:
        color_image_dict,depth_image_dict,color_width, color_height = camera.get_images()
        color_image_dict_np = np.array(color_image_dict[right_camera_sn],dtype=object)
        print(color_image_dict_np.shape)
        depth_image_dict_np = np.array(depth_image_dict[right_camera_sn],dtype=object)
        print(depth_image_dict_np.shape)
