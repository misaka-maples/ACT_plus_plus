import os, datetime, sys
import threading
import time
import json
from pyorbbecsdk import *
import numpy as np
from queue import Queue
import cv2
from typing import Union, Any, Optional

# 获取当前脚本的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# 添加到 sys.path
sys.path.append(parent_dir)
config_file_path = os.path.join(os.path.dirname(__file__), "/workspace/config/multi_device_sync_config.json")
# from utils import frame_to_bgr_image

multi_device_sync_config = {}

MAX_DEVICES = 5  # 假设最多支持 5 台设备
MAX_QUEUE_SIZE = 1  # 最大帧队列长度

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # elif color_format == OBFormat.I420:
    #     image = i420_to_bgr(data, width, height)
    #     return image
    # elif color_format == OBFormat.NV12:
    #     image = nv12_to_bgr(data, width, height)
    #     return image
    # elif color_format == OBFormat.NV21:
    #     image = nv21_to_bgr(data, width, height)
    #     return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image
class CAMERA_HOT_PLUG:
    def __init__(self,resolution=[640,480],fps=30):
        self.mutex = threading.Lock()
        self.ctx = Context()
        self.device_list = self.ctx.query_devices()
        self.curr_device_cnt = self.device_list.get_count()
        self.pipelines: list[Pipeline] = []
        self.configs: list[Config] = []
        self.serial_number_list: list[str] = ["" for _ in range(self.curr_device_cnt)]
        self.color_frames_queue: dict[str, Queue] = {}
        self.depth_frames_queue: dict[str, Queue] = {}
        self.reolution = resolution
        self.fps = fps
        self.color_profile = object
        self.depth_profile = object
        # self.temporal_filter = TemporalFilter(alpha=0.5)  # Modify alpha based on desired smoothness

        self.setup_cameras()
        self.start_streams()
        print("相机初始化完成")
        self.multi_device_sync_config = multi_device_sync_config
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
            # print(serial_number,sync_config_json,multi_device_sync_config['serial_number'])
            sync_config = device.get_multi_device_sync_config()
            sync_config.mode = self.sync_mode_from_str(sync_config_json["config"]["mode"])
            sync_config.color_delay_us = sync_config_json["config"]["color_delay_us"]
            sync_config.depth_delay_us = sync_config_json["config"]["depth_delay_us"]
            sync_config.trigger_out_enable = sync_config_json["config"]["trigger_out_enable"]
            sync_config.trigger_out_delay_us = sync_config_json["config"]["trigger_out_delay_us"]
            sync_config.frames_per_trigger = sync_config_json["config"]["frames_per_trigger"]
            device.set_multi_device_sync_config(sync_config)

            try:
                profile_list: StreamProfileList = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                # color_profile = profile_list.get_default_video_stream_profile()
                self.color_profile:StreamProfile = profile_list.get_video_stream_profile(self.reolution[0],self.reolution[1],OBFormat.RGB,self.fps)
                config.enable_stream(self.color_profile)

                profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                # depth_profile = profile_list.get_default_video_stream_profile()
                self.depth_profile:StreamProfile = profile_list.get_video_stream_profile(self.reolution[0],self.reolution[1],OBFormat.Y16,self.fps)
                config.enable_stream(self.depth_profile)

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
            if type(frames).__name__ == "Frame":
                frames = frames.as_frame_set()
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

                depth_data = np.frombuffer(np.ascontiguousarray(depth_frame.get_data()), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width()))
                depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()

                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)
                # print("color_image.shape:", color_image.shape)
                # print("depth_image.shape:", depth_image.shape)
                depth_image_dict[serial_number] = depth_data
                # depth_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
                # if serial_number == 'CP1L44P0004Y':
                #     cv2.imwrite("color.png",color_image)
                #     # cv2.imwrite("depth.png", depth_image)
                #     np.save("depth.npy",depth_data)
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
            multi_device_sync_config[device['serial_number']] = device
            print(f"📷 Device {device['serial_number']}: {device['config']['mode']}")

    def get_color_intrinsic_matrix(self):
        arg:np.array = np.eye(3)
        intrinsic:OBCameraIntrinsic = self.color_profile.get_intrinsic()
        arg[0,0]=intrinsic.fx
        arg[1,1]=intrinsic.fy
        arg[0,2]=intrinsic.cx
        arg[1,2]=intrinsic.cy
        return arg
if __name__ == "__main__":
    camera = CAMERA_HOT_PLUG(fps=30)
    right_camera_sn = 'CP1L44P0004Y'
    # print(camera.multi_device_sync_config)
    # time.sleep(1)
#     [[612.32965088   0.         643.00286865]
#  [  0.         612.49316406 360.31613159]
#  [  0.           0.           1.        ]]
    while True:

        color_image_dict,depth_image_dict,color_width, color_height = camera.get_images()
        color_image_dict_np = np.array(color_image_dict[right_camera_sn],dtype=np.uint8)
        # print(color_image_dict_np.shape)
        # cv2.imwrite("color.png",color_image_dict_np)
        cv2.imshow("color",color_image_dict_np)
        cv2.waitKey(1)
        # depth_image_dict_np = np.array(depth_image_dict[right_camera_sn])
        # np.save("depth.npy",depth_image_dict_np)
        # print(depth_image_dict_np.shape)
        # arg = camera.get_color_intrinsic_matrix()
        # print(arg)
