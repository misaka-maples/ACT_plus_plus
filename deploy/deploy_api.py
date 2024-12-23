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
# from decorator import EMPTY
from pyorbbecsdk import *
from tqdm import tqdm

from utils import frame_to_bgr_image, visualize_joints
# from pynput.keyboard import Listener, Key
from policy_action_generation import ActionGenerator
frames_queue_lock = Lock()
import datetime

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
config_file_path = os.path.join(os.path.dirname(__file__), "/home/zhnh/Documents/project/act_arm_project/pyorbbecsdk-main/config/multi_device_sync_config.json")
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
        for device_index in range(curr_device_cnt):
            with frames_queue_lock:
                # 尝试从队列中获取帧，如果队列为空则返回None
                frames = frames_queue[device_index].get() if not frames_queue[device_index].empty() else None
            if frames is None:
                continue  # 如果没有帧，跳过当前循环
            color_frame = frames.get_color_frame() if frames else None
            if color_frame:
                # 将彩色帧转换为BGR图像
                color_image = frame_to_bgr_image(color_frame)
                color_image = cv2.resize(color_image, (640, 480))
                images[device_index] = color_image
            if  len(images) == 2:
                all_frames_processed = True  # 如果队列为空，设置标志为True
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

def on_new_frame_callback(frames: FrameSet, index: int):
    global frames_queue
    global MAX_QUEUE_SIZE
    assert index < MAX_DEVICES
    with frames_queue_lock:
        if frames_queue[index].qsize() >= MAX_QUEUE_SIZE:
            frames_queue[index].get()
        frames_queue[index].put(frames)

max_timesteps=0
def main():
    global curr_device_cnt, max_timesteps, qpos_list, images_dict, QposRecorder
    read_config(config_file_path)
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("No device connected")
        return
    if device_list.get_count() == 1:
        print("1 device connected")
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
    save_image_dir = os.path.join(os.getcwd(), "..", "color_images")
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)
    start_streams(pipelines, configs)
    max_timesteps = 100
    qpos_list = []
    images_dict = {cam_name: [] for cam_name in camera_names}  # 用于存储每个相机的图片
    actions_list = []
    qpos_list_ = []
    config = {
        'eval': True,  # 表示启用了 eval 模式（如需要布尔类型，直接写 True/False）
        'task_name': 'train',
        'ckpt_dir': r'/home/zhnh/Documents/project/act_arm_project/models/auto_1_12-21',
        'policy_class': 'ACT',
        'chunk_size': 210,
        'backbone': 'dino_v2',
    }
    ActionGenerator1= ActionGenerator(config)
    global stop_processing
    try:
        for i in tqdm(range(max_timesteps)):
            # 创建并启动监听器
            start = time.time()
            print(f"\n"
                  f"---------------------------------------------episode{i}--------------------------------------------------------\n")
            image = process_frames(pipelines)
            angle_qpos = posRecorder.get_state()
            radius_qpos = [math.radians(j) for j in angle_qpos]
            radius_qpos[6] = posRecorder.real_right_arm.rm_get_tool_voltage()[1]
            # print(radius_qpos[6])
            qpos_list.append(radius_qpos)
            images_dict['right_wrist']=image[0]
            images_dict['top']=image[1]
            cv2.imwrite(os.path.join(save_image_dir,current_time.strftime("%m-%d %H:%M") + f"top{i}.png"), np.array(images_dict['top']))
            cv2.imwrite(os.path.join(save_image_dir,current_time.strftime("%m-%d %H:%M") + f"right{i}.png"), np.array(images_dict['right_wrist']))
            ActionGenerator1.image_dict = images_dict
            ActionGenerator1.qpos_list = radius_qpos
            # cv2.imshow("top" ,image[0])
            # # cv2.imshow("Device {}".format(i), image)
            # key = cv2.waitKey(1)
            # if key == ord('q') or key == ESC_KEY:
            #     return

            now = time.time()

            actions = ActionGenerator1.get_action()
            # actions = [i - 2 for i in actions]
                # actions[2] = -actions[2]
            step_caculate = time.time()
            qpos_list_.append(ActionGenerator1.qpos_list)
            actions_list.append(actions)
            power = actions[6]
            print(power,step_caculate-now, step_caculate - start)
            actions = [math.degrees(i) for i in actions[:6]]
            print(f":---------------------------------------actions--------------------"
                  f"--------------------------:\n{actions}")
            posRecorder.real_right_arm.rm_movej(actions, 10, 0, 0, 1)
            angle_qpos[6] = posRecorder.real_right_arm.rm_get_tool_voltage()[1]
            # print(angle_qpos[:6], actions)

            # result_action = interpolate_with_step_limit(angle_qpos[:6], actions, 2)
            # print(f"result-action:{result_action}")
            # for i in result_action:
            #     print(f"i:", i)
            #     posRecorder.real_right_arm.rm_movej_canfd(i,False)
                # time.sleep(0.01)
            if power > 3:
                posRecorder.real_right_arm.rm_set_tool_voltage(3)
            else:
                posRecorder.real_right_arm.rm_set_tool_voltage(0)
            # print(len(qpos_list),len(actions_list))
    except KeyboardInterrupt:
        print("Interrupted by user")
        stop_processing = True
    finally:
        print("===============Stopping pipelines==============")
        path_save_image = os.path.join(r"/home/zhnh/Documents/project/act_arm_project/deploy",
                                           "deploy_image", current_time.strftime("%m-%d %H:%M") + ".png")
        visualize_joints(qpos_list_, actions_list, path_save_image)
        stop_streams(pipelines)
def interpolate_with_step_limit(array1, array2, step=10):
    result = []
    for i in range(len(array1)):
        start = array1[i]
        end = array2[i]
        current = start

        result.append([current])

        while abs(current - end) > step:
            if current < end:
                current += step
            else:
                current -= step

            result[i].append(current)

        # Append the final endpoint to ensure it matches array2
        if current != end:
            result[i].append(end)
    # print(result)
    max_length = max(len(sub_array) for sub_array in result)
    for sub_array in result:
        # sub_array = list(sub_array)  # 确保是列表
        # 如果长度不足，则用最后一个元素填充
        while len(sub_array) < max_length:
            sub_array.append(sub_array[-1])
    actions = [[] for _ in range(max_length)]
    for j in range(max_length):
        for i in result:
            # print(j)
            actions[j].append(i[j])
    # print(actions)
    return actions

if __name__ == "__main__":
    current_time = datetime.datetime.now()
    max_timesteps = 3000
    start = time.time()
    main()
    end = time.time()
    print(f"total time in 1 roll:{end-start}")