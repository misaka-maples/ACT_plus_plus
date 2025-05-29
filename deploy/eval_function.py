# from datetime import datetime

import os, datetime, sys
# 获取当前脚本的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# 添加到 sys.path
sys.path.append(parent_dir)
import threading
import time
import json
import threading
import serial
import numpy as np
import cv2
from queue import Queue
from pyorbbecsdk import *
from utils import frame_to_bgr_image
from camera_hot_plug import CAMERA_HOT_PLUG
from gpcontrol import GPCONTROL
MAX_DEVICES = 5  # 假设最多支持 5 台设备
MAX_QUEUE_SIZE = 10  # 最大帧队列长度
multi_device_sync_config = {}

config_file_path = os.path.join(os.path.dirname(__file__), "../config/multi_device_sync_config.json")
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
complete_sign = 0
complete_sign_0 = 0
tz = pytz.timezone('Asia/Shanghai')
current_time = datetime.datetime.now(tz)
# datetime.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
STATE_NAMES = JOINT_NAMES +["gripper_pos"]+ ["gripper_force"]
camera_names = ['top', 'right_wrist','left_wrist']

class eval:
    def __init__(self,camera,persistentClient,gp_contrpl,real_robot=False,data_true=False,ckpt_dir=r'/workspace/exchange/4-24/act',ckpt_name="policy_step_78000_seed_0.ckpt",hdf5_path=r'/workspace/exchange/4-24/hdf5_file_exchange/episode_6.hdf5',state_dim=8,temporal_agg=False):
        self.real_robot = real_robot
        self.data_true = data_true
        self.ckpt_dir = ckpt_dir
        self.ckpt_name=ckpt_name
        self.hdf5_path = hdf5_path
        self.state_dim = state_dim
        self.temporal_agg = temporal_agg
        if self.real_robot:
            self.camera:CAMERA_HOT_PLUG = camera
            self.persistentClient:PersistentClient = persistentClient
            self.gp_contrpl:GPCONTROL = gp_contrpl
            # self.gp_contrpl.start()
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
        global complete_sign,complete_sign_0
        actions_list = []
        qpos_list_ = []
        loss = []
        self.radius_qpos_list = []

        if self.data_true:

            loop_len = 100
            task_complete_step = None
            square_size = 100
        else:
            data_dict = Modify_hdf5()
            dict_ = data_dict.check_hdf5(self.hdf5_path)
            # print(dict_["action"].shape)
            loop_len = len(dict_['top'])
        config = {
            'ckpt_dir': self.ckpt_dir,
            'max_timesteps': loop_len,
            'ckpt_name': self.ckpt_name,
            'backbone': 'resnet18',
            'temporal_agg':self.temporal_agg,
        }
        image_dict = {i:[] for i in camera_names}
        # print(image_dict)
        name_to_serial = {
            v['config']['camera_name']: serial for serial, v in self.camera.multi_device_sync_config.items()
        }
        ActionGeneration = ActionGenerator(config)
        for i in tqdm(range(loop_len)):
            # print(f"roll:{i}")
            ActionGeneration.t = i
            if self.real_robot:
                if self.data_true:
                    # self.updata_frame()
                    left_qpos = self.persistentClient.get_arm_position_joint(1)
                    right_qpos = self.persistentClient.get_arm_position_joint(2)
                    left_gp ,right_gp= self.gp_contrpl.state
                    print(left_gp ,right_gp)
                    # left_pos=left_gp[1]
                    # left_force = left_gp[2]
                    # right_pos=right_gp[1]
                    # right_force = right_gp[2]
                    # print(left_gp)
                    # radius_qpos = [math.radians(j) for j in left_qpos]
                    radius_qpos = left_qpos
                    gpstate, gppos, gpforce = map(lambda x: str(x) if not isinstance(x, str) else x, left_gp)
                    radius_qpos.extend([int(gppos, 16), int(gpforce, 16)])
                    # radius_qpos.extend([math.radians(j) for j in right_qpos])
                    radius_qpos.extend(right_qpos)
                    gpstate, gppos, gpforce = map(lambda x: str(x) if not isinstance(x, str) else x, right_gp)
                    radius_qpos.extend([int(gppos, 16), int(gpforce, 16)])
                    color_image_dict,depth_image_dict,color_width, color_height = camera.get_images()
                    for camera_name in camera_names:
                        serial_number = name_to_serial.get(camera_name)
                        if serial_number and serial_number in color_image_dict:
                            self.image[camera_name] = np.array(color_image_dict[serial_number], dtype=np.uint8)
                            image_dict[camera_name] = self.image[camera_name]
                            # print(image_dict[camera_name].shape)    
                        # qpos = self.persistentClient.get_arm_position_joint(1)
                        # radius_qpos = [math.radians(j) for j in qpos]
                        # img_copy = [row[:] for row in image_dict[camera_name]]  # 深拷贝，防止改到原图
                        # height = len(img_copy)
                        # width = len(img_copy[0])
                        # print(height,width)
                        # square_color = [0, 0, 255] if task_complete_step is  None else [0, 255, 0]  
                        # if square_color == [0,0,255]:
                        #     print("红色")
                        # elif  square_color == [0, 255, 0]:
                        #     print("绿色")
                        # # 左下角：行范围 [height - square_size, height)
                        # for row in range(height - square_size, height):
                        #     for col in range(square_size):
                        #         if 0 <= row < height and 0 <= col < width:
                        #             img_copy[row][col] = square_color
                        # image_dict[camera_name] = np.array(img_copy)
                else:
                    for camera_name in camera_names:
                        image_dict[camera_name]=np.array(dict_[camera_name][i])
                        # qpos = self.persistentClient.get_arm_position_joint(1)
                        # radius_qpos = [math.radians(j) for j in qpos]
                        radius_qpos = dict_['qpos'][i]
            else:
                
                for camera_name in camera_names:
                    image_dict[camera_name] = np.array(dict_[camera_name][i])
                radius_qpos = dict_['qpos'][i]
            # print(np.array(radius_qpos).shape)

            self.radius_qpos_list.append(radius_qpos)
            # print(radius_qpos)
            # print(image_dict)
            ActionGeneration.image_dict = image_dict
            ActionGeneration.qpos_list = radius_qpos
            actions = ActionGeneration.get_action()
            # print(qpos)
            # print(list(np.degrees(actions)))
            left_arm_action = actions[:6]
            right_arm_action = actions[8:14]
            # left_arm_action = np.rad2deg(actions[:6])
            # right_arm_action= np.rad2deg(actions[8:14])
            left_gp = actions[6:8]
            right_gp = actions[14:16]
            # print(left_gp,right_gp)
            # print(actions[6:8],actions[14:16])
            print(left_arm_action,left_gp,right_arm_action,right_gp)
            if self.real_robot:
                # if right_arm_action[5]>360:
                #     right_arm_action[5]=358
                if self.is_close(self.persistentClient.get_arm_position_pose(1),[-129.127, -810.615, -288.951, 2.4716, -0.00248988, 2.28385],1) and complete_sign == 0:
                    complete_sign = 1
                    print("第一段已到位")
                    break
                # if self.is_close(self.persistentClient.get_arm_position_joint(1),[-101.05369  , -77.50083 ,   53.14054  ,  14.273354 , -61.44039 ,   19.2856  ],0.5) and complete_sign == 0:
                #     complete_sign = 1
                #     print("第一段已到位")
                #     break
                if self.is_close(self.persistentClient.get_arm_position_pose(1),[629.137, -161.689, 590.811, 1.6477, 1.38221, 2.1665],0.5) and complete_sign == 1:
                    # complete_sign = 1
                    complete_sign_0 = 1
                    print("第二段已到位")
                    if self.is_close(self.persistentClient.get_arm_position_joint(1),[-99.9248, -44.9142, 13.987, -5.93674, -49.1103, -33.6326],0.5):

                        break
                # print(f"左手位置：{self.persistentClient.get_arm_position_pose(1)},{complete_sign}")
                # print(type(right_gp),type(right_arm_action))
                if right_arm_action.size == 0:
                    pass
                else:
                    self.persistentClient.set_arm_position(list(right_arm_action), "joint", 2)
                self.persistentClient.set_arm_position(list(left_arm_action), "joint", 1)
                if right_gp.size == 0:
                    self.gp_contrpl.set_state([int(left_gp[0]),0])
                else:
                    self.gp_contrpl.set_state([int(left_gp[0]),int(right_gp[0])])
            actions_list.append(actions)
        today = current_time.strftime("%m-%d-%H-%M")
        path_save_image = os.path.join(os.getcwd(), "deploy_image", f"{today}")
        os.makedirs(path_save_image, exist_ok=True)
        image_path = os.path.join(path_save_image, config['backbone']+"_"+ os.path.splitext(config['ckpt_name'])[0]+ ".png")
        loss_apth = os.path.join(path_save_image, 'loss' + current_time.strftime("%m-%d+8-%H-%M") + ".png")
        radius_qpos_list_ = [row[:self.state_dim] for row in self.radius_qpos_list]
        visualize_joints(radius_qpos_list_, actions_list, image_path, STATE_NAMES=STATE_NAMES)
        print("执行完成")
# complete_sign = 0
if __name__ == '__main__':
    camera = CAMERA_HOT_PLUG(fps=30)
    
    robot = PersistentClient('192.168.3.15', 8001)
    gpcontrol = GPCONTROL()
    gpcontrol.start()
    time.sleep(3)
    eval(camera=camera,
         persistentClient=robot,
         gp_contrpl=gpcontrol,
         real_robot=True,
         data_true=False,
         ckpt_dir=r'/workspace/exchange/5-9/exchange/act_overwrited',
         ckpt_name='policy_step_10000_seed_0.ckpt',
         hdf5_path=r'/workspace/exchange/5-9/exchange/episode_22.hdf5',
         state_dim=16,
         temporal_agg=True)
    # time.sleep(2)
    print("第二段")
    eval(camera=camera,
         persistentClient=robot,
         gp_contrpl=gpcontrol,
         real_robot=True,
         data_true=False,
         ckpt_dir=r'/workspace/exchange/5-9/duikong/act',
         ckpt_name='policy_best.ckpt',
         hdf5_path=r'/workspace/exchange/5-9/duikong/episode_23.hdf5',
         state_dim=8,
         temporal_agg=True)
