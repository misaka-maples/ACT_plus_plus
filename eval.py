# from datetime import datetime
import os, datetime, sys
import threading
import time
import json
import threading
import numpy as np
import cv2
from queue import Queue

MAX_DEVICES = 5  # 假设最多支持 5 台设备
MAX_QUEUE_SIZE = 10  # 最大帧队列长度
multi_device_sync_config = {}
config_file_path = os.path.join(os.path.dirname(__file__), "./config/multi_device_sync_config.json")
from networkx.readwrite.json_graph.tree import tree_data
from deploy.hdf5_edit_utils import Modify_hdf5
from deploy.policy_action_generation import ActionGenerator
import numpy as np
import math
from tqdm import tqdm
from deploy.visualize_episodes import visualize_joints
from tcp_tx import PersistentClient
current_time = datetime.datetime.now()
# datetime.
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
STATE_NAMES = JOINT_NAMES +["gripper_pos"]+ ["gripper_force"]
camera_names = ['top', 'right_wrist','left_wrist']

class eval:
    def __init__(self,real_robot=False,data_true=False):
        self.real_robot = real_robot
        self.data_true = data_true
        if self.real_robot:
            self.camera = CAMERA_HOT_PLUG()
            self.persistentClient = PersistentClient('192.168.3.15', 8001)
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

        if self.data_true:
            self.gp_contrpl = GPCONTROL()
            self.gp_contrpl.start()
            loop_len = 200
            task_complete_step = None
            square_size = 100
        else:
            data_dict = Modify_hdf5()
            dict_ = data_dict.check_hdf5(r'/data/ACT/episode/hdf5_file_exchange_5-29/episode_1.hdf5')
            # print(dict_["action"].shape)
            loop_len = len(dict_['top'])
        config = {
            'ckpt_dir': r'/data/ACT/episode/hdf5_file_exchange_5-29/act_8p',
            'max_timesteps': loop_len,
            'ckpt_name': "policy_step_100_seed_0_0.ckpt",
            'backbone': 'resnet50',
            'temporal_agg':True,
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
                    left_qpos = self.persistentClient.get_arm_position_joint(1)
                    right_qpos = self.persistentClient.get_arm_position_joint(2)
                    left_gp ,right_gp= self.gp_contrpl.state
                    # left_pos=left_gp[1]
                    # left_force = left_gp[2]
                    # right_pos=right_gp[1]
                    # right_force = right_gp[2]
                    # print(left_gp)
                    radius_qpos = [math.radians(j) for j in left_qpos]
                    gpstate, gppos, gpforce = map(lambda x: str(x) if not isinstance(x, str) else x, left_gp)
                    radius_qpos.extend([int(gppos, 16), int(gpforce, 16)])
                    radius_qpos.extend([math.radians(j) for j in right_qpos])
                    gpstate, gppos, gpforce = map(lambda x: str(x) if not isinstance(x, str) else x, right_gp)
                    radius_qpos.extend([int(gppos, 16), int(gpforce, 16)])
                    # radius_qpos.append(left_pos)
                    # radius_qpos.append(left_force)
                    # radius_qpos.append(right_qpos)
                    # radius_qpos.append(right_pos)
                    # radius_qpos.append(right_force)
                    # print(radius_qpos)
                    # if self.is_close(self.persistentClient.get_arm_position_joint(1)[:3],[-121.42, -599.741, -209.687]) or i >30:
                    #     task_complete_step = 1
                    for camera_name in camera_names:
                        image_dict[camera_name] = self.image[camera_name]
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
            # print(image_dict)
            ActionGeneration.image_dict = image_dict
            ActionGeneration.qpos_list = radius_qpos
            actions = ActionGeneration.get_action()
            # print(qpos)
            # print(list(np.degrees(actions)))
            left_arm_action = np.rad2deg(actions[:6])
            right_arm_action= np.rad2deg(actions[8:14])
            left_gp = actions[6:8]*0.4
            right_gp = 120-actions[14:16]
            # print(left_gp,right_gp)
            # print(actions[6:8],actions[14:16])
            # print(left_arm_action,right_arm_action)
            if self.real_robot:
                if right_arm_action[5]>360:
                    right_arm_action[5]=358
                self.persistentClient.set_arm_position(list(right_arm_action), "joint", 2)
                self.persistentClient.set_arm_position(list(left_arm_action), "joint", 1)
                print(int(left_gp[0]),int(right_gp[0]))
                self.gp_contrpl.state_data_1=int(left_gp[0])
                self.gp_contrpl.state_data_2=int(right_gp[0])
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
