import sys
import os
from pathlib import Path
# from test_qpos_image import qpos_list
import datetime

from sympy.physics.units import current

current_time = datetime.datetime.now()
def get_env():
    root_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_path))
    # print(root_path)

    detr_path = os.path.join(root_path, 'detr')
    sys.path.append(str(detr_path))
    # print(detr_path)

    robomimic_path = os.path.join(root_path, 'robomimic', 'robomimic')
    sys.path.append(str(robomimic_path))
    # print(robomimic_path)
get_env()

from visualize_episodes import visualize_joints
import cv2
# get_env()
from hdf5_edit import get_state
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from tqdm import tqdm
import time
from policy_test import ActionGenerator
from robotic_arm_package.robotic_arm import *
import math
from image_recorder_ros2 import ImageRecorder
# def get_env():

class QposRecorder:
    def __init__(self, ):
        self.joint_state_right = None
        self.joint_state_left = None
        # self.real_left_arm = Arm(RM65, "192.168.1.16")
        self.real_right_arm = Arm(RM65, "192.168.1.18")

    def get_state(self):
        # joint_state_left = self.refrom visualize_episodes import visualize_joints
        # import cv2al_left_arm.Get_Current_Arm_State()
        self.joint_state_right = self.real_right_arm.Get_Current_Arm_State()
        # self.joint_state_left = joint_state_left[1]  #[22,1,,1]
        self.joint_state_right = self.joint_state_right[1]
        self.joint_state_right.append(self.real_right_arm.Get_Tool_Voltage()[1])
        # print(self.joint_state_right)
        return self.joint_state_right


if __name__ == "__main__":
    rclpy.init()
    cv2.namedWindow("right_wrist", cv2.WINDOW_NORMAL)
    camera_config = {
        "top": {
            "topic": "/camera_01/color/image_raw",
            "qos": 1
        },
        # "camera_left_wrist": {
        #     "topic": "/camera_02/color/image_raw",
        #     "qos": 1
        # },
        "right_wrist": {
            "topic": "/camera_02/color/image_raw",
            "qos": 1
        }
    }
    # _,_,qpos_list_demo = get_state('/home/zhnh/Documents/xzx_projects/aloha_deploy/act-plus-plus/episode_7.hdf5')
    camera_top_node = ImageRecorder(
        "image_recorder_node", camera_config, is_debug=False)
    # arm_recorder = RealmanArmRecorder("right_arm_recorder")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(camera_top_node)
    executor.spin_once(timeout_sec=1)
    camera_image = camera_top_node.get_images()
    QposRecorder = QposRecorder()
    qpos_list = [math.radians(i) for i in QposRecorder.get_state()]
    actions_list = []
    qpos_list_ = []
    config = {
        'image_dict': camera_image,
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
        'num_queries': 100,
    }
    ActionGenerator= ActionGenerator(config)
    for i in range(2000):
        print(f"steps",i)
        executor.spin_once(timeout_sec=1)
        camera_image = camera_top_node.get_images()

        if camera_image['top'] is None or camera_image['right_wrist'] is None:
            continue
        else:
            # print(camera_image['top'])
            cv2.imshow("right_wrist", camera_image['right_wrist'])
            cv2.waitKey(1)
        # qpos = qpos_list_demo[i]
        ActionGenerator.image_dict=camera_image
        ActionGenerator.qpos_list=[math.radians(i)for i in QposRecorder.get_state()]
        actions= ActionGenerator.get_action()
        actions = [i - 2 for i in actions]
        actions[2] = -actions[2]
        print(f":-----------------------------actions------------------------------------:\n{actions}")
        qpos_list_.append(ActionGenerator.qpos_list)
        actions_list.append(actions)
        power = actions[6]
        actions = [math.degrees(i) for i in actions[:6]]
        QposRecorder.real_right_arm.Movej_Cmd(actions, 10, 0, 0, True)
        if power > 2:
            QposRecorder.real_right_arm.Set_Tool_Voltage(3, True)
        else:
            QposRecorder.real_right_arm.Set_Tool_Voltage(0, True)
        # print(len(qpos_list),len(actions_list))
        if i %1==0 and i!=0:
            path_save_image = os.path.join("/home/zhnh/Documents/xzx_projects/aloha_deploy/act-plus-plus/deploy","deploy_image",current_time.strftime("%m-%d %H:%M")+".png")
            print(path_save_image)
            visualize_joints(qpos_list_, actions_list, path_save_image)

    # time.sleep(0.5)
