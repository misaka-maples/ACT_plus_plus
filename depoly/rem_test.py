import sys
import os
from pathlib import Path
from visualize_episodes import visualize_joints
import cv2
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
from hdf5_edit import get_state
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from tqdm import tqdm
import time
from policy_test import get_action
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
        # joint_state_left = self.real_left_arm.Get_Current_Arm_State()
        self.joint_state_right = self.real_right_arm.Get_Current_Arm_State()
        # self.joint_state_left = joint_state_left[1]  #[22,1,,1]
        self.joint_state_right = self.joint_state_right[1]
        self.joint_state_right.append(self.real_right_arm.Get_Tool_Voltage()[1])
        # print(self.joint_state_right)
        return self.joint_state_right


def main(camera_image,qpos=None):
    # executor.add_node(arm_recorder)
    image_dict = camera_image
    qpos_list = QposRecorder.get_state()

    print(f"qposlist_first_state:{qpos_list[0]}")
    qpos_list = [math.radians(i)for i in qpos_list]

    config = {
        'image_dict': image_dict,
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
    action = get_action(config)
    actions = [i - 2 for i in action]
    actions[2] = -actions[2]
    # qpos_list = [math.degrees(i) for i in qpos_list]
    print(f"qpos list:{qpos_list}, actions:{actions}")
    return actions,qpos_list
    # try:
    #     executor.spin()
    # finally:
    #     camera_top_node.destroy_node()
    #     rclpy.shutdown()


if __name__ == "__main__":
    # get_env()

    qpos_list=[]
    actions_list=[]
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
    _,_,qpos_list_demo = get_state('/home/zhnh/Documents/xzx_projects/aloha_deploy/act-plus-plus/episode_7.hdf5')

    camera_top_node = ImageRecorder(
        "image_recorder_node", camera_config, is_debug=False)
    # arm_recorder = RealmanArmRecorder("right_arm_recorder")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(camera_top_node)

    QposRecorder = QposRecorder()
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

        actions,qpos = main(camera_image)
        qpos_list.append(qpos)
        actions_list.append(actions)

        # print(actions)
        # actions_list.append(actions)
        power = actions[6]
        actions = [math.degrees(i) for i in actions[:6]]
        # print(actions)
        # dip=1
        # power = action[6]
        # print(f"dip:{dip}")

        QposRecorder.real_right_arm.Movej_Cmd(actions, 10, 0, 0, True)

        if power > 2:

            QposRecorder.real_right_arm.Set_Tool_Voltage(3, True)
        else:
            QposRecorder.real_right_arm.Set_Tool_Voltage(0, True)
        print(len(qpos_list),len(actions_list))
        if i %10==0 and i!=0:
            visualize_joints(qpos_list, actions_list, "/home/zhnh/Documents/xzx_projects/aloha_deploy/act-plus-plus/depoly/rem_test_plot.png")

    # time.sleep(0.5)
