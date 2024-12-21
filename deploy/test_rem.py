imporimport sys
import os
from pathlib import Path
import datetime
import math
import logging
from collections import deque

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from tqdm import tqdm
import time

from visualize_episodes import visualize_joints
from hdf5_edit import get_state
from policy_test import ActionGenerator as OriginalActionGenerator
from robotic_arm_package.robotic_arm import *
from image_recorder_ros2 import ImageRecorder

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_env():
    root_path = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_path))

    detr_path = os.path.join(root_path, 'detr')
    sys.path.append(str(detr_path))

    robomimic_path = os.path.join(root_path, 'robomimic', 'robomimic')
    sys.path.append(str(robomimic_path))

get_env()

class SmoothActionGenerator(OriginalActionGenerator):
    def __init__(self, config, smooth_window=5):
        super().__init__(config)
        self.smooth_window = smooth_window
        self.action_history = deque(maxlen=smooth_window)

    def get_action(self):
        raw_action = super().get_action()
        self.action_history.append(raw_action)
        smoothed_action = [sum(actions) / len(actions) for actions in zip(*self.action_history)]
        return smoothed_action

class QposRecorder:
    def __init__(self):
        self.joint_state_right = None
        self.joint_state_left = None
        self.real_right_arm = Arm(RM65, "192.168.1.18")

    def get_state(self):
        try:
            self.joint_state_right = self.real_right_arm.Get_Current_Arm_State()
            self.joint_state_right = self.joint_state_right[1]
            self.joint_state_right.append(self.real_right_arm.Get_Tool_Voltage()[1])
            # 检查关节状态是否在合理范围内（假设-π到π）
            if all(-math.pi <= angle <= math.pi for angle in self.joint_state_right):
                return self.joint_state_right
            else:
                logging.error("关节状态超出合理范围")
                return []
        except Exception as e:
            logging.error(f"获取关节状态时发生错误: {e}")
            return []

def is_valid_action(actions):
    # 假设机械臂每个关节的有效范围在 -180 到 180 度
    for action in actions:
        if not (-180 <= action <= 180):
            return False
    return True

class ActionPublisherNode(Node):
    def __init__(self):
        super().__init__('action_publisher_node')
        camera_config = {
            "top": {
                "topic": "/camera_01/color/image_raw",
                "qos": 10  # 提高QoS等级
            },
            "right_wrist": {
                "topic": "/camera_02/color/image_raw",
                "qos": 10
            }
        }
        self.camera_top_node = ImageRecorder("image_recorder_node", camera_config, is_debug=False)
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.camera_top_node)

        config = {
            'image_dict': {},
            'qpos_list': [],
            'eval': True,
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
        self.ActionGen = SmoothActionGenerator(config)
        self.QposRecorder = QposRecorder()

        self.qpos_list_ = []
        self.actions_list = []

        # 创建定时器，每0.5秒调用一次回调函数
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.step = 0

    def timer_callback(self):
        if self.step >= 2000:
            self.get_logger().info("已完成所有步骤。")
            self.destroy_timer(self.timer)
            return

        start_time = time.time()
        self.get_logger().info(f"步骤开始: {self.step}")

        # 执行器轮询
        self.executor.spin_once(timeout_sec=0.1)  # 短时间轮询以处理事件
        self.get_logger().info("完成 spin_once")

        # 获取图像数据
        self.get_logger().info("获取图像数据")
        camera_image = self.camera_top_node.get_images()
        self.get_logger().debug(f"获取到的图像: {camera_image}")

        if camera_image['top'] is None或camera_image['right_wrist'] is None:
            self.get_logger().warning("图像数据不完整，跳过当前循环")
            return  # 等待下一个定时器触发
        else:
            self.get_logger().info("显示图像")
            cv2.imshow("right_wrist", camera_image['right_wrist'])
            cv2.waitKey(1)

        # 获取关节状态并生成动作
        self.get_logger().info("获取关节状态")
        qpos = self.QposRecorder.get_state()
        if not qpos:
            self.get_logger().warning("获取到的关节状态为空或超出范围，跳过当前循环")
            return
        self.get_logger().debug(f"当前关节状态 (弧度): {qpos}")
        self.ActionGen.qpos_list = qpos

        try:
            self.get_logger().info("生成动作")
            actions = self.ActionGen.get_action()
            self.get_logger().debug(f"生成的动作 (原始): {actions}")
            if not isinstance(actions, list)或len(actions) < 7:
                self.get_logger().error(f"无效的动作输出: {actions}")
                return
        except Exception as e:
            self.get_logger().error(f"生成动作时发生错误: {e}")
            return

        # 调整动作
        actions = [angle - 2 for angle in actions]
        actions[2] = -actions[2]
        self.get_logger().info(f"调整后的动作: {actions}")

        # 验证动作有效性
        if not is_valid_action(actions):
            self.get_logger().error(f"调整后的动作超出有效范围: {actions}")
            return

        self.qpos_list_.append(qpos)
        self.actions_list.append(actions)
        power = actions[6]
        actions_deg = [math.degrees(angle) for angle in actions[:6]]

        # 执行动作
        try:
            self.get_logger().info("执行动作")
            self.QposRecorder.real_right_arm.Movej_Cmd(actions_deg, 10, 0, 0, True)
        except Exception as e:
            self.get_logger().error(f"执行动作时发生错误: {e}")
            return

        # 设置工具电压
        self.get_logger().info("设置工具电压")
        try:
            if power > 2:
                self.QposRecorder.real_right_arm.Set_Tool_Voltage(3, True)
            else:
                self.QposRecorder.real_right_arm.Set_Tool_Voltage(0, True)
        except Exception as e:
            self.get_logger().error(f"设置工具电压时发生错误: {e}")

        # 数据可视化和保存
        try:
            if self.step % 10 == 0 and self.step != 0:  # 降低保存频率
                current_time = datetime.datetime.now()  # 更新当前时间
                path_save_image = os.path.join(
                    "/home/zhnh/Documents/xzx_projects/aloha_deploy/act-plus-plus/deploy",
                    "deploy_image",
                    current_time.strftime("%m-%d_%H-%M") + ".png"
                )
                visualize_joints(self.qpos_list_, self.actions_list, path_save_image)
                self.get_logger().info(f"保存图像和动作数据到: {path_save_image}")
        except Exception as e:
            self.get_logger().error(f"保存图像和动作数据时发生错误: {e}")

        end_time = time.time()
        loop_duration = end_time - start_time
        self.get_logger().info(f"循环 {self.step} 耗时: {loop_duration:.2f} 秒")

        self.step += 1

def main(args=None):
    rclpy.init(args=args)
    node = ActionPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点已被手动终止。")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
