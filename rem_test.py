import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from tqdm import tqdm
import time
from policy_test import main1
from robotic_arm_package.robotic_arm import *

class ImageRecorder(Node):
    def __init__(self, name, config, is_debug=False):
        super().__init__(name)

        self.get_logger().info("Begin initializing ImageRecorde Node... ...")

        self.is_debug = is_debug
        # convert sensor_msgs.msg.Image to cv2
        self.bridge = CvBridge()
        # collect camera_names
        self.camera_names = []
        # count cameras num
        self.camera_counts = 0

        self.init_done = False

        for camera, sub_config in config.items():
            # print(f"{camera} : {sub_config}")

            self.camera_counts += 1
            self.camera_names.append(camera)

            setattr(self, f'{camera}_image', None)
            setattr(self, f'{camera}_sec', None)
            setattr(self, f'{camera}_nsec', None)
            setattr(self, f'{camera}_init_done', False)

            if (camera == "camera_top"):
                # callback_func = self.image_callback_top
                # Sub
                self._subscription = self.create_subscription(
                    Image, sub_config['topic'], self.image_callback_top, sub_config['qos'])
            elif (camera == "camera_left_wrist"):
                # callback_func = self.image_callback_left_wrist
                self._subscription = self.create_subscription(
                    Image, sub_config['topic'], self.image_callback_left_wrist, sub_config['qos'])
            elif (camera == "camera_right_wrist"):
                self._subscription = self.create_subscription(
                    Image, sub_config['topic'], self.image_callback_right_wrist, sub_config['qos'])
            else:
                raise NotImplementedError

        # self.init_timer = self.create_timer(3, self.init_cameras)

        self.get_logger().info('Success initializing image_recorder_node!\n\
        #                         Waiting for camera initialization... ...\n')
        # self.init_cameras()

    def init_cameras(self):
        done_count = 0
        for cam in self.camera_names:
            if (getattr(self, f'{cam}_init_done')):
                self.get_logger().info(f'camere: {cam} init done!... ...')
                done_count += 1
            else:
                self.get_logger().info(f'camere: {cam} not ready... ...')
            if done_count == self.camera_counts:
                self.init_done = True
        if self.init_done:
            self.get_logger().info("All cameras init done!")
            self.init_timer.cancel()

    def image_callback(self, cam_name, data):
        setattr(self, f'{cam_name}_timestamp', data.header.stamp)
        setattr(self, f'{cam_name}_sec', data.header.stamp.sec)
        setattr(self, f'{cam_name}_nsec', data.header.stamp.nanosec)

        if (data is not None):
            setattr(self, f'{cam_name}_init_done', True)

        if self.is_debug:
            timestamp = getattr(self, f'{cam_name}_timestamp')
            self.get_logger().debug(f"Get {cam_name} data:\ntimestamp: {timestamp}\nimage_size: {data}")

    def image_callback_top(self, data):
        cam_name = 'camera_top'
        setattr(self, f'{cam_name}_data', data)
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        return self.image_callback(cam_name=cam_name, data=data)

    def image_callback_left_wrist(self, data):
        cam_name = 'camera_left_wrist'
        setattr(self, f'{cam_name}_data', data)
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        return self.image_callback(cam_name=cam_name, data=data)

    def image_callback_right_wrist(self, data):
        cam_name = 'camera_right_wrist'
        setattr(self, f'{cam_name}_data', data)
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        return self.image_callback(cam_name=cam_name, data=data)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict
class QposRecorder():
    def __init__(self, ):
        real_left_arm = Arm(RM65,"192.168.1.16")
        real_right_arm = Arm(RM65, "192.168.1.18")
    def get_state(self):
        joint_state_left = real_left_arm.Get_Current_Arm_State()
        joint_state_right = real_right_arm.Get_Current_Arm_State()
        joint_state_left = joint_state_left[1]#[22,1,,1]
        joint_state_right = joint_state_right[1]
        return joint_state_left+joint_state_right
def main(args=None):
    rclpy.init(args=args)

    camera_config = {
        "camera_top": {
            "topic": "/camera_03/color/image_raw",
            "qos": 1
        },
        "camera_left_wrist": {
            "topic": "/camera_02/color/image_raw",
            "qos": 1
        },
        "camera_right_wrist": {
            "topic": "/camera_01/color/image_raw",
            "qos": 1
        }
    }

    camera_top_node = ImageRecorder(
        "image_recorder_node", camera_config, is_debug=False)
    arm_recorder = RealmanArmRecorder("right_arm_recorder")
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(camera_top_node)


    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(arm_recorder)
    DT = 0.1
    max_timestamps = 100
    for t in range(max_timestamps):
        t0 = time.time()
        executor.spin_once(timeout_sec=DT)
        time.sleep(max(0, DT - (time.time() - t0)))
    image_dict = ImageRecorder.get_images()
    qpos = QposRecorder.get_state()
    config = {
        'image_dict': image_dict,
        'qpos': qpos,
        'eval': True,  # 表示启用了 eval 模式（如需要布尔类型，直接写 True/False）
        'task_name': 'sim_transfer_cube_scripted',
        'ckpt_dir': './zdataset',
        'policy_class': 'ACT',
        'kl_weight': 10,
        'chunk_size': 100,
        'hidden_dim': 512,
        'batch_size': 8,
        'dim_feedforward': 3200,
        'num_steps': 2000,
        'lr': 1e-5,
        'seed': 0,
    }
    action = main1(**config)
    return action
    # try:
    #     executor.spin()
    # finally:
    #     camera_top_node.destroy_node()
    #     rclpy.shutdown()

if __name__ == "__main__":
    main()
