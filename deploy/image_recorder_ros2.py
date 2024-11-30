import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from tqdm import tqdm
import time

class ImageRecorder(Node):
    def __init__(self, name, config, is_debug=False):
        super().__init__(name)

        self.get_logger().info("Begin initializing ImageRecorde Node... ...")
        print(config)

        self.is_debug = is_debug
        # convert sensor_msgs.msg.Image to cv2
        self.bridge = CvBridge()
        # collect camera_names
        self.camera_names = []
        # count cameras num
        self.camera_counts = 0

        self.init_done = False

        for camera, sub_config in config.items():
            print(f"{camera} : {sub_config}")

            self.camera_counts += 1
            self.camera_names.append(camera)

            setattr(self, f'{camera}_image', None)
            setattr(self, f'{camera}_sec', None)
            setattr(self, f'{camera}_nsec', None)
            setattr(self, f'{camera}_init_done', False)

            if (camera == "top"):
                # callback_func = self.image_callback_top
                # Sub
                self._subscription_top = self.create_subscription(
                    Image, sub_config['topic'], self.image_callback_top, sub_config['qos'])
            elif (camera == "left_wrist"):
                # callback_func = self.image_callback_left_wrist
                self._subscription_left = self.create_subscription(
                    Image, sub_config['topic'], self.image_callback_left_wrist, sub_config['qos'])
            elif (camera == "right_wrist"):
                self._subscription_right = self.create_subscription(
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
        print("---------")
        # image = getattr(self, f'{cam_name}_image')
        # print(image.size)

        if (data is not None):
            setattr(self, f'{cam_name}_init_done', True)

        if self.is_debug:
            timestamp = getattr(self, f'{cam_name}_timestamp')
            self.get_logger().debug(f"Get {cam_name} data:\ntimestamp: {timestamp}\nimage_size: {data}")

    def image_callback_top(self, data):
        cam_name = 'top'
        setattr(self, f'{cam_name}_data', data)
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        return self.image_callback(cam_name=cam_name, data=data)

    def image_callback_left_wrist(self, data):
        cam_name = 'left_wrist'
        setattr(self, f'{cam_name}_data', data)
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        return self.image_callback(cam_name=cam_name, data=data)

    def image_callback_right_wrist(self, data):
        cam_name = 'right_wrist'
        setattr(self, f'{cam_name}_data', data)
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        return self.image_callback(cam_name=cam_name, data=data)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

