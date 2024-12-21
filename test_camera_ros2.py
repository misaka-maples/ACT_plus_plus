import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageRecorder(Node):
    def __init__(self, name, config, is_debug=False):
        super().__init__(name)

        self.get_logger().info("Begin initializing ImageRecorder ... ...")

        self.is_debug = is_debug

        self.bridge = CvBridge()

        self.camera_names = []
        self.camera_counts = 0

        for camera, sub_config in config.items():

            self.camera_counts += 1
            self.camera_names.append(camera)

            setattr(self, f'{camera}_image', None)
            setattr(self, f'{camera}_sec', None)
            setattr(self, f'{camera}_nsec', None)
            setattr(self, f'{camera}_init_done', False)

            if (camera == "top"):
                callback_func = self.image_callback_top
                self._subscriptions_top = self.create_subscription(
                    Image, sub_config['topic'], callback_func, sub_config['qos']
                )
            elif (camera == "left_wrist"):
                callback_func = self.image_callback_left_wrist
                self._subscriptions_left = self.create_subscription(
                    Image, sub_config['topic'], callback_func, sub_config['qos']
                )
            elif (camera == "right_wrist"):
                callback_func = self.image_callback_right_wrist
                self._subscriptions_right = self.create_subscription(
                    Image, sub_config['topic'], callback_func, sub_config['qos']
                )
            else:
                raise NotImplementedError

            self.get_logger().info("Success initializing ImageRecorder\n")

    def init_cameras(self):
        while True:
            init_done = True
            for cam in self.camera_names:
                if (not getattr(self, f'{cam}_init_done')):
                    init_done = False
                    self.get_logger().info(f'camera: {cam} not ready... ...')

            if init_done:
                break

    def image_callback(self, cam_name, data):
        setattr(self, f"{cam_name}_timestamp", data.header.stamp)
        # self.get_logger().info(getattr(self, f"{cam_name}_image"))
        if (data is not None):
            setattr(self, f"{cam_name}_init_done", True)

    def image_callback_top(self, data):
        cam_name = "top"
        setattr(self, f"{cam_name}_data", data)
        setattr(self, f"{cam_name}_image", self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        return self.image_callback(cam_name=cam_name, data=data)

    def image_callback_right_wrist(self, data):
        cam_name = "right_wrist"
        setattr(self, f"{cam_name}_data", data)
        setattr(self, f"{cam_name}_image", self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        return self.image_callback(cam_name=cam_name, data=data)

    def image_callback_left_wrist(self, data):
        cam_name = "left_wrist"
        setattr(self, f"{cam_name}_data", data)
        setattr(self, f"{cam_name}_image", self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))
        return self.image_callback(cam_name=cam_name, data=data)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict


def main(args=None):
    rclpy.init(args=args)

    camera_config = {
        "top": {
            "topic": "/camera_01/color/image_raw",
            "qos": 2
        },
        "right_wrist": {
            "topic": "/camera_02/color/image_raw",
            "qos": 2
        }
    }

    image_recorder_node = ImageRecorder(
        name="image_recorder_node",
        config=camera_config,
        is_debug=False
    )

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(image_recorder_node)

    try:
        executor.spin()
    finally:
        image_recorder_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
