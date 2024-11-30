from image_recorder_ros2 import ImageRecorder
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
from tqdm import tqdm
import time

def main():

    rclpy.init()

    camera_config = {
        "top": {
            "topic": "/camera_01/color/image_raw",
            "qos": 1
        },
        "right_wrist": {
            "topic": "/camera_02/color/image_raw",
            "qos": 1
        }
    }
    recorder = ImageRecorder("image_recorder", config=camera_config, is_debug=False)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(recorder)

    try:
        executor.spin()

    finally:
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
