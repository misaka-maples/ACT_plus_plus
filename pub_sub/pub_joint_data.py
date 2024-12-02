import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import random
import json


class JointDataPublisher(Node):
    def __init__(self, name, config):
        super().__init__(name)
        self.config = config
        self.publisher = self.create_publisher(
            String, self.config['topic'], self.config['qos'])
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        joint_qpos = [random.random() for _ in range(7)]
        joint_qvel = [random.random() for _ in range(7)]
        joint_data = {
            "joint_qpos": joint_qpos,
            "joint_qvel": joint_qvel
        }
        msg = String()
        msg.data = json.dumps(joint_data)
        self.publisher.publish(msg)

        self.get_logger().info(f"Published: \
            topic: {self.config['topic']}, data: {joint_data}")


def main(args=None):
    rclpy.init(args=args)

    left_config = {'topic': '/left_joint/data', 'qos': 10}
    left_arm_publisher = JointDataPublisher(
        'left_arm_data_publisher', left_config)

    right_config = {'topic': '/right_joint/data', 'qos': 10}
    right_arm_publisher = JointDataPublisher(
        'right_arm_data_publisher', right_config)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(left_arm_publisher)
    executor.add_node(right_arm_publisher)

    try:
        executor.spin()
    finally:
        left_arm_publisher.destroy_node()
        right_arm_publisher.destroy_node()
        rclpy.shutdown()
