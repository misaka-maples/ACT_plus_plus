import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import json


class RealmanArmRecorder(Node):
    def __init__(self, name, config, is_debug=True):
        super().__init__(name)

        self.is_debug = is_debug
        self.config = config

        self.name = name
        self.qpos = [2, 2, 2, 0, 0, 0, 0]
        self.action = [1, 0, 0, 0, 0, 0, 0]
        self.qvel = None

        self.get_logger().info(f"Begin initializing {name} node")
        self.get_logger().info(f"config: {config}")

        self._subscription = self.create_subscription(
            JointState,
            config['topic'],
            self.callback_func,
            config['qos']
        )

        # Action Subscriptions
        self._action_subscription = self.create_subscription(
            JointState,
            config['action_topic'],
            self.action_callback,
            config['qos']
        )

        self.get_logger().info("Init arm recorder done!")

    def callback_func(self, msg):
        # print("-----\n")
        self.get_logger().info(f"Received msg: {msg}")
        # data = json.loads(msg.position)
        data = msg.position
        data.append(0.0)
        self.qpos = data
        self.qvel = data
        if self.is_debug:
            self.get_logger().info(f"Received: {self.config['topic']} \
                qpos: {self.qpos} qvel: {self.qvel}")

    def action_callback(self, msg):
        self.get_logger().info(f"Received action: {msg}")
        data = msg.position
        self.action = data


def main(args=None):
    rclpy.init(args=args)

    arm_recorder = RealmanArmRecorder("right_arm_recorder")

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(arm_recorder)

    try:
        executor.spin()
    finally:
        arm_recorder.destory_node()
        rclpy.shtdown()
