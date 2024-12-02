import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

# 用于调试的交互式环境
import IPython

e = IPython.embed

# 初始化箱子位姿，可以从外部改变
BOX_POSE = [None]  # 格式为 [x, y, z, qw, qx, qy, qz]


# 创建模拟环境
def make_sim_env(task_name):
    """
    创建机器人双臂操作的模拟环境。

    动作空间:
        [left_arm_qpos (6),             # 左臂关节角度
         left_gripper_positions (1.md),    # 左手爪位置（归一化：0 关闭，1.md 打开）
         right_arm_qpos (6),            # 右臂关节角度
         right_gripper_positions (1.md)]   # 右手爪位置（归一化：0 关闭，1.md 打开）

    观测空间:
        {"qpos": 关节位置（关节角度和手爪归一化位置）,
         "qvel": 关节速度（角速度和手爪归一化速度）,
         "images": 图像观测}
    """
    if 'sim_transfer_cube' in task_name:  # 方块传递任务
        xml_path = os.path.join(XML_DIR, 'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)  # 加载物理模型
        task = TransferCubeTask(random=False)  # 定义任务
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT, flat_observation=False)
    elif 'sim_insertion' in task_name:  # 插入任务
        xml_path = os.path.join(XML_DIR, 'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)  # 加载物理模型
        task = InsertionTask(random=False)  # 定义任务
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT, flat_observation=False)
    else:
        raise NotImplementedError  # 如果任务未定义，抛出异常
    return env


# 机器人双臂任务的基类
class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        """
        在执行一步动作前的操作。处理动作并转换为环境可用的格式。
        """
        # 分离左右臂动作和手爪动作
        left_arm_action = action[:6]
        right_arm_action = action[7:7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        # 将手爪动作从归一化值反归一化
        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        # 手爪的两个关节同步动作
        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        # 合并为环境所需的完整动作
        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)

    def initialize_episode(self, physics):
        """
        初始化每个任务的初始状态。
        """
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        """
        获取当前关节位置，归一化手爪的状态。
        """
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        """
        获取关节速度和归一化手爪速度。
        """
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        """
        获取任务环境的状态（如目标物体的位置等）。
        """
        raise NotImplementedError

    def get_observation(self, physics):
        """
        获取当前的观测数据，包括关节状态、环境状态和摄像头图像。
        """
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
        return obs

    def get_reward(self, physics):
        # 返回当前任务的奖励值，具体逻辑由子类实现
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    """
    方块传递任务的实现。
    左右手配合将方块从一个手传递到另一个手。
    """
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4  # 最大奖励为 4，表示成功传递方块

    def initialize_episode(self, physics):
        """
        初始化每个任务的初始状态。
        包括设置机器人的初始姿态和方块的位置。
        """
        # 当前任务不随机化环境配置，需要从外部设置 BOX_POSE
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE  # 重置机器人关节位置
            np.copyto(physics.data.ctrl, START_ARM_POSE)  # 重置控制器
            assert BOX_POSE[0] is not None  # 确保 BOX_POSE 已设置
            physics.named.data.qpos[-7:] = BOX_POSE[0]  # 设置方块位置
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        """
        获取方块的状态（如位置和姿态）。
        """
        env_state = physics.data.qpos.copy()[16:]  # 关节状态之后的部分
        return env_state

    def get_reward(self, physics):
        """
        奖励函数：
        - 奖励值反映机器人抓取、提升和传递方块的状态。
        """
        all_contact_pairs = []
        # 遍历所有接触对，获取接触几何体的名称
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # 判断方块是否被抓取、提升或传递
        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:  # 右手触碰到方块
            reward = 1
        if touch_right_gripper and not touch_table:  # 提升方块
            reward = 2
        if touch_left_gripper:  # 尝试传递方块
            reward = 3
        if touch_left_gripper and not touch_table:  # 成功传递
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    """
    插入任务的实现。
    左手握持插座，右手握持插销，将插销插入插座。
    """
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4  # 最大奖励为 4，表示成功插入

    def initialize_episode(self, physics):
        """
        初始化任务的初始状态。
        包括设置机器人的初始姿态和物体的位置。
        """
        # 当前任务不随机化环境配置，需要从外部设置 BOX_POSE
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE  # 重置机器人关节位置
            np.copyto(physics.data.ctrl, START_ARM_POSE)  # 重置控制器
            assert BOX_POSE[0] is not None  # 确保 BOX_POSE 已设置
            physics.named.data.qpos[-7*2:] = BOX_POSE[0]  # 设置插销和插座的位置
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        """
        获取插销和插座的状态。
        """
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        """
        奖励函数：
        - 奖励值反映机器人抓取、对齐和插入插销的状态。
        """
        all_contact_pairs = []
        # 遍历所有接触对，获取接触几何体的名称
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # 判断插销和插座是否被抓取或插入
        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1.md", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1.md", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1.md") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # 同时抓取插销和插座
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table):  # 提起插销和插座
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # 插销对准插座
            reward = 3
        if pin_touched:  # 成功插入
            reward = 4
        return reward


def get_action(master_bot_left, master_bot_right):
    """
    从机器人状态获取动作，用于控制模拟环境。
    """
    action = np.zeros(14)
    # 左臂动作
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    # 右臂动作
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # 手爪动作
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """
    测试双臂模拟环境的远程操控功能。
    """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]  # 设置方块初始位置

    # 初始化左右臂机器人
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)

    # 创建模拟环境
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]

    # 设置绘图窗口
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        # 获取主机器人的动作
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)  # 执行动作
        episode.append(ts)

        # 更新绘图数据
        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()  # 测试远程操控功能
