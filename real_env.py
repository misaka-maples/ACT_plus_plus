import time
import numpy as np
import collections
from joint_recorder import JointRecorder
from image_recorder import ImageRecorder
from test_rem import ArmStatePublisher
import dm_env
import random
import rclpy.
import threading
from rclpy.node import Node
import os
from tqdm import tqdm
import h5py
import argparse
import cv2


class RealEnv:

    def __init__(self):

        rclpy.init(args=None)
        # init arm joint recorder
        left_arm_config = {
            'topic': '/left_joint/data',
            'qos': 10
        }
        self.left_arm_recorder = JointRecorder(
            'left_arm_recorder', left_arm_config, is_debug=True)

        right_arm_config = {
            'topic': '/right_joint/data',
            'qos': 10
        }
        self.right_arm_recorder = JointRecorder(
            'right_arm_recorder', right_arm_config, is_debug=True)

        # init image recorder
        camera_config = {
            "camera_top": {
                "topic": "/camera_03/color/image_raw",
                "qos": 10
            },
            # "camera_left_wrist": {
            #     "topic": "/camera_01/color/image_raw",
            #     "qos": 10
            # },
            # "camera_right_wrist": {
            #     "topic": "/camera_02/color/image_raw",
            #     "qos": 10
            # }
        }

        self.image_recorder = ImageRecorder(
            "image_recorder_node", camera_config, is_debug=True)

        self.realman_recorder = ArmStatePublisher()



        # self.thread_left_arm_recorder = threading.Thread(target=self.start_recorder, args=(self.left_arm_recorder,))
        # self.thread_right_arm_recorder = threading.Thread(target=self.start_recorder, args=(self.right_arm_recorder,))
        # self.thread_image_recorder = threading.Thread(target=self.start_recorder, args=(self.image_recorder,))

        # self.thread_left_arm_recorder.start()
        # self.thread_right_arm_recorder.start()
        # self.thread_image_recorder.start()

        # executor = rclpy.executors.MultiThreadedExecutor()
        # executor.add_node(self.left_arm_recorder)
        # executor.add_node(self.right_arm_recorder)
        # executor.add_node(self.image_recorder)

        # try:
        #     executor.spin()
        # finally:
        #     self.left_arm_recorder.destroy_node()
        #     self.right_arm_recorder.destroy_node()
        #     self.image_recorder.destroy_node()
        #     rclpy.shutdown()

    def start_recorder(node):
        rclpy.spin()

    def get_qpos(self):
        # left_qpos_raw = [random.random() for _ in range(7)]
        # right_qpos_raw = [random.random() for _ in range(7)]
        # left_qpos_raw = self.left_arm_recorder.qpos
        # right_qpos_raw = self.right_arm_recorder.qpos
        left_qpos_raw = self.realman_recorder.filtered_joint_state
        right_qpos_raw = self.realman_recorder.filtered_joint_state1

        left_arm_qpos = np.array(left_qpos_raw[:6], dtype=np.float64)
        right_arm_qpos = np.array(right_qpos_raw[:6], dtype=np.float64)
        left_gripper_qpos = np.array([left_qpos_raw[-1]], dtype=np.float64)
        right_gripper_qpos = np.array([right_qpos_raw[-1]], dtype=np.float64)

        # print(left_qpos_raw)
        # print(len(left_arm_qpos))
        # print(len(left_gripper_qpos))

        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    def get_qvel(self):
        left_qvel_raw = [random.random() for _ in range(7)]
        right_qvel_raw = [random.random() for _ in range(7)]
        # left_qvel_raw = self.left_arm_recorder.qvel
        # right_qvel_raw = self.right_arm_recorder.qvel
        left_arm_qvel = np.array(left_qvel_raw[:6], dtype=np.float64)
        right_arm_qvel = np.array(right_qvel_raw[:6], dtype=np.float64)
        left_gripper_qvel = np.array([left_qvel_raw[-1]], dtype=np.float64)
        right_gripper_qvel = np.array([right_qvel_raw[-1]], dtype=np.float64)

        # print(self.left_arm_recorder.qvel)
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    def get_effort(self):
        left_robot_effort = np.array([random.random()
                                     for _ in range(7)], dtype=np.float64)
        right_robot_effort = np.array(
            [random.random() for _ in range(7)], dtype=np.float64)
        return np.concatenate([left_robot_effort, right_robot_effort])

    def get_images(self):
        print(self.image_recorder.get_images())
        return self.image_recorder.get_images()

    def get_base_vel(self):
        base_vel = [random.random() for _ in range(2)]
        return np.array(base_vel, dtype=np.float64)\

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        obs['base_vel'] = self.get_base_vel()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            # set real arms and cameras
            # -------
            pass
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )

    def step(self, action, base_action=None, get_obs=True):
        state_len = int(len(action)/2)
        left_action = action[:state_len]
        right_action = action[state_len:]
        # set arm
        # -----------------------
        if get_obs:
            obs = self.get_observation()
        else:
            obs = None

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )


def make_real_env():
    env = RealEnv()
    return env


def get_action():
    action = [random.random() for _ in range(14)]

    return action


def test():

    env = make_real_env()
    ts = env.reset(fake=True)
    episode = [ts]

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(env.left_arm_recorder)
    executor.add_node(env.right_arm_recorder)
    executor.add_node(env.image_recorder)
    executor.add_node(env.realman_recorder)

    time.sleep(2)

    try:
        for t in range(1000):
            print(f'times = {t}')
            executor.spin_once(timeout_sec=0.1)

            action = get_action()
            ts = env.step(action=action)
            episode.append(ts)
            time.sleep(0.02)
    finally:
        executor.shutdown()
        env.left_arm_recorder.destroy_node()
        env.right_arm_recorder.destroy_node()
        env.image_recorder.destroy_node()
        rclpy.shutdown()


def capture_one_episode(camera_names, max_timesteps, dataset_dir, dataset_name, overwrite, save_dir):
    env = make_real_env()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(env.left_arm_recorder)
    executor.add_node(env.right_arm_recorder)
    executor.add_node(env.image_recorder)

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        # print(f"Dataset already exist at {
        #       dataset_path}\nHint: set overwrite to True.")
        exit()

    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    time0 = time.time()

    DT = 0.1

    for t in tqdm(range(max_timesteps)):
        t0 = time.time()
        action = get_action()
        executor.spin_once(timeout_sec=0.1)
        t1 = time.time()
        ts = env.step(action)
        t2 = time.time()
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])
        time.sleep(max(0, DT - (time.time() - t0)))  # 采集数据

    executor.shutdown()
    env.left_arm_recorder.destroy_node()
    env.right_arm_recorder.destroy_node()
    env.image_recorder.destroy_node()
    rclpy.shutdown()

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        # '/base_action': [],
    }

    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    drop_num = 0
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        for cam_name in camera_names:
            if ((ts.observation['images'][cam_name])) is not None and (ts.observation['images'][cam_name].size > 0):
                data_dict[f'/observations/images/{cam_name}'].append(
                    ts.observation['images'][cam_name])
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/observations/effort'].append(
                    ts.observation['effort'])
                data_dict['/action'].append(action)
                # data_dict['/base_action'].append(ts.observation['base_vel'])
            else:
                drop_num += 1

    max_timesteps = max_timesteps - int(drop_num/len(camera_names))

    # np.save(f'{save_dir}/test_qpos.npy', data_dict['/observations/qpos'])

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    np.savetxt(f'{save_dir}/qpos.txt', data_dict['/observations/qpos'])
    np.savetxt(f'{save_dir}/qvel.txt', data_dict['/observations/qvel'])
    np.savetxt(f'{save_dir}/effort.txt', data_dict['/observations/effort'])
    np.savetxt(f'{save_dir}/action.txt', data_dict['/action'])
    for cam_name in camera_names:
        for i in range(len(data_dict[f'/observations/images/{cam_name}'])):
            cv2.imwrite(f'{save_dir}/{cam_name}_{i}.jpg',
                        data_dict[f'/observations/images/{cam_name}'][i])
        # print(len(data_dict[f'/observations/images/{cam_name}']))
        # np.savetxt(f'{save_dir}/{cam_name}_image.txt', data_dict[f'/observations/images/{cam_name}'])

    COMPRESS = True

    if COMPRESS:
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        compressed_len = []
        print(data_dict['/observations/images/camera_top'])
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode(
                    '.jpg', image, encode_param)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))

            data_dict[f'/observations/images/{cam_name}'] = compressed_list

        t0 = time.time()
        # compressed_len = np.concatenate((compressed_len[0],compressed_len[1.md],compressed_len[2],))
        compressed_len = np.array(compressed_len)
        print(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False  # 将sim属性设置为False
        root.attrs['compress'] = COMPRESS  # 添加压缩属性

        # 创建 observations 组
        obs = root.create_group('observations')
        image = obs.create_group('images')

        # 根据COMPRESS的值来创建不同大小的图像数据集
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(
                    cam_name, (max_timesteps, padded_size), dtype='uint8', chunks=(1, padded_size)
                )
            else:
                _ = image.create_dataset(
                    cam_name, (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3)
                )

        # 截断数据字典到max_timesteps
        data_dict['/observations/qpos'] = data_dict['/observations/qpos'][:max_timesteps]
        data_dict['/observations/qvel'] = data_dict['/observations/qvel'][:max_timesteps]
        data_dict['/observations/effort'] = data_dict['/observations/effort'][:max_timesteps]
        data_dict['/action'] = data_dict['/action'][:max_timesteps]

        # 创建数据集
        _ = obs.create_dataset('qpos', (max_timesteps, 14))
        _ = obs.create_dataset('qvel', (max_timesteps, 14))
        _ = obs.create_dataset('effort', (max_timesteps, 14))
        _ = root.create_dataset('action', (max_timesteps, 14))

        # 将数据字典中的数据写入到文件
        for name, array in data_dict.items():
            root[name][...] = array

        # 如果启用压缩，添加压缩长度数据集
        if COMPRESS:
            compressed_len = np.random.randint(1, 1000, size=(len(camera_names), max_timesteps))  # 假设压缩数据长度
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            root['/compress_len'][...] = compressed_len
    print(f'Saving: {time.time() - t0:.1f} secs')

    return True


if __name__ == "__main__":
    # test()
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--max_timesteps', type=int)
    # max_timesteps
    args = parser.parse_args()

    # camera_names = ["camera_top", "camera_left_wrist", "camera_right_wrist"]
    # camera_names = ["camera_top", "camera_left_wrist", "camera_right_wrist"]
    camera_names = ["camera_top"]
    dataset_dir = "/home/zhnh/Documents/xzx_projects/aloha_dataset/episode_save_dir"
    dataset_name = "episode_03"
    # max_timestep = 50

    save_dir = "/home/zhnh/Documents/xzx_projects/aloha_dataset/origin_data_save_dir"
    # /home/xu/aloha_dataset/save_data03

    # for i in range(10,50):
    #     dataset_name = f"episode_{i}"
    # capture_one_episode(camera_names=camera_names,
    #                     max_timesteps=50,
    #                     dataset_dir=dataset_dir,
    #                     dataset_name=dataset_name,
    #                     overwrite=True,
    #                     save_dir=save_dir)
    capture_one_episode(camera_names=camera_names,
                        max_timesteps=args.max_timesteps,
                        dataset_dir=args.dataset_dir,
                        dataset_name=args.dataset_name,
                        overwrite=True,
                        save_dir=args.save_dir)
