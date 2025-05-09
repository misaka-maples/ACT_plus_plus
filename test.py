import numpy as np
from tcp_tx import PersistentClient
import time 
from deploy.eval_function import eval,CAMERA_HOT_PLUG,GPCONTROL
camera = CAMERA_HOT_PLUG()
time.sleep(5)
robot = PersistentClient('192.168.3.15', 8001)
gpcontrol = GPCONTROL()
gpcontrol.start()


data = np.load('/workspace/exchange/grasp_pos6d_series/0506_target_2_pose_series.npy')
np.set_printoptions(precision=4, suppress=True)
# client = PersistentClient('192.168.3.15', 8001)
gpcontrol.state_data_2 = 200
time.sleep(1)
for index,i in enumerate(data):
    print(i)
    if index == 3:
        print("关闭夹爪")
        gpcontrol.state_data_2 = 0
        time.sleep(2)
    robot.set_arm_position(i.tolist(),'pose',2)
eval(camera=camera,persistentClient=robot,gp_contrpl=gpcontrol,real_robot=True,data_true=False,ckpt_dir=r'/workspace/exchange/4-30/hdf5_file_exchange_4-30',ckpt_name='policy_step_80000_seed_0.ckpt',hdf5_path=r'/workspace/exchange/4-30/hdf5_file_exchange_4-30/episode_20.hdf5',state_dim=16,temporal_agg=True)
# time.sleep(2)
print("第二段")
eval(camera=camera,persistentClient=robot,gp_contrpl=gpcontrol,real_robot=True,data_true=False,ckpt_dir=r'/workspace/exchange/4-30/hdf5_file_duikong_4-30/act',ckpt_name='policy_best.ckpt',hdf5_path=r'/workspace/exchange/4-30/hdf5_file_duikong_4-30/episode_21.hdf5',state_dim=8,temporal_agg=True)

# print(data)