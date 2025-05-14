import numpy as np
from tcp_tx import PersistentClient
import time 
import sys
sys.path.append("./")

from deploy.eval_function import eval,CAMERA_HOT_PLUG,GPCONTROL
from cjb.main_es import main

camera = CAMERA_HOT_PLUG()
time.sleep(5)
robot = PersistentClient('192.168.3.15', 8001)
gpcontrol = GPCONTROL()
gpcontrol.start()
def gp():
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
def first_traj():
    
    # eval(camera=camera,
    #         persistentClient=robot,
    #         gp_contrpl=gpcontrol,
    #         real_robot=True,
    #         data_true=False,
    #         ckpt_dir=r'/workspace/exchange/5-9/exchange/act',
    #         ckpt_name='policy_step_40000_seed_0.ckpt',
    #         hdf5_path=r'/workspace/exchange/5-9/exchange/episode_22.hdf5',
    #         state_dim=16,
    #         temporal_agg=True)
    # # time.sleep(2)
    main()
    # print("第二段")
    # eval(camera=camera,
    #         persistentClient=robot,
    #         gp_contrpl=gpcontrol,
    #         real_robot=True,
    #         data_true=False,
    #         ckpt_dir=r'/workspace/exchange/5-9/duikong/act',
    #         ckpt_name='policy_best.ckpt',
    #         hdf5_path=r'/workspace/exchange/5-9/duikong/episode_23.hdf5',
    #         state_dim=8,
    #         temporal_agg=True)
if __name__ == "__main__":
    # gp()
    first_traj()
# print(data)