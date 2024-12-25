import os
from ModelMqtt import ModelMqttConfig , ModelMqtt
import numpy as np
from policy_action_generation import ActionGenerator
import cv2
def save_video(image_list, fps=30):
    # image_list = [frame for frame in image_list]
    #
    array = np.array(image_list)
    print(array.shape)
    try:
        frame_height, frame_width, _ = image_list[0].shape

        # 定义视频写入器
        file_path = os.path.join(os.getcwd())
        path = os.path.join(file_path, f"frame.mp4")
        video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # 将每一帧写入视频
        for frame in image_list:
            # image_list = frame[:, :, [2, 1, 0]]  # 交换图像的B和R通道
            frame = frame[:, :, [2, 1, 0]]
            video_writer.write(frame)

        # 释放视频写入器
        video_writer.release()
        print(f"\nVideo saved successfully at {file_path}")

    except Exception as e:

        print(f"Error saving video:\n {e}")

def main():
    try :
        # cv2.namedWindow("right", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("top", cv2.WINDOW_NORMAL)
        global max_timesteps
        image_list = []
        config = ModelMqttConfig()
        config.host = '192.168.1.101'
        model_mqtt = ModelMqtt(config)
        model_mqtt.start()
        model_mqtt._start_pub_thread()
        config_ = {
            'eval': True,  # 表示启用了 eval 模式（如需要布尔类型，直接写 True/False）
            'task_name': 'train',
            'ckpt_dir': r'/home/zhnh/Documents/project/act_arm_project/models/1224_resnet18',
            'policy_class': 'ACT',
            'chunk_size': 210,
            'backbone': 'resnet18',
            'temporal_agg': True,
            'max_timesteps': max_timesteps,
        }
        ActionGeneration = ActionGenerator(config_)
        for i in range(max_timesteps):
            ActionGeneration.t = i
            data_dict = model_mqtt.get_observations()
            # print(data_dict)
            if data_dict is not  None or 0:
                image_list.append(data_dict['right_camera'])
                image_dict = {
                    'top': data_dict['top_camera'],
                    'right_wrist': data_dict['right_camera'],
                }
                radius_qpos = data_dict['qpos']
                ActionGeneration.image_dict = image_dict
                ActionGeneration.qpos_list = radius_qpos
                actions = ActionGeneration.get_action()
                model_mqtt.enqueue_actions(actions)
    except Exception as e:
        print(e)
    finally:
        save_video(image_list, 30)


if __name__ == "__main__":
    max_timesteps = 4000
    main()