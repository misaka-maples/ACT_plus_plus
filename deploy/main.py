from flask import Flask, request, jsonify
import time
import numpy as np
import json
from gpcontrol import GPCONTROL
from camera_hot_plug import CAMERA_HOT_PLUG
from tcp_tx import PersistentClient
from eval_function import eval
# 自定义 Flask 应用类
class MyFlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        # 注册路由
        self.app.add_url_rule('/get_image', 'get_image', self.get_image, methods=['GET'])
        self.app.add_url_rule('/get_robot_state', 'get_robot_state', self.get_robot_state, methods=['GET'])
        self.app.add_url_rule('/post_data', 'post_data', self.post_data, methods=['POST'])
        self.app.add_url_rule('/set_gp_state', 'set_gp_state', self.set_gp_state, methods=['POST'])
        self.app.add_url_rule('/eval_function', 'eval_function', self.eval_function, methods=['POST'])

        self.gpcontrol = GPCONTROL()
        self.gpcontrol.start()
        time.sleep(1)
        self.camera = CAMERA_HOT_PLUG()
        self.robot_arm = PersistentClient(host='192.168.3.15',port=8001)


    # 简单的 GET 请求
    def get_image(self):
        data = request.get_json()  # 获取 JSON 
        data_pased = json.loads(data)
        if data_pased['commond'] == 'get_image':
            color_image_dict,depth_image_dict,color_width, color_height= self.camera.get_images()
            if data_pased['camera_pose']=='right_camera':
                color = color_image_dict['CP1L44P0004Y']
                depth = depth_image_dict['CP1L44P0004Y']
                data_dict = {
                    'color':color,
                    'depth':depth
                }
            json_data_np = self.convert_ndarray(data_dict)
            x = json.dumps(json_data_np)
            return jsonify(x)
        else:
            print("dadsdad")
            return jsonify("error")
    def get_robot_state(self):
        data = request.get_json()
        data_= json.loads(data)
        robot_id = data_['robot_id']
        mode = data_['mode']
        if mode == 'joint':
            pose = self.robot_arm.get_arm_position_joint(robotnum=robot_id)
        elif mode == 'pose':
            pose = self.robot_arm.get_arm_position_pose(robotnum=robot_id)
        pose_ = json.dumps(pose)
        return jsonify(pose_)
    # 简单的 POST 请求
    def post_data(self):
        data = request.get_json()  # 获取 JSON 数据
        print("Received data:", data)
        response = {"message": "Data received successfully"}
        return jsonify(response)
    def convert_ndarray(self,obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_ndarray(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarray(item) for item in obj]
        return obj
    def get_pose(self):
        pass
    def set_pose(self):
        pass
    def set_gp_state(self):
        data = request.get_json()  # 获取 JSON 数据
        data_pased = json.loads(data)
        print(data_pased)
        state = self.gpcontrol.set_state(data_pased['state'])
        return jsonify(state)
    def set_robot_state(self):
        
        data = request.get_json()  # 获取 JSON 数据
        data_pased = json.loads(data)
        robot_id = data_pased['robot_id']
        value = data_pased['pose']
        model = model['model']
        print(data_pased)
        state = self.robot_arm.set_arm_position(value=value,model='joint',robotnum=robot_id)
        return jsonify(state)
    def eval_function(self):
        data = request.get_json()  # 获取 JSON 数据
        data_pased = json.loads(data)
        # robot_id = data_pased['robot_id']
        # value = data_pased['pose']
        # model = model['model']
        print(data_pased)
        eval(camera=self.camera,persistentClient=self.robot_arm,gp_contrpl=self.gpcontrol,real_robot=True,data_true=False,ckpt_dir=r'/workspace/exchange/4-24/act',ckpt_name='policy_best.ckpt')
        # state = self.robot_arm.set_arm_position(value=value,model='joint',robotnum=robot_id)
        return jsonify("succes")
        



# 启动 Flask 应用
if __name__ == '__main__':
    app_instance = MyFlaskApp()
    app_instance.app.run(debug=False, host='0.0.0.0', port=5000)
