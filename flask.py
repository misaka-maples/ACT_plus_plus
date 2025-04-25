from flask import Flask, request, jsonify
import time
import numpy as np
import json

# 自定义 Flask 应用类
class MyFlaskApp:
    def __init__(self):
        self.app = Flask(__name__)

        # 注册路由
        self.app.add_url_rule('/get_image', 'get_image', self.get_image, methods=['GET'])
        self.app.add_url_rule('/post_data', 'post_data', self.post_data, methods=['POST'])

    # 简单的 GET 请求
    def get_image(self):
        data = request.get_json()  # 获取 JSON 
        json_data = {
                    "top_camera": np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8),
                    "right_camera": np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8),
                    "qpos": np.array([1, 2, 3, 4, 5]).tolist(),
                }
        json_data_np = self.convert_ndarray(json_data)
        x = json.dumps(json_data_np)
        return jsonify(x)

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

# 启动 Flask 应用
if __name__ == '__main__':
    app_instance = MyFlaskApp()
    app_instance.app.run(debug=True, host='0.0.0.0', port=5000)
