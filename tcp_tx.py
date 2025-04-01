import socket
import threading
import re
import time


class PersistentClient:
    HEADER = b'&'
    FOOTER = b'^'
    ENCODING = 'utf-8'

    RECV_TIMEOUT = 10         # 🟢 读取超6时时间

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None  # 连接对象
        self.connected = False  # 连接状态
        self.recive_data = None  # 接收数据
        # 建立初始连接
        self.connect()

        #初始参数
        self.vel = 100  #速度
        self.acc = 100  #加速度
        self.dcc = 100  #减速度
        # self._receive_thread = threading.Thread(target=self._receive_data, daemon=True)
        # self._receive_thread.start()  # 启动接收线程  
          
    def _frame_data(self, data):
        """封装数据包（增加协议头和尾部）"""
        if not isinstance(data, bytes):
            data = data.encode(self.ENCODING)
        return self.HEADER + data + self.FOOTER

    def connect(self):
        """建立长连接"""
        if self.sock:
            self.sock.close()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)  # 连接超时

        try:
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(self.RECV_TIMEOUT)  # 🟢 设置 `recv()` 超时
            self.connected = True
            print("[INFO] 成功建立长连接")
        except (ConnectionRefusedError, TimeoutError) as e:
            print(f"[ERROR] 连接失败: {e}")
            self.connected = False

    def send_message(self, message):
        """发送数据（仅发送，写入线程）"""
        if not self.connected:
            print("[WARNING] 连接已断开，正在尝试重新连接...")
            self.connect()

        try:
            framed_data = self._frame_data(message)
            # print(framed_data)
            self.sock.sendall(framed_data)

            return True
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[ERROR] 连接断开: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"[ERROR] 未知错误: {e}")
            return False

    def _receive_data(self,robotnum):
        """实时接收数据（独立线程，不影响写入）"""
        data = None
        while True:
            # start_time = time.time()  # 记录开始时间
            # print("开始接收数据")
            if self.connected:
                try:
                    data = self.sock.recv(1024)
                    data = data.decode(self.ENCODING)
                    if not data:
                        print("[WARNING] 接收数据为空，继续监听...")
                        continue
                    # print(f"接收到的数据：{data}")
                    if "readyToNext" not in data :
                        match = re.search(r'robotId:(\d+)', data)
                        id_num = int(match.group(1)) if match else None
                        if id_num != robotnum:
                            print(f"接收数据的ID号不匹配：{id_num} != {robotnum}")
                            continue
                    else:
                        # print(f"接收到readyToNext数据")
                        match = re.search(r"readyToNext,(\d+)", data)  # 匹配逗号后面的数字
                        id_num_ready = int(match.group(1)) if match else None
                        if id_num_ready != robotnum:
                            print(f"接收数据的ID号不匹配：{id_num_ready} != {robotnum}")
                            continue
                except socket.timeout:
                    print("[WARNING] 接收超时，继续监听...")
                    break  # 超时后继续监听
                except (ConnectionResetError, BrokenPipeError):
                    print("[WARNING] 服务器断开连接，正在重连...")
                    self.connected = False
                    self.connect()
                self.recive_data = data
                # time.sleep(0.01)  # 🟢 等待0.01秒，避免过快循环
                return self.recive_data
            
            
    def close(self):
        """关闭连接"""
        if self.sock:
            self.sock.close()
            self.connected = False
            print("[INFO] 连接已关闭")

    def set_open(self,robotnum):
        message = f"open,{robotnum}"
        self.send_message(message)


    def set_stop(self,robotnum):
        message = f"stop,{robotnum}"
        self.send_message(message)

    def set_clear(self,robotnum):
        message = f"clear,{robotnum}"
        self.send_message(message)
    def set_close(self,robotnum):
        message = f"close,{robotnum}"
        self.send_message(message)
    def set_reset(self,robotnum):
        message = f"reset,{robotnum}"
        self.send_message(message)
    def set_arm_position(self, value:list, model:str,robotnum:str, timeout=5):
        """
        设置机械臂位置（关节模式或位姿模式），阻塞直到收到完成信号

        :param value: 目标位置列表（关节角或位姿坐标）
        :param model: 模式选择，可选 "joint"（关节） 或 "pose"（位姿）
        :param timeout: 超时时间（秒）
        :return: 成功返回True，超时或失败返回False
        """
        # 参数校验
        if not isinstance(value, list) or len(value) != 6:
            print("[ERROR] 输入必须是包含6个数值的列表")
            return False

        if model not in ["joint", "pose"]:
            print("[ERROR] 模式参数必须是 'joint' 或 'pose'")
            return False

        try:
            # 构造命令字符串
            value_str = ",".join(f"{x:.4f}" for x in value)
            cmd_type = "ACS" if model == "joint" else "PCS"
            command = f"set,{robotnum},{self.vel},{cmd_type},0,0,{value_str},0,{self.acc},{self.dcc}"

            # 发送命令
            if not self.send_message(command):
                return False
            # finsh = 0
            # print(f"发送数据：{command}")
            # 阻塞等待响应
            while True:
                # 读取接收数据（假设有非阻塞读取方法）
                data = self._receive_data(robotnum)

                time.sleep(0.01)
                print(f'wating data')
                if "readyToNext" in data:  # 根据实际的返回数据格式修改
                    
                    print(f"data:{data}")
                    break


        except Exception as e:
            print(f"[ERROR] 设置位置失败: {e}")
            return False

    def get_arm_position_joint(self,robotnum):
            """
            获取机械臂位姿：
            发送请求后等待缓冲区中出现响应数据，最多等待5秒。
            :return: 返回接收到的字符串响应数据，或None（超时）
            """
            while True:

                message = f"get,{robotnum},ACS"
                # print("发送数据：", message)
                self.send_message(message)
                # print("发送数据成功")
                # response = self.recive_data
                response = self._receive_data(robotnum)
                # print("接收到的数据：", response)
                if response == None:
                    print("[ERROR] get超时，继续等待数据...")
                    continue
                if  "readyToNext" in response:
                    continue
                else:
                    match = re.search(r'getPos:"([^"]+)"', response)
                    # print("匹配结果：", match)
                    if match:
                        # 提取并解析六个浮动数据
                        data_string = match.group(1)  # 获取 "2,0,0,0,-7.2092,133.368,500.813,-1.63063,-0.0261585,-1.57236,0"
                        data_list = data_string.split(',')[4:10]  # 获取从第5到第10个数据（索引从0开始）
                        # 将数据转换为浮动数并返回
                        return [float(i) for i in data_list]
                    else:
                        print("[ERROR] 无法解析位置数据")
                        break
    def get_arm_position_pose(self, robotnum):
        """
        获取机械臂位姿：
        发送请求后等待缓冲区中出现响应数据，最多等待5秒。
        :return: 返回接收到的字符串响应数据，或None（超时）
        """
        while True:
            message = f"get,{robotnum},PCS"
            message = message.strip()
            self.send_message(message)
            # print("发送数据成功")
            response = self._receive_data(robotnum)
            # response = self.recive_data
            # print("接收到的数据：", response)
            match = re.search(r'getPos:"([^"]+)"', response)
            if match:
                # 提取并解析六个浮动数据
                data_string = match.group(1)  # 获取 "2,0,0,0,-7.2092,133.368,500.813,-1.63063,-0.0261585,-1.57236,0"
                data_list = data_string.split(',')[4:10]  # 获取从第5到第10个数据（索引从0开始）
                # 将数据转换为浮动数并返回
                # print(f"匹配到的数据：{data_list}")
                return [float(i) for i in data_list]
            else:
                print("[ERROR] 无法解析位置数据")
                # return None
        # 返回接收到的响应字符串

if __name__ == "__main__":
    client = PersistentClient('192.168.2.14', 8001)
    client.set_close(1)
    client.set_clear(1)
    client.set_open(1)  
    while True:
        try:
            message = input("> ")  # 🟢 从命令行获取输入1
    
            if message.lower() == "exit":
                print("[INFO] 退出客户端...")
                client.close()
                break
    
            if message == "1,1":
                # while True:
                    joint_pose = client.get_arm_position_joint(1)
                    pose = client.get_arm_position_pose(1)
                    print(joint_pose,pose)
                    time.sleep(0.1)
            if message == "1,2":
                deta = client.get_arm_position_pose(2)#[191.699, -68.037, 585.006, 1.08749, 1.40397, 1.99036]
                print(deta)
            
            if message == "2":
                # client.set_arm_position( [-81.1963, -580.862, 115.466, 2.73102, -0.00482064, 2.98929],"pose",1)
                client.set_arm_position([-9.19798, -84.536, 631.215, 1.42741, -0.0901151, 2.83646],"pose",1)
            if message == "3":
                i=1
                while i <=10:
                    client.set_arm_position([1.7504, -451.32, 831.73, 0.113867, 0.173187, -0.04549], "pose", 2) 
                    print(client.get_arm_position_pose(2))
                    client.set_arm_position([28.7504, -451.32, 831.73, 0.113867, 0.173187, -0.04549], "pose", 2)
                    print(client.get_arm_position_pose(2))
                    time.sleep(0.1)
                    i += 1
    
    
            if message == "4":
                deta = client.set_arm_position([50.5687,-42.0996,43.3454,2.36686,-42.1309,50.4586], "joint", 2)
            if message == "5":
                deta = client.send_message("stop,1")
            if message == "6":
                deta = client.send_message("open,1")
    
    
    
        except KeyboardInterrupt:
            print("\n[INFO] 终止客户端...")
            # client.set_stop(1)
            client.close()
            break
