import socket

import re
import time


class PersistentClient:
    HEADER = b'&'
    FOOTER = b'^'
    ENCODING = 'utf-8'

    RECV_TIMEOUT = 2         # 🟢 读取超6时时间

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None  # 连接对象
        self.connected = False  # 连接状态

        # 建立初始连接
        self.connect()

        #初始参数
        self.vel = 100  #速度
        self.acc = 100  #加速度
        self.dcc = 100  #减速度


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
            print(framed_data)
            self.sock.sendall(framed_data)

            return True
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"[ERROR] 连接断开: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"[ERROR] 未知错误: {e}")
            return False

    # def _receive_data(self):
    #     """实时接收数据（独立线程，不影响1写入）"""
    #     while True:
    #         if self.connected:
    #             try:
    #                 data = self.sock.recv(1024)
    #                 if data:
    #                     print(f"[INFO] 实时接收到数据: {data.decode(self.ENCODING)}")
    #             except socket.timeout:
    #                 continue  # 🟢 超时不报错，继续监听
    #             except (ConnectionResetError, BrokenPipeError):
    #                 print("[WARNING] 服务器断开连接，正在重连...")
    #                 self.connected = False
    #                 self.connect()
    def _receive_data(self):
        """实时接收数据（独立线程，不影响写入）"""
        data = None
        while True:
            if self.connected:
                try:
                    data = self.sock.recv(1024)
                    data = data.decode(self.ENCODING)

                except socket.timeout:
                    continue  # 超时后继续监听
                except (ConnectionResetError, BrokenPipeError):
                    print("[WARNING] 服务器断开连接，正在重连...")
                    self.connected = False
                    self.connect()
                return data


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
            finsh = 0
            # 阻塞等待响应
            while finsh <=0:
                # 读取接收数据（假设有非阻塞读取方法）
                data = self._receive_data()

                if "readyToNext:" in data:  # 根据实际的返回数据格式修改
                    finsh = 1
                    # return True


        except Exception as e:
            print(f"[ERROR] 设置位置失败: {e}")
            return False

    def get_arm_postion_joint(self,robotnum):
            """
            获取机械臂位姿：
            发送请求后等待缓冲区中出现响应数据，最多等待5秒。
            :return: 返回接收到的字符串响应数据，或None（超时）
            """

            message = f"get,{robotnum},ACS"
            self.send_message(message)

            response = self._receive_data()


            match = re.search(r'getPos:"([^"]+)"', response)
            if match:
                # 提取并解析六个浮动数据
                data_string = match.group(1)  # 获取 "2,0,0,0,-7.2092,133.368,500.813,-1.63063,-0.0261585,-1.57236,0"
                data_list = data_string.split(',')[4:10]  # 获取从第5到第10个数据（索引从0开始）
                # 将数据转换为浮动数并返回
                return [float(i) for i in data_list]
            else:
                print("[ERROR] 无法解析位置数据")
                # return None
            # 返回接收到的响应字符串

    def get_arm_position_pose(self, robotnum):
        """
        获取机械臂位姿：
        发送请求后等待缓冲区中出现响应数据，最多等待5秒。
        :return: 返回接收到的字符串响应数据，或None（超时）
        """
        message = f"get,{robotnum},PCS"
        message = message.strip()
        self.send_message(message)
        response = self._receive_data()
        # print("接收到的数据：", response)


        match = re.search(r'getPos:"([^"]+)"', response)
        if match:
            # 提取并解析六个浮动数据
            data_string = match.group(1)  # 获取 "2,0,0,0,-7.2092,133.368,500.813,-1.63063,-0.0261585,-1.57236,0"
            data_list = data_string.split(',')[4:10]  # 获取从第5到第10个数据（索引从0开始）
            # 将数据转换为浮动数并返回
            return [float(i) for i in data_list]
        else:
            print("[ERROR] 无法解析位置数据")
            # return None
        # 返回接收到的响应字符串


if __name__ == "__main__":
    client = PersistentClient('192.168.3.15', 8001)
    client.set_stop(1)
    # client.set_open(1)
    # while True:
    #     try:
    #         message = input("> ")  # 🟢 从命令行获取输入
    #
    #         if message.lower() == "exit":
    #             print("[INFO] 退出客户端...")
    #             client.close()
    #             break
    #
    #         if message == "1,1":
    #             deta = client.get_arm_position_pose(1)
    #             print(deta)
    #
    #         if message == "1,2":
    #             deta = client.get_arm_position_pose(2)
    #         #     print(deta)
    #         #
    #         # if message == "2":
    #         #     deta = client.set_arm_position([0.912, -53.9504, 762.368, -0.775564, 0.0850297, -0.015176],"pose",2)
    #         #
    #         # if message == "3":
    #         #     i=1
    #         #     while i <=5:
    #         #         client.set_arm_position([-124.802, -689.052, -91.1308, 2.50594, -0.008948, 3.02646], "pose", 1)
    #         #         # time.sleep(0.1)
    #         #         print(client.get_arm_position_pose(1))
    #         #         client.set_arm_position([-104.802, -689.052, -91.1308, 2.50594, -0.008948, 3.02646], "pose", 1)
    #         #         print(client.get_arm_position_pose(1))
    #                 i += 1
    #
    #
    #         if message == "4":
    #             # deta = client.set_arm_position([50.5687,-42.0996,43.3454,2.36686,-42.1309,50.4586], "joint", 2)
    #         if message == "5":
    #             deta = client.send_message("stop,1")
    #         if message == "6":
    #             deta = client.send_message("open,1")
    #
    #
    #
    #     except KeyboardInterrupt:
    #         print("\n[INFO] 终止客户端...")
    #         client.close()
    #         break
