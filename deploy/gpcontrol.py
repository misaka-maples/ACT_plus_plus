import threading
import time
import serial
class GPCONTROL(threading.Thread):
    def __init__(self, DEFAULT_SERIAL_PORTS=("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2")):
        super().__init__()
        self.serial_ok = True  # 串口正常标志

        self.state_flag = 128
        self.running = True
        self.control_command = ""
        self.DEFAULT_SERIAL_PORTS = DEFAULT_SERIAL_PORTS
        self.BAUD_RATE = 50000
        self.id = 1
        self.min_data = b'\x00\x00\xFF\xFF\xFF\xFF\x00\x00'
        self.max_data = b'\x00\xFF\xFF\xFF\xFF\xFF\x00\x00'
        self.ser = self.open_serial()
        self.is_sending = False
        self.state_data_1 = 128
        self.state_data_2 = 0
        self.task_complete = False
        self.is_configured = False
        self.state = ()
        # 初始化CAN设置
        self.send_data(b'\x49\x3B\x42\x57\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E')
        self.send_data(b'\x49\x3B\x42\x57\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E')
        self.read_data()
        self.send_data(b'\x49\x3B\x44\x57\x01\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E')
        self.read_data()

    def run(self):
        while self.running:
            if not self.serial_ok:
                print("[WARNING] 串口失联，尝试重连...")
                while not self.serial_ok and self.running:
                    self.reconnect_serial()
                    time.sleep(1)  # 每秒尝试一次，避免死循环卡死
                print("[INFO] 串口重连成功，继续工作")

            if self.state_data_1 < 0:
                self.state_data_1 = 0
            if self.state_data_2 < 0:
                self.state_data_2 = 0
            state_1 = self.set_gp_state(self.state_data_1, can_id=0)
            state_2 = self.set_gp_state(self.state_data_2, can_id=1)
            self.state = [state_1, state_2]
            time.sleep(0.2)

    def open(self):
        self.state_data_2=255
    def close_(self):
        self.state_data_2=0
    def stop(self):
        self.running = False
        print("[INFO] Gripper thread stopping...")

    def open_serial(self):
        for port in self.DEFAULT_SERIAL_PORTS:
            try:
                ser = serial.Serial(port, self.BAUD_RATE, timeout=1)
                print(f"串口 {port} 已打开，波特率 {self.BAUD_RATE}")
                return ser
            except Exception as e:
                print(f"无法打开串口 {port}: {e}")
        print(f"无法打开任何串口: {', '.join(self.DEFAULT_SERIAL_PORTS)}")

    def set_state(self,value):
        """修改 self.state_flag"""
        self.state_data_1 = value[0]
        self.state_data_2 = value[1]
        time.sleep(2)
        return self.state
    def send_data(self, data):
        """发送数据到串口"""
        ser = self.ser
        if ser and ser.is_open:
            try:
                ser.write(data)
                # print(f"发送数据: {data.hex()}")
            except Exception as e:
                print(f"[ERROR] 发送数据失败: {e}")
                self.serial_ok = False
        else:
            print("[ERROR] 串口未打开，无法发送数据")
            self.serial_ok = False


    def filter_can_data(self, data):
        """根据头（0x5A）和尾（0xA5）过滤数据"""
        valid_frames = []

        # 查找所有以 0x5A 开头并以 0xA5 结尾的数据帧
        start_idx = 0
        while start_idx < len(data):
            # 查找下一个0x5A
            start_idx = data.find(b'\x5A', start_idx)
            if start_idx == -1:  # 如果找不到0x5A，退出循环
                break

            # 查找下一个0xA5
            end_idx = data.find(b'\xA5', start_idx)
            if end_idx == -1:  # 如果找不到0xA5，退出循环
                break

            # 提取有效数据帧（包括0x5A和0xA5）
            frame = data[start_idx:end_idx + 1]

            # 确保数据帧长度合理（至少 8 字节）
            if len(frame) >= 8:
                valid_frames.append(frame)

            # 设置起始索引，继续查找下一个帧
            start_idx = end_idx + 1
        return valid_frames
    def read_data(self):
        """读取串口返回数据并过滤符合头尾要求的数据"""
        ser = self.ser
        if ser and ser.is_open:
            try:
                data = ser.read(32)  # 读取最大 32 字节
                if data:
                    valid_frames = self.filter_can_data(data)
                    if valid_frames:
                        back_data = 0
                        for frame in valid_frames:
                            if frame[:2].hex() == '5aff':
                                continue
                            else:
                                back_data = frame.hex()
                        return valid_frames, back_data
                else:
                    print("[WARNING] 未收到数据")
            except Exception as e:
                print(f"[ERROR] 读取数据失败: {e}")
                self.serial_ok = False
        else:
            print("[ERROR] 串口未打开，无法读取数据")
            self.serial_ok = False
        return None


    def send_can_data(self, can_id, data, channel):
        """
        发送 CAN 数据帧
        :param ser: 串口对象
        :param can_id: 4字节 CAN ID
        :param data: 发送数据，最大 64 字节
        """
        can_id_bytes = can_id  # CAN ID 转换成 4字节

        data_length = len(data)
        if data_length > 64:
            data = data[:64]  # 限制数据长度为 64 字节
        channel = channel & 0x01  # 确保 channel 只有1位
        frame_header = b'\x5A'  # 帧头
        frame_info_1 = (data_length | channel << 7).to_bytes(1, 'big')  # CAN通道0, DLC数据长度
        frame_info_2 = b'\x00'  # 发送类型: 正常发送, 标准帧, 数据帧, 不加速
        frame_data = data.ljust(64, b'\x00')  # 数据填充到 64 字节
        frame_end = b'\xA5'  # 帧尾

        send_frame = frame_header + frame_info_1 + frame_info_2 + can_id_bytes + frame_data[:data_length] + frame_end
        # print("发送 CAN 帧:", send_frame.hex())
        self.send_data(send_frame)
        # _,data = self.read_data()
        # return data
    def open_half_gp(self):
        half_open_gp = b'\x00\x7f\xFF\xFF\xFF\xFF\x00\x00'
        while 1:
            self.send_can_data(b'\x00\x00\x00\x01', half_open_gp, 0x01)
            data = self.read_data() 
            if data is not None:
                _, gpdata = data
                while gpdata == 0:
                    self.send_can_data(b'\x00\x00\x00\x01', half_open_gp, 0x01)
                    data = self.read_data()
                    if data is not None:
                        _, gpdata = data
                gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
                return [gpstate,gppos,gpforce]
        
    def open_all_gp(self):
        self.state_data_2=255
        open_gp = b'\x00\xff\xFF\xFF\xFF\xFF\x00\x00'
        while 1:
            self.send_can_data(b'\x00\x00\x00\x01', open_gp, 0x01)
            data = self.read_data() 
            if data is not None:
                _, gpdata = data
                while gpdata == 0:
                    self.send_can_data(b'\x00\x00\x00\x01', open_gp, 0x01)
                    data = self.read_data()
                    if data is not None:
                        _, gpdata = data
                gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
                return [gpstate,gppos,gpforce]
    def set_gp_state(self,value,can_id=1):
        assert 0 <= value <= 255, "value must be between 0 and 255"
        open_gp = b'\x00' + value.to_bytes(1, 'big') + b'\xFF\xFF\xFF\xFF\x00\x00'
        
        while 1:
            self.send_can_data(b'\x00\x00\x00\x01', open_gp, can_id)
            data = self.read_data() 
            if data is not None:
                _, gpdata = data
                while gpdata == 0:
                    self.send_can_data(b'\x00\x00\x00\x01', open_gp, can_id)
                    data = self.read_data()
                    if data is not None:
                        _, gpdata = data
                gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
                return [gpstate,gppos,gpforce]
    def close_gp(self):
        close_gp = b'\x00\x00\xFF\xFF\xFF\xFF\x00\x00'
        while 1:
            self.send_can_data(b'\x00\x00\x00\x01', close_gp, 0x01)
            data = self.read_data() 
            if data is not None:
                _, gpdata = data
                while gpdata == 0:
                    self.send_can_data(b'\x00\x00\x00\x01', close_gp, 0x01)
                    data = self.read_data()
                    if data is not None:
                        _, gpdata = data
                gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
                return [gpstate,gppos,gpforce]
        
    def control_gp(self, gpstate, gppos, gpforce):
        gpstate = gpstate.to_bytes(2, 'big')
        gppos = gppos.to_bytes(2, 'big')
        gpforce = gpforce.to_bytes(2, 'big')
        gpcontrol_data = b'\x00\x00' + gpstate + gppos + b'\x00\x00' + gpforce
        print(f"gpcontrol_data: {gpcontrol_data.hex()}")
            
        while 1:   
            self.send_can_data(b'\x00\x00\x00\x01', gpcontrol_data, 0x01)
            data = self.read_data()
            if data is not None:
                _, gpdata = data
                while gpdata == 0:
                    self.send_can_data(b'\x00\x00\x00\x01', gpcontrol_data, 0x01)
                    data = self.read_data()
                    if data is not None:
                        _, gpdata = data
                gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
                return [gpstate,gppos,gpforce]
            # return data
    
    def close(self):
        if self.ser:
            self.ser.close()
    def reconnect_serial(self):
        """尝试重新连接串口"""
        print("[WARNING] 尝试重新连接串口...")
        self.close()
        time.sleep(1)
        self.ser = self.open_serial()
        if self.ser and self.ser.is_open:
            print("[INFO] 串口重连成功，重新发送初始化指令")
            self.serial_ok = True
            # 重新初始化CAN设置
            self.send_data(b'\x49\x3B\x42\x57\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E')
            self.send_data(b'\x49\x3B\x42\x57\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E')
            self.read_data()
            self.send_data(b'\x49\x3B\x44\x57\x01\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E')
            self.read_data()
        else:
            print("[ERROR] 串口重连失败")
            self.serial_ok = False

if __name__ == "__main__":
    gpcontrol = GPCONTROL()
    gpcontrol.start()
    time.sleep(2)
    # gpcontrol.set_state([0,0])
    # time.sleep(2)
    while True:
        print(gpcontrol.state)

        gpcontrol.set_state([128,128])
        time.sleep(2)
        gpcontrol.set_state([128,0])
    # print(gpcontrol.state)
    # gpcontrol.state
    gpcontrol.stop()
    # print(gpcontrol.state)
