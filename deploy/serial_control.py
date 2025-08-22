# -*- coding: utf-8 -*-
import threading
import time
import serial
import serial.tools.list_ports

# 默认波特率
BAUD_RATE = 50000

# CAN 初始化指令
SET_CAN0 = b'\x49\x3B\x42\x57\x00' + b'\x00' * 15 + b'\x45\x2E'
SET_CAN1 = b'\x49\x3B\x42\x57\x01' + b'\x00' * 15 + b'\x45\x2E'
START_CAN = b'\x49\x3B\x44\x57\x01\x00\x01\x01' + b'\x00' * 12 + b'\x45\x2E'

class GripperCANController():
    def __init__(self, baudrate=BAUD_RATE):
        self.port = None
        self.baudrate = baudrate
        self.ser = None
        self.value_sources = {}
        self._running = threading.Event()  # 正确的运行标志
        self._thread = None
        self.gpstate = list([None, None])  # 初始化夹爪状态列表
        self.interval = 0.05
        self.gripper_id = 0x000
        self.value = 0
        self.bool = True
        self.start_signal = False
    def run(self):
        can_id = b'\x00\x00\x00\x08'
        while self._running.is_set():  # 如果线程被标记为运行状态
            can_data = self.calculate_can_data(self.value)
            if self.bool:
                _,self.gpstate[0] = self.set_gp_state(can_data, can_id, 0x000)
                _,self.gpstate[1] = self.set_gp_state(can_data, can_id, 0x001)
                self.bool = False
            canid,data = self.set_gp_state(can_data, can_id, self.gripper_id)
            if canid == '08':
                self.gpstate[0] = data
            elif canid == '88':
                self.gpstate[1] = data
            time.sleep(self.interval)
        else:
            time.sleep(self.interval)

    def start_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._running.set()
            self._thread = threading.Thread(target=self.run, daemon=True)
            self._thread.start()
        return True, "控制线程已启动"


    def stop_thread(self):
        """停止线程采集"""
        self._running.clear()  # 通知线程退出
        if self._thread is not None:
            self._thread.join()  # 等待线程安全退出
            self._thread = None
            self.close()
    def set_value_func(self,channel,value):
        if channel == 1:
            self.gripper_id =0x000
        elif channel == 2:
            self.gripper_id =0x001
        self.value = value
    def connect_ACM_port(self):
        port = self.get_serial_ports_list()
        if any('ACM' in dev for dev in port):
            self.open_serial(port[0])
        else:
            print("未找到ACM串口")
    def get_gpstate(self):
        _,gpdata = self.read_data(self.ser)
        if gpdata[:2] == '08':
            gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
            self.gpstate[0] = gpstate,gppos,gpforce 
        elif gpdata[:2] == '88':
            gpstate,gppos,gpforce = gpdata[16:18],gpdata[18:20],gpdata[22:24]
            self.gpstate[1] = gpstate,gppos,gpforce 
        if gpdata == 0:
            pass 
        return gpstate
    def get_serial_ports_list(self):
        """获取可用串口列表"""
        all_ports = [port.device for port in serial.tools.list_ports.comports()]
        acm_ports = [port for port in all_ports if "ACM" in port]
        if acm_ports:
            return acm_ports
        else:
            return [port.device for port in serial.tools.list_ports.comports()]
    def open_serial(self, port=None):
        self.port = port or self.port
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"串口 {self.port} 已打开")
            self.initialize_can()
            return True, f"串口 {self.port} 打开成功"
        except Exception as e:
            print(f"串口打开失败: {e}")
            return False, f"串口打开失败: {str(e)}"

    def is_open(self):
        return self.ser is not None and self.ser.is_open

    def close(self):
        if self.is_open():
            self.ser.close()
            print("串口已关闭")

    def initialize_can(self):
        if not self.is_open():
            raise Exception("串口未打开")

        for cmd in [SET_CAN0, SET_CAN1, START_CAN]:
            self.send_data(self.ser,cmd)
            time.sleep(0.1)
            self.read_data(self.ser)
        print("CAN 初始化完成")
        
    def reconnect_serial_connection(self):
        try:
            
            port = self.ser.port
            self.ser.close()
            self.ser = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=1)
            return True, f"串口 {port} 重连成功"
        except Exception as e:
            return False, f"重连失败: {str(e)}"

    @staticmethod
    def calculate_can_data(value=0):
        left_data = b'\x00\x00\xFF\xFF\xFF\xFF\x00\x00'
        right_data = b'\x00\xFF\xFF\xFF\xFF\xFF\x00\x00'
        new_data = bytearray()
        for i in range(len(left_data)):
            new_byte = int(left_data[i] + (right_data[i] - left_data[i])*value*0.01)
            new_data.append(new_byte)
        return bytes(new_data)

    @staticmethod
    def send_can_frame(ser, can_id_bytes, data, channel):
        frame_header = b'\x5A'
        frame_info_1 = (len(data) | (channel << 7)).to_bytes(1, 'big')
        frame_info_2 = b'\x00'
        frame_data = data.ljust(64, b'\x00')
        frame_end = b'\xA5'
        frame = frame_header + frame_info_1 + frame_info_2 + can_id_bytes + frame_data[:len(data)] + frame_end
        GripperCANController.send_data(ser, frame)
        # GripperCANController.read_data(ser)
    @staticmethod
    def send_data(ser:serial.Serial, data):
        if ser and ser.is_open:
            ser.write(data)
            # print(f"[{time.time():.2f}] 发送: {data.hex().upper()}")
    def set_gp_state(self,value,can_id=1, channel=0x000):
        while 1:
            self.send_can_frame(self.ser, can_id, value, channel)
            data = self.read_data(self.ser) 
            if data is not None:
                _, gpdata = data
                while gpdata == 0:
                    self.send_can_frame(self.ser, can_id, value, channel)
                    data = self.read_data(self.ser)
                    if data is not None:
                        _, gpdata = data

                canid, gpstate,gppos,gpforce = gpdata[2:4],gpdata[16:18],gpdata[18:20],gpdata[22:24]

                return canid,[gpstate,gppos,gpforce]
    @staticmethod
    def filter_can_data(data):
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
    @staticmethod
    def read_data(ser: serial.Serial):
        """读取串口返回数据并过滤符合头尾要求的数据"""
        
        if ser and ser.is_open:
            try:
                data = ser.read(32)  # 读取最大 32 字节
                if data:
                    valid_frames = GripperCANController.filter_can_data(data)
                    if valid_frames:
                        back_data = 0
                        # print(valid_frames)
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
        else:
            print("[ERROR] 串口未打开，无法读取数据")
        return None

if __name__ == "__main__":
    gpcontrol = GripperCANController()
    gpcontrol.connect_ACM_port()
    gpcontrol.start_thread()
    time.sleep(1)
    i = 0
    while i < 20:
        print(gpcontrol.gpstate)

        time.sleep(1)
        i += 1
        if i == 10:
            gpcontrol.stop_thread()

        if i == 11:
            gpcontrol.connect_ACM_port()
            gpcontrol.start_thread()