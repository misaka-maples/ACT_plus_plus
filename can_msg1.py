import serial
import time
import struct
import tkinter as tk
from tkinter import ttk

# 设备默认串口
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 50000

# 设置can1参数
set_can1 = b'\x49\x3B\x42\x57\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E'

# 开启can0、1通道
start_can1 = b'\x49\x3B\x44\x57\x01\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x45\x2E'


def open_serial(port=DEFAULT_SERIAL_PORT, baudrate=BAUD_RATE):
    """打开串口"""
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"串口 {port} 已打开，波特率 {baudrate}")
        return ser
    except Exception as e:
        print(f"无法打开串口 {port}: {e}")
        return None


def send_data(ser, data):
    """发送数据到串口"""
    if ser and ser.is_open:
        ser.write(data)
        print(f"发送数据: {data.hex()}")
    else:
        print("串口未打开，无法发送数据")


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


def read_data(ser):
    """读取串口返回数据并过滤符合头尾要求的数据"""
    if ser and ser.is_open:
        data = ser.read(32)  # 读取最大 64 字节
        if data:
            valid_frames = filter_can_data(data)
            if valid_frames:
                for frame in valid_frames:
                    print(f"接收符合条件的CAN数据: {frame.hex()}")
                return valid_frames
            else:
                print("未收到符合条件的数据帧")
        else:
            print("未收到数据")
    else:
        print("串口未打开，无法读取数据")
    return None


def send_can_data(ser, can_id, data, channel):
    """
    发送 CAN 数据帧
    :param ser: 串口对象
    :param can_id: 4字节 CAN ID
    :param data: 发送数据，最大 64 字节
    """
    can_id_bytes = struct.pack(">I", can_id)  # CAN ID 转换成 4字节

    data_length = len(data)
    if data_length > 64:
        data = data[:64]  # 限制数据长度为 64 字节

    frame_header = b'\x5A'  # 帧头
    frame_info_1 = (data_length | channel << 7).to_bytes(1, 'big')  # CAN通道0, DLC数据长度
    frame_info_2 = b'\x00'  # 发送类型: 正常发送, 标准帧, 数据帧, 不加速
    frame_data = data.ljust(64, b'\x00')  # 数据填充到 64 字节
    frame_end = b'\xA5'  # 帧尾

    send_frame = frame_header + frame_info_1 + frame_info_2 + can_id_bytes + frame_data[:data_length] + frame_end
    print("发送 CAN 帧:", send_frame.hex())
    send_data(ser, send_frame)


def calculate_can_data(slider_value):
    """基于滑动条值计算 CAN 数据"""
    # 假设左边为低值，右边为高值
    left_data = b'\x00\x00\xFF\xFF\xFF\xFF\x00\x00'
    right_data = b'\x00\xFF\xFF\xFF\xFF\xFF\x00\x00'

    # 简单的比例映射：根据slider_value计算CAN数据的变化
    new_data = bytearray()
    for i in range(len(left_data)):
        new_byte = int(left_data[i] + (right_data[i] - left_data[i]) * slider_value)
        new_data.append(new_byte)

    return bytes(new_data)


# GUI 部分
class CANApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CAN 数据发送")
        self.ser = open_serial()  # 打开串口
        self.is_sending = False
        self.is_configured = False  # 配置标志位

        # 滑动条，设置较长的长度
        self.slider = tk.Scale(root, from_=0, to=1, orient="horizontal", resolution=0.01, label="调节 CAN 数据",
                               length=500)
        self.slider.pack(padx=20, pady=20)

        # 启动按钮
        self.start_button = tk.Button(root, text="开始发送", command=self.start_sending)
        self.start_button.pack(pady=10)

        # 停止按钮
        self.stop_button = tk.Button(root, text="停止发送", command=self.stop_sending)
        self.stop_button.pack(pady=10)

    def start_sending(self):
        self.is_sending = True
        self.send_can_data()

    def stop_sending(self):
        self.is_sending = False

    def send_can_data(self):
        if self.ser and self.is_sending:
            # 配置设备（仅第一次）
            if not self.is_configured:
                send_data(self.ser, set_can1)  # 发送配置指令
                read_data(self.ser)
                send_data(self.ser, start_can1)  # 启动 CAN 通道
                read_data(self.ser)
                self.is_configured = True  # 标记为已配置

            # 发送 CAN 数据，基于拖动条值
            can_data = calculate_can_data(self.slider.get())
            send_can_data(self.ser, 0x001, can_data, 0x01)  # 发送 CAN 数据
            read_data(self.ser)
            self.root.after(100, self.send_can_data)  # 每100毫秒继续发送

    def close(self):
        if self.ser:
            self.ser.close()


# 启动GUI应用
def main():
    root = tk.Tk()
    app = CANApp(root)

    # 在关闭窗口时停止发送并关闭串口
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_sending(), app.close(), root.quit()))
    root.mainloop()


if __name__ == "__main__":
    main()
