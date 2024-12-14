import tkinter as tk
# from Robotic_Arm.rm_robot_interface import *
import time
import threading
from get_data import QposRecorder
QposRecorder = QposRecorder()
# 假设电磁铁控制程序
# real_right_arm = Arm(RM65, "192.168.1.18")

# 全局变量，用于记录按钮状态
# button_state = {"power_on": False, "power_off": False}


def power_on_magnet():
    """
    按下“上电”按钮，设置状态为 True。
    """
    QposRecorder.real_right_arm.rm_set_tool_voltage(3)
    print("电磁铁已上电")

def power_off_magnet():
    """
    按下“下电”按钮，设置状态为 True。
    """
    QposRecorder.real_right_arm.rm_set_tool_voltage(0)  # 下电
    print("电磁铁已下电")
# def control_magnet():
#     """
#     持续运行的循环，依据按钮状态控制电磁铁。
#     """
#     while True:
#         if button_state["power_on"]:
#             QposRecorder.real_right_arm.rm_set_tool_voltage(3)  # 上电
#             print("电磁铁已上电")  # 控制台输出（可选）
#             # button_state["power_on"] = False  # 复位状态
#
#         elif button_state["power_off"]:
#             QposRecorder.real_right_arm.rm_set_tool_voltage(0)  # 下电
#             print("电磁铁已下电")  # 控制台输出（可选）
#             # button_state["power_off"] = False  # 复位状态
#
#         # time.sleep(0.1)  # 控制循环频率


# def start_control_loop():
#     """
#     启动控制循环线程。
#     """
#     control_thread = threading.Thread(target=control_magnet, daemon=True)
#     control_thread.start()


# 创建主窗口
root = tk.Tk()
root.title("电磁铁控制")
root.geometry("300x200")  # 设置窗口大小

# 上电按钮
button_on = tk.Button(root, text="上电", command=power_on_magnet, width=15, height=2)
button_on.pack(pady=20)

# 下电按钮
button_off = tk.Button(root, text="下电", command=power_off_magnet, width=15, height=2)
button_off.pack(pady=20)

# 启动电磁铁控制循环线程
# start_control_loop()

# 主循环
root.mainloop()
