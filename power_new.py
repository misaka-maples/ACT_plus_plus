import tkinter as tk
from Robotic_Arm.rm_robot_interface import *
import time
import threading
# from get_data import QposRecorder
# 假设电磁铁控制程序
# real_right_arm = Arm(RM65, "192.168.1.18")

# 全局变量，用于记录按钮状态
# button_state = {"power_on": False, "power_off": False}

class QposRecorder:
    def __init__(self):
        self.joint_state_right=None
        self.real_right_arm = (RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E))
        self.arm_ini = self.real_right_arm.rm_create_robot_arm("192.168.1.18",8080, level=3)
        # self.robot_controller = RoboticArm("192.168.1.18", 8080, 3)
    def get_state(self):
        self.joint_state_right = self.real_right_arm.rm_get_current_arm_state()
        # print(f"get state test", self.joint_state_right)
        return_action = self.joint_state_right[1]['joint']
        # print(return_action)
        return return_action
QposRecorder0 = QposRecorder()

def power_on_magnet():
    """
    按下“上电”按钮，设置状态为 True。
    """
    QposRecorder0.real_right_arm.rm_set_tool_voltage(3)
    print("电磁铁已上电")

def power_off_magnet():
    """
    按下“下电”按钮，设置状态为 True。
    """
    QposRecorder0.real_right_arm.rm_set_tool_voltage(0)  # 下电
    print("电磁铁已下电")
root = tk.Tk()
root.title("电磁铁控制")
root.geometry("300x200")  # 设置窗口大小

# 上电按钮
button_on = tk.Button(root, text="上电", command=power_on_magnet, width=15, height=2)
button_on.pack(pady=20)

# 下电按钮
button_off = tk.Button(root, text="下电", command=power_off_magnet, width=15, height=2)
button_off.pack(pady=20)

# 主循环
root.mainloop()
