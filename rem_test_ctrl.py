from robotic_arm_package.robotic_arm import *  # 导入机械臂库
from depoly.policy_test import get_action
import math
                                   #action为弧度值，需要做匹配转化
# def radians_to_degrees(radians):
#     degrees = radians * (180 / math.pi)
#     return degrees
# 
#
# degree_value = radians_to_degrees(action)



# 创建机械臂实例
rem_right = Arm(RM65, "192.168.1.18")
# rem_left = Arm(RM65, "192.168.1.16")

#
# print(rem_right.Arm_Socket_State())
#
# print(rem_right.API_Version())

# rem_right.Movej_Cmd([0,0,0.1,0.1,0.1,5],5, 0, 0,True)
# rem_right.Movej_Cmd([0,20,0.1,0.1,0.1,5],5, 0, 0,True)
# rem_right.Movej_Cmd([0,40,0.1,0.1,0.1,5],5, 0, 0,True)

  #设置关节最大速度
# rem_right.Set_Joint_Speed(1,10,True)
# rem_right.Set_Joint_Speed(2,10,True)
# rem_right.Set_Joint_Speed(3,20,True)
# rem_right.Set_Joint_Speed(4,20,True)
# rem_right.Set_Joint_Speed(5,20,True)
# rem_right.Set_Joint_Speed(6,20,True)
# print(rem_right.Get_Joint_Speed())

res, joint=rem_right.Get_Joint_Degree()

print(joint)
# 透传低跟随
# for i in range(20):
#   joint[5] +=1
#   joint[1] +=0.5
#   joint[2] +=0.5
#   print(joint)
#   rem_right.Movej_CANFD(joint,False) #False为低跟随、True为高跟随
#   time.sleep(0.1)

res, joint,pose,arm_err,sys_err=rem_right.Get_Current_Arm_State()
print(joint)
# print(rem_right.Get_Joint_Acc())
# rem_right.Movej_CANFD([0,38,0,0,0,5], False) #运动指令

# rem_right.Set_Tool_Voltage(3,True) #末端工具电源
rem_right.Set_Tool_Voltage(0,True) #末端工具电源
rem_right.Arm_Socket_Close()
