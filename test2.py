from deploy.get_data_auto_with_gp import QposRecorder

QposRecorder1=QposRecorder()
state = QposRecorder1.real_right_arm.rm_get_current_arm_state()
print(state)