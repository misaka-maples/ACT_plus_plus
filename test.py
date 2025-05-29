import numpy as np
import time 
import sys
sys.path.append("./")
import os, datetime, sys
# 获取当前脚本的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录
parent_dir = os.path.abspath(os.path.join(current_dir, "./deploy"))

# 添加到 sys.path
sys.path.append(parent_dir)
from cjb.robot_control.tcp_tx import PersistentClient

from deploy.eval_function import eval,CAMERA_HOT_PLUG,GPCONTROL
# -*- coding: utf-8 -*-
"""
完整双线程实现版本 - 机械臂孔装配控制系统
包含：
1. 独立传感器数据采集线程（10ms周期）
2. 线程安全的全局状态管理
3. 完整行为树逻辑（Approach/Contact/Aligned/Push/Wiggle）
4. 进化策略优化接口
"""

import threading
import json
import time
import py_trees as pt
from cjb.force_sensor.sensor_yx import *
from cjb.robot_control.tcp_tx import PersistentClient

# ==================== 全局状态与配置 ====================
class GlobalState:
    """线程安全的全局状态容器"""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            # [-129.127, -810.615, -288.951, 2.4716, -0.00248988, 2.28385]
            # [-118.127, -815, -288.951, 2.4716, -0.00248988, 2.28385]
            'position': [0.0, 0.0, 0.0],
            'safe_position': [-129.127, -810.615, -288.951],
            'contact_position': [-118.127, -820, -288.951],
            'force': [0.0, 0.0, 0.0],
            'velocity': 100.0,
            'target_x': 0.0,
            'moving': False,
            'data_ready': False,
            'last_update': 0.0,
            'R': [0.0, 0.0, 0.0],
            'radius': 0.0,
            'angle': 0.0
        }

    def update(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if key in self.data:
                    self.data[key] = value
            self.data['last_update'] = time.time()

    def get(self, key):
        with self.lock:
            return self.data.get(key)

g_state = GlobalState()

CONFIG = {
    'max_velocity': 100.0,
    'acceleration': 100.0,
    'deceleration': 100.0,
    'control_period': 0.001,
    'default_pose': [2.4716, -0.00248988, 2.28385],
    'home_position': [-129.127, -810.615, -288.951],
    'target_position': [-115, -815, -288.951],
    'last_time': 0.0,
    'current_time': 0.0,
    'time_total': 0.0
}

# ==================== 数据采集线程 ====================
class SensorThread(threading.Thread):
    def __init__(self, serial_port, baud_rate):
        super().__init__(daemon=True)
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self.zero_offsets = None
        self.running = False
        self.force_threshold = 2

    def run(self):
        self.ser, self.zero_offsets = init_sensor(self.serial_port, self.baud_rate)
        if not self.ser:
            print("[ERROR] 传感器初始化失败")
            return

        client_data = PersistentClient('192.168.3.15', 8001)
        if not client_data:
            print("[ERROR] 客户端初始化失败")
            return

        self.running = True
        print("[SensorThread] 数据采集启动")

        while self.running:
            try:
                force_data = get_sensor_data(self.ser, self.zero_offsets)
                while not force_data:
                    force_data = get_sensor_data(self.ser, self.zero_offsets)
                if client_data:
                    pose = client_data.get_arm_position_pose(1)
                else:
                    pose = None

                if abs(force_data[1]) <= self.force_threshold and abs(force_data[2]) <= self.force_threshold:
                    g_state.update(
                        safe_position=pose[:3]
                    )
                if abs(force_data[0]) <= self.force_threshold and abs(force_data[1]) <= self.force_threshold and abs(force_data[2]) <= self.force_threshold:
                    g_state.update(
                        contact_position=pose[:3]
                    )
                if force_data and pose:
                    g_state.update(
                        position=pose[:3],
                        force=force_data[:3],
                        data_ready=True
                    )

            except Exception as e:
                print(f"[SensorThread] 数据采集异常: {str(e)}")

            time.sleep(CONFIG['control_period'])

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()
        print("[SensorThread] 数据采集停止")

# ==================== 机械臂控制接口 ====================
def get_position():
    return g_state.get('position')

def get_force_data():
    return g_state.get('force')

def set_end_effector_position(x, y, z, name):
    if not client:
        print("[ERROR] 机械臂未连接")
        return False
    x_current, y_current, z_current = get_position()
    F_x, F_y, F_z = get_force_data()

    if abs(x_current - x) < 1e-2 and abs(y_current - y) < 1e-2 and abs(z_current - z) < 1e-2:
        print(f"[{name}] 相同点，直接跳过")
        return True
    else:
        target_pose = [x, y, z] + CONFIG['default_pose']
        client.set_arm_position(
            target_pose, "pose", 1,
            g_state.get('velocity')
        )

        g_state.update(
            target_x=x,
            moving=abs(x_current - x) > 1e-2
        )

        if name == 'Wiggle':
            print(f"[{name}] 位置设置: X={x:.2f}, Y={y:.2f}, Z={z:.2f}, 半径={g_state.get('radius'):.2f}mm, 角度={np.degrees(g_state.get('angle')):.1f}°\n[{name}] 当前力值：Fx={F_x:.3f}N, Fy={F_y:.3f}N, Fz={F_z:.3f}N")
        else:
            print(f"[{name}] 位置设置: X={x:.2f}, Y={y:.2f}, Z={z:.2f}\n[{name}] 当前力值：Fx={F_x:.3f}N, Fy={F_y:.3f}N, Fz={F_z:.3f}N")
        return True

# ==================== 行为树节点实现 ====================
class ContactStateEstimator:
    def __init__(self):
        self.state = "Searching"
        self.x_buffer = []
        self.f_resx_buffer = []
        self.x_x0 = None
        self.v_ref = None
        self.epsilon = 0.01
        self.alpha = 0.1
        self.f_x_threshold = 0.5
        self.f_y_threshold = 0.5
        self.f_xy_aligned_threshold = 0.1
        self.f_x_min_threshold = 0.5

    def blackman_filter(self, data, N=50):
        n = np.arange(N)
        w = 0.42 - 0.5 * np.cos(2 * np.pi * n / N) + 0.08 * np.cos(4 * np.pi * n / N)
        w = w / np.sum(w)
        return np.convolve(data, w, mode='valid')

    def is_local_maximum(self, f_resx):
        self.f_resx_buffer.append(f_resx)
        if len(self.f_resx_buffer) > 50:
            self.f_resx_buffer.pop(0)
        if len(self.f_resx_buffer) < 11:
            return False

        filtered = self.blackman_filter(np.array(self.f_resx_buffer))
        idx = len(filtered) - 1
        if idx < 5:
            return False

        current = filtered[idx - 5]
        if current < self.f_x_min_threshold:
            return False

        is_max = True
        for i in range(1, 6):
            if idx - 5 - i >= 0 and filtered[idx - 5 - i] >= current:
                is_max = False
            if idx - 5 + i < len(filtered) and filtered[idx - 5 + i] >= current:
                is_max = False
        return is_max

    def update(self):
        x_x, _, _ = get_position()
        F_x, F_y, F_z = get_force_data()

        self.x_buffer.append(x_x)
        if len(self.x_buffer) > 5:
            self.x_buffer.pop(0)

        if len(self.x_buffer) >= 5:
            filtered = self.blackman_filter(np.array(self.x_buffer))
            mu, sigma = np.mean(filtered), np.std(filtered)
            z_score = (x_x - mu) / sigma if sigma != 0 else 0
        else:
            z_score = 0

        if self.state == "Searching":
            if self.x_x0 is None:
                self.x_x0 = x_x
            if abs(x_x - self.x_x0) > self.epsilon:
                self.state = "Stuck"
        else:
            if self.state == "Stuck" and z_score > 1:
                self.state = "Unstuck"
                F_resx = F_x
                if self.is_local_maximum(F_resx) and \
                        abs(F_x) < self.f_xy_aligned_threshold and \
                        abs(F_y) < self.f_xy_aligned_threshold:
                    self.state = "Aligned"
                    self.v_ref = self.alpha * CONFIG['max_velocity']
            elif self.state != "Stuck" and self.v_ref and not g_state.get('moving'):
                self.state = "Stuck"
            elif self.state == "Unstuck" and \
                    (abs(F_x) > self.f_x_threshold or abs(F_y) > self.f_y_threshold):
                self.state = "Stuck"

        return self.state

class Finish(pt.behaviour.Behaviour):
    def __init__(self, name="Finish"):
        super().__init__(name=name)
        self.force_threshold = 0.2
        self.pos_threshold = CONFIG['target_position'][0]

    def update(self):
        current_x, current_y, current_z = get_position()
        F_x, F_y, F_z = get_force_data()
        if current_x >= self.pos_threshold and abs(F_x) < self.force_threshold and abs(F_y) < self.force_threshold and abs(F_z) < self.force_threshold:
            wiggle._reset_parameters()
            print(f"===== Finish，当前位置：X={current_x:.2f}, Y={current_y:.2f}, Z={current_z:.2f}, 定位耗时：{CONFIG['time_total']:.3f}s =====")
            CONFIG['last_time'] = 0.0
            CONFIG['current_time'] = 0.0
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE

class Approach(pt.behaviour.Behaviour):
    def __init__(self, name="Approach"):
        super().__init__(name=name)
        self.force_threshold = 0.2
        self.flag = False

    def update(self):
        while not self.flag:
            set_end_effector_position(*CONFIG['home_position'], name="Approach")
            self.flag = True
            return pt.common.Status.RUNNING
        while abs(get_position()[0] - CONFIG['home_position'][0]) < 1e-2:
            print("===== Approach 完成 =====")
            self.flag = False
            return pt.common.Status.SUCCESS
        return pt.common.Status.RUNNING

class Contact(pt.behaviour.Behaviour):
    def __init__(self, name="Contact"):
        super().__init__(name=name)
        self.force_threshold = 0.2
        self.clean_flag = False
        self.run_flag = False

    def update(self):
        if CONFIG['last_time'] == 0.0:
            CONFIG['last_time'] = time.time()
        F_x, F_y, F_z = get_force_data()
        x, _, _ = get_position()
        if abs(F_x) > self.force_threshold:
            if abs(F_x) < 3:
                if g_state.get('R')[0] == 0:
                    pos = get_position()
                    g_state.update(R=pos[:3])
                    wiggle.setup()
                CONFIG['current_time'] = time.time()
                CONFIG['time_total'] = CONFIG['current_time'] - CONFIG['last_time']
                print(f"===== Contact 完成，竖向接触力Fx={F_x:.3f}N, 耗时：{CONFIG['time_total']:.3f}s =====")
                CONFIG['last_time'] = 0.0
                CONFIG['current_time'] = 0.0
                self.run_flag = False
                return pt.common.Status.SUCCESS
            else:
                print(f"*** 受力过大,清除原运动轨迹,返回安全位置：{g_state.get('contact_position')} ***\n当前力值：Fx={F_x:.3f}N,Fy={F_y:.3f}N,Fz={F_z:.3f}N")
                if not self.clean_flag:
                    client.set_stop(1)  # 清除原运动轨迹
                    self.clean_flag = True
                set_end_effector_position(*g_state.get('contact_position'), name="Contact")
                return pt.common.Status.RUNNING
        else:
            if not self.run_flag:
                set_end_effector_position(-118.127, -820, -288.951, name="Contact")
                self.run_flag = True
                return pt.common.Status.RUNNING
        return pt.common.Status.RUNNING

class Aligned(pt.behaviour.Behaviour):
    def __init__(self, name="Aligned"):
        super().__init__(name=name)
        self.estimator = ContactStateEstimator()

    def update(self):
        state = self.estimator.update()
        print(f"[Aligned] 当前接触状态为: {state}")
        return pt.common.Status.SUCCESS if state == "Aligned" else pt.common.Status.FAILURE

class Push(pt.behaviour.Behaviour):
    def __init__(self, name="Push"):
        super().__init__(name=name)
        self.force_threshold = 0.2
        self.pos_threshold = CONFIG['target_position'][0]
        self.clean_flag = False

    def update(self):
        x, y, z = get_position()
        F_x, F_y, F_z = get_force_data()

        if x >= self.pos_threshold:
            if abs(F_x) > self.force_threshold or abs(F_y) > self.force_threshold or abs(F_z) > self.force_threshold:
                print(f"===== Push 完成，当前深度：{x:.2f}，到达目标深度：{self.force_threshold}，但未处于销孔中心 =====")
                return pt.common.Status.FAILURE
            else:
                print(f"===== Push 完成，已处于销孔中心，当前位置：X={x:.2f}, Y={y:.2f}, Z={z:.2f} =====")
                return pt.common.Status.SUCCESS

        if abs(F_x) > self.force_threshold or abs(F_y) > self.force_threshold or abs(F_z) > self.force_threshold:
            return pt.common.Status.FAILURE
        else:
            x += 0.5
            set_end_effector_position(x, y, z, name="Push")
            return pt.common.Status.RUNNING

    # def update(self):
    #     x, y, z = get_position()
    #     F_x, F_y, F_z = get_force_data()
    #
    #     if x >= self.pos_threshold:
    #         if abs(F_x) > self.force_threshold or abs(F_y) > self.force_threshold or abs(F_z) > self.force_threshold:
    #             print(f"===== Push 完成，当前深度：{x:.2f}，到达目标深度：{self.pos_threshold}，但未处于销孔中心 =====")
    #             return pt.common.Status.FAILURE
    #         else:
    #             print(f"===== Push 完成，已处于销孔中心，当前位置：X={x:.2f}, Y={y:.2f}, Z={z:.2f} =====")
    #             return pt.common.Status.SUCCESS
    #
    #     if abs(F_x) > self.force_threshold or abs(F_y) > self.force_threshold or abs(F_z) > self.force_threshold:
    #         print(f"*** [Push] 受力过大,清除原运动轨迹,返回安全位置：{g_state.get('safe_position')} ***\n当前力值：Fx={F_x:.3f}N,Fy={F_y:.3f}N,Fz={F_z:.3f}N")
    #         if not self.clean_flag:
    #             client.set_stop(1)  # 清除原运动轨迹
    #             self.clean_flag = True
    #         set_end_effector_position(*g_state.get('safe_position'), name="Push")
    #         return pt.common.Status.FAILURE
    #     else:
    #         self.clean_flag = False
    #         x += 1
    #         set_end_effector_position(x, y, z, name="Push")
    #
    #         return pt.common.Status.RUNNING

class Wiggle(pt.behaviour.Behaviour):
    """摆动调整（带动态圆心和参数自适应）"""

    def __init__(self, name="Wiggle"):
        super().__init__(name=name)
        # 初始参数
        self.radius = 1.0
        self.max_radius = 10.0
        self.radius_step = 1
        self.n = 24

        # 初始参数备份
        self.origin_radius = 1.0
        self.origin_max = 10.0
        self.origin_step = 1
        self.origin_n = 24

        # 动态调整缩放因子（待优化）
        self.scale_radius = 0.5
        self.scale_max_radius = 0.5
        self.scale_radius_step = 0.5
        self.scale_n = 0.5

        self.angle = 0.0
        self.force_threshold = 0.2
        self.clean_flag = False

    def setup(self):
        pos = get_position()
        g_state.update(R=pos)
        print(f"[Wiggle] 设置初始圆心: R_x={pos[0]:.2f}, R_y={pos[1]:.2f}, R_z={pos[2]:.2f}")

    def _adjust_parameters(self):
        """动态调整摆动参数，使用优化的缩放因子"""
        self.radius_step = max(self.radius_step * self.scale_radius_step, 0.1)
        self.n = max(int(self.n * self.scale_n), 1)
        self.radius = max(self.radius * self.scale_radius, 0.5)
        self.max_radius = max(self.max_radius * self.scale_max_radius, 1)
        self.angle = 0
        print(f"[Wiggle] 参数调整: 半径={self.radius:.2f}mm, 步长={self.radius_step:.2f}, 分割数={self.n}")

    def _reset_parameters(self, update_time=True):
        """重置为初始参数，添加 update_time 参数控制是否更新时间统计"""
        if update_time:
            CONFIG['current_time'] = time.time()
            CONFIG['time_total'] = CONFIG['current_time'] - CONFIG['last_time']
        self.radius = self.origin_radius
        self.max_radius = self.origin_max
        self.radius_step = self.origin_step
        self.n = self.origin_n
        print("[Wiggle] 参数重置为初始值")

    def update(self):
        if CONFIG['last_time'] == 0.0:
            CONFIG['last_time'] = time.time()
        R_x, R_y, R_z = g_state.get('R')
        F_x, F_y, F_z = get_force_data()
        current_x, _, _ = get_position()

        # if abs(F_y) > 5 or abs(F_z) > 5:
        #     print(f"*** 受力过大,清除原运动轨迹,返回安全位置：{g_state.get('R')} ***\n当前力值：Fx={F_x:.3f}N,Fy={F_y:.3f}N,Fz={F_z:.3f}N")
        #     if not self.clean_flag:
        #         client.set_stop(1)  # 清除原运动轨迹
        #         self.clean_flag = True
        #     set_end_effector_position(*g_state.get('R'), name="Wiggle")
        #     return pt.common.Status.RUNNING
        # else:
        #     self.clean_flag = False

        if abs(R_x - current_x) > 0.5 and abs(F_x) < self.force_threshold and abs(F_y) < self.force_threshold and abs(F_z) < self.force_threshold:
            new_pos = get_position()
            g_state.update(R=new_pos[:3])
            self._adjust_parameters()
            print(f"[Wiggle] 圆心更新: R_x={new_pos[0]:.2f}, R_y={new_pos[1]:.2f}, R_z={new_pos[2]:.2f}")

        angle_step = 2 * np.pi / self.n
        self.angle += angle_step

        if self.angle >= 2 * np.pi:
            self.angle = 0
            self.radius = min(self.radius + self.radius_step, self.max_radius)

        g_state.update(radius=self.radius)
        g_state.update(angle=self.angle)

        y = R_y + self.radius * np.cos(self.angle)
        z = R_z + self.radius * np.sin(self.angle)

        set_end_effector_position(current_x, y, z, name="Wiggle")

        return pt.common.Status.RUNNING
    
class EvolutionStrategy:
    def __init__(self, param_bounds, K=50, max_generations=100):
        self.param_bounds = param_bounds
        self.K = K
        self.max_generations = max_generations
        self.param_names = list(param_bounds.keys())
        self.num_params = len(self.param_names)
        self.sigma = 0.1
        self.c = 10
        self.gamma = 0.9

    def initialize_population(self, xi, Sigma_epsilon):
        population = np.array([np.random.multivariate_normal(xi, Sigma_epsilon) for _ in range(self.K)])
        for i in range(self.K):
            for j, param_name in enumerate(self.param_names):
                low, high = self.param_bounds[param_name]
                population[i, j] = np.clip(population[i, j], low, high)
        return population

    def evaluate_fitness(self, params, num_rollouts=1, t_max=30.0):
        wiggle_params = dict(zip(self.param_names, params))
        wiggle.origin_radius = wiggle_params['radius']
        wiggle.origin_max = wiggle_params['max_radius']
        wiggle.origin_step = wiggle_params['radius_step']
        wiggle.origin_n = int(wiggle_params['n'])
        wiggle.scale_radius = wiggle_params['scale_radius']
        wiggle.scale_max_radius = wiggle_params['scale_max_radius']
        wiggle.scale_radius_step = wiggle_params['scale_radius_step']
        wiggle.scale_n = wiggle_params['scale_n']
        wiggle._reset_parameters(update_time=False)

        phi_values = []
        execution_times = []
        force_deviations = []
        for _ in range(num_rollouts):
            # 重置行为树状态
            tree.setup()

            # 重置机械臂到初始位置
            set_end_effector_position(*CONFIG['home_position'], name="Reset")
            while abs(get_position()[0] - CONFIG['home_position'][0]) > 1e-2:  # 等待到达
                time.sleep(CONFIG['control_period'])
            print(f"[Reset] 机械臂已重置到初始位置: {CONFIG['home_position']}")

            start_time = time.time()
            while time.time() - start_time < t_max:
                tree.tick()
                if tree.root.status == pt.common.Status.SUCCESS:
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                    phi_values.append(0)
                    F_x, F_y, F_z = get_force_data()
                    force_dev = np.mean([abs(F_x), abs(F_y), abs(F_z)])
                    force_deviations.append(force_dev)
                    break
                elif tree.root.status == pt.common.Status.FAILURE:
                    phi_values.append(1)
                    force_dev = np.mean([abs(F_x), abs(F_y), abs(F_z)])
                    force_deviations.append(force_dev)
                    break
                time.sleep(CONFIG['control_period'])

        avg_execution_time = np.mean(execution_times) if execution_times else t_max
        avg_phi = np.mean(phi_values)
        avg_force_dev = np.mean(force_deviations)
        fitness = (avg_execution_time / t_max) + avg_phi * avg_force_dev
        return fitness

    def optimize(self):
        try:
            print("进入optimize")
            xi = np.zeros(self.num_params)
            for j, param_name in enumerate(self.param_names):
                low, high = self.param_bounds[param_name]
                xi[j] = (low + high) / 2
            Sigma_epsilon = np.eye(self.num_params) * self.sigma**2

            best_fitness = float('inf')
            best_params = None

            for generation in range(self.max_generations):
                population = self.initialize_population(xi, Sigma_epsilon)
                fitness_values = np.array([self.evaluate_fitness(ind) for ind in population])

                min_fitness = np.min(fitness_values)
                max_fitness = np.max(fitness_values)
                if max_fitness == min_fitness:
                    J_tilde = np.zeros_like(fitness_values)
                else:
                    J_tilde = (fitness_values - min_fitness) / (max_fitness - min_fitness)

                exp_terms = np.exp(-self.c * J_tilde)
                P_k = exp_terms / np.sum(exp_terms)

                xi_new = np.sum(P_k[:, np.newaxis] * population, axis=0)
                diff = population - xi_new
                Sigma_temp = np.sum([P_k[k] * np.outer(diff[k], diff[k]) for k in range(self.K)], axis=0)
                Sigma_epsilon = Sigma_epsilon + self.gamma * (Sigma_temp - Sigma_epsilon)
                xi = xi_new

                best_idx = np.argmin(fitness_values)
                if fitness_values[best_idx] < best_fitness:
                    best_fitness = fitness_values[best_idx]
                    best_params = population[best_idx].copy()
                    print(f"Generation {generation}: Best Fitness = {best_fitness:.3f}, Best Params = {best_params}")
        except:
            pass

        return True

# camera = CAMERA_HOT_PLUG()
# time.sleep(5)
robot = PersistentClient('192.168.3.15', 8001)
gpcontrol = GPCONTROL()
gpcontrol.start()
gpcontrol.state_data_1 = 0
def gp(id):
    data = np.load(f'/workspace/exchange/grasp_pos6d_series/0522_target_{id}_pose_series.npy')
    np.set_printoptions(precision=4, suppress=True)
    # client = PersistentClient('192.168.3.15', 8001)
    gpcontrol.state_data_2 = 140
    time.sleep(2)
    for index,i in enumerate(data):
        print(i)
        if id == 0:


            i[0]-=20
        if index == 3:
            print("关闭夹爪")
            time.sleep(2)
            gpcontrol.state_data_2 = 0
            time.sleep(2)
            
            print(i)
        robot.set_arm_position(i.tolist(),'pose',2)
# def first_traj():
    


#     # main()


if __name__ == "__main__":
    gp(id=6)
    time.sleep(100)
    eval(camera=camera,
            persistentClient=robot,
            gp_contrpl=gpcontrol,
            real_robot=True,
            data_true=False,
            ckpt_dir=r'/workspace/exchange/5-9/exchange_overwrited/act_overwrited',
            ckpt_name='policy_step_10000_seed_0.ckpt',
            hdf5_path=r'/workspace/exchange/5-9/exchange_overwrited/episode_22.hdf5',
            state_dim=16,
            temporal_agg=True)
    time.sleep(2)
    
    client = PersistentClient('192.168.3.15', 8001)
    sensor_thread = SensorThread("/dev/ttyUSB0", 460800)

    root = pt.composites.Selector(name="Root", memory=False)
    finish = Finish(name="Finish")
    sequence = pt.composites.Sequence(name="Sequence", memory=True)
    approach = Approach(name="Approach")
    contact = Contact(name="Contact")
    repeat = pt.composites.Selector(name="Repeat", memory=False)
    aligned = Aligned(name="Aligned")
    push = Push(name="Push")
    wiggle = Wiggle(name="Wiggle")

    repeat.add_children([aligned, push, wiggle])
    sequence.add_children([approach, contact, repeat])
    root.add_children([finish, sequence])

    tree = pt.trees.BehaviourTree(root)

    sensor_thread.start()
    time.sleep(2)
    while not g_state.get('data_ready'):
        print("*** 数据未准备就绪 ***")
        time.sleep(0.01)

    print("===== 初始化完成，系统启动完成 =====")

    # 运行进化策略优化
    print("===== 开始进化策略优化 =====")
    param_bounds = {
        'radius': [0.5, 2.0],
        'max_radius': [5.0, 8.0],
        'radius_step': [0.1, 2.0],
        'n': [8, 48],
        'scale_radius': [0.1, 1.0],
        'scale_max_radius': [0.1, 1.0],
        'scale_radius_step': [0.1, 1.0],
        'scale_n': [0.1, 1.0]
    }
    es = EvolutionStrategy(param_bounds, K=1, max_generations=1)
    best_params = es.optimize()
    # wiggle_params = dict(zip(es.param_names, best_params))
    # wiggle.origin_radius = wiggle_params['radius']
    # wiggle.origin_max = wiggle_params['max_radius']
    # wiggle.origin_step = wiggle_params['radius_step']
    # wiggle.origin_n = int(wiggle_params['n'])
    # wiggle.scale_radius = wiggle_params['scale_radius']
    # wiggle.scale_max_radius = wiggle_params['scale_max_radius']
    # wiggle.scale_radius_step = wiggle_params['scale_radius_step']
    # wiggle.scale_n = wiggle_params['scale_n']
    # wiggle._reset_parameters(update_time=False)
    # with open("best_params.json", "w") as f:
    #     json.dump(wiggle_params, f)
    # print(data)
    print("第二段")
    eval(camera=camera,
        persistentClient=robot,
        gp_contrpl=gpcontrol,
        real_robot=True,
        data_true=False,
        ckpt_dir=r'/workspace/exchange/5-9/duikong/act',
        ckpt_name='policy_best.ckpt',
        hdf5_path=r'/workspace/exchange/5-9/duikong/episode_23.hdf5',
        state_dim=8,
        temporal_agg=True)