from gpcontrol import GPCONTROL
import time
import tqdm
if __name__ == "__main__":
    gp = GPCONTROL()
    gp.start_gp()   # 启动线程
    time.sleep(2)   # 给线程一点时间初始化

    try:
        while True:
            print("Main thread printing:", gp.state)
            time.sleep(1)  # 主线程必须阻塞，否则线程会随进程结束
    except KeyboardInterrupt:
        gp.stop()
