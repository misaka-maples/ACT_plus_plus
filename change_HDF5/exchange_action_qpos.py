import h5py
import os


def swap_hdf5_data(file_path, action_key="action", qpos_key="observations/qpos"):
    """
    修改 HDF5 文件，将指定 key 的数据对换。

    参数：
    - file_path (str): HDF5 文件路径
    - action_key (str): action 数据的键名
    - qpos_key (str): qpos 数据的键名
    """
    try:
        with h5py.File(file_path, 'r+') as hdf5_file:
            # 检查键是否存在
            if action_key not in hdf5_file or qpos_key not in hdf5_file:
                print(f"Error: '{action_key}' or '{qpos_key}' not found in the file '{file_path}'.")
                return

            # 检查形状是否一致
            if hdf5_file[action_key].shape != hdf5_file[qpos_key].shape:
                print(f"Error: Shape mismatch between '{action_key}' and '{qpos_key}' in '{file_path}'.")
                return

            # 读取数据
            action_data = hdf5_file[action_key][:]
            qpos_data = hdf5_file[qpos_key][:]

            # 交换数据
            hdf5_file[action_key][...] = qpos_data
            hdf5_file[qpos_key][...] = action_data

            print(f"Successfully swapped '{action_key}' and '{qpos_key}' data in '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while processing '{file_path}': {e}")


def process_multiple_files(directory, start=2, end=50, file_prefix="episode_", file_suffix=".hdf5"):
    """
    对多个 HDF5 文件进行处理，在指定范围内循环调用 swap_hdf5_data。

    参数：
    - directory (str): 文件所在的目录路径
    - start (int): 开始的文件编号
    - end (int): 结束的文件编号
    - file_prefix (str): 文件名前缀
    - file_suffix (str): 文件名后缀
    """
    for i in range(start, end + 1):
        file_name = f"{file_prefix}{i}{file_suffix}"
        file_path = os.path.join(directory, file_name)

        if os.path.exists(file_path):
            print(f"Processing file: {file_name}")
            swap_hdf5_data(file_path)
        else:
            print(f"File not found: {file_name}")


# 示例用法
directory = r"E:\0"
process_multiple_files(directory, start=0, end=11)

# # 示例用法
# file_path = r"D:\BYD\git_ku\ACT_plus_plus-master\ACT_plus_plus-master\hdf5_model\save_dir\episode_2.hdf5"
# swap_hdf5_data(file_path, action_key="action", qpos_key="observations/qpos")

