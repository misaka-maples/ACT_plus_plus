import h5py


def modify_hdf5_cameras(file_path):
    """
    修改 HDF5 文件中的摄像头数据。

    将 `observations/images/camera_top` 修改为：
    - `camera_top`
    - `left_wrist`
    - `right_wrist`

    参数:
        file_path (str): HDF5 文件的路径。
    """
    try:
        with h5py.File(file_path, 'r+') as f:
            f.attrs['compress']=False
            print(f.attrs['compress'])
            # f.attrs['compress'] = False  # 添加压缩属性
            # 检查是否存在原始的路径
            original_path = 'observations/images/top'
            if original_path not in f:
                raise KeyError(f"Path '{original_path}' not found in the HDF5 file.")
            original_path_pos = 'observations/qpos'
            if original_path_pos not in f:
                raise KeyError(f"Path '{original_path}' not found in the HDF5 file.")
            original_path_actions = 'action'
            if original_path_pos not in f:
                raise KeyError(f"Path '{original_path}' not found in the HDF5 file.")
            # 获取原始数据
            camera_top_data = f[original_path][:]
            qpos = f[original_path_pos][:]
            actions = f[original_path_actions][:]
            print(f'hdf5_edit_qpos: {qpos.shape}')
            print(f'hdf5_edit_action: {actions.shape}')

            # qpos = qpos[:, :7]
            # actions = actions[:, :7]

            # 创建新的路径并写入数据
            new_paths = [
                'observations/images/top',
                'observations/images/left_wrist',
                # 'observations/images/right_wrist'
            ]
            new_qpos_path = [
                'observations/qpos',
            ]
            new_actions_path = [
                'action'
            ]
            for path in new_actions_path:
                # 如果路径已存在，删除旧的路径
                if path in f:
                    del f[path]
                # 写入新的数据
                f.create_dataset(path, data=actions)
            for path in new_qpos_path:
                # 如果路径已存在，删除旧的路径
                if path in f:
                    del f[path]
                # 写入新的数据
                f.create_dataset(path, data=qpos)
            for path in new_paths:
                # 如果路径已存在，删除旧的路径
                if path in f:
                    del f[path]
                # 写入新的数据
                f.create_dataset(path, data=camera_top_data)

            print("Modification complete. Paths updated:")
            for path in new_paths:
                print(f"  - {path}")
            for path in new_qpos_path:
                print(f"  - {path}")
    except Exception as e:
        print(f"Error modifying HDF5 file: {e}")


if __name__ == '__main__':
    modify_hdf5_cameras('./is_sim_1_compress_0/episode_2.hdf5')
