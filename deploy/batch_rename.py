import os

def rename_hdf5_files(directory, start, end, new_start=60):
    filtered = []

    # 获取目录下所有的 .hdf5 文件
    files = [f for f in os.listdir(directory) if f.endswith('.hdf5')]
    files.sort()

    # 第一步：筛选出需要修改的文件
    for filename in files:
        try:
            file_index = int(filename.split("_")[1].split(".")[0])
            if start <= file_index <= end:
                filtered.append(filename)
        except ValueError:
            continue

    print(f"待处理文件：{filtered}")

    # 第二步：临时重命名，避免冲突
    temp_files = []
    for filename in filtered:
        old_path = os.path.join(directory, filename)
        temp_path = os.path.join(directory, f"temp_{filename}")
        os.rename(old_path, temp_path)
        temp_files.append(temp_path)

    # 第三步：从 new_start 开始正式重命名
    index = new_start
    for temp_path in temp_files:
        new_name = f'episode_{index}.hdf5'
        new_path = os.path.join(directory, new_name)

        if os.path.exists(new_path):
            print(f'目标文件 {new_name} 已存在，跳过重命名 {temp_path}')
        else:
            os.rename(temp_path, new_path)
            print(f'已将 {os.path.basename(temp_path)} 重命名为 {new_name}')
            index += 1

# 使用示例
directory = '/workspace/exchange/4-7'
rename_hdf5_files(directory, start=51, end=71, new_start=60)
