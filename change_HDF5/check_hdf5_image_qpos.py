import h5py
import os
import numpy as np
from PIL import Image
import time


def create_folder(path):
    """创建文件夹"""
    os.makedirs(path, exist_ok=True)


def is_image_data(name, shape):
    """
    判断数据是否是图像数据：
    - 3D 数据：彩色图 (H, W, C)，C 为 1, 3, 4
    - 4D 数据：批量图像 (N, H, W, C)，C 为 1, 3, 4
    """
    if len(shape) == 3 and shape[2] in [1, 3, 4]:  # (H, W, C) -> 彩色图
        return True
    if len(shape) == 4 and shape[3] in [1, 3, 4]:  # (N, H, W, C) -> 批量图像
        return True
    return False


def save_image(name, data, output_folder, path_parts, idx=None):
    """保存图像数据到文件夹"""
    # 根据路径分类（如 top 和 right_wrist）
    category = path_parts[1] if len(path_parts) > 1 else "root"
    if len(path_parts) > 2:  # 进一步细分相机视角
        subcategory = path_parts[2]
        category_folder = os.path.join(output_folder, "images", category, subcategory)
    else:
        category_folder = os.path.join(output_folder, "images", category)
    create_folder(category_folder)

    # 生成唯一的文件名
    base_name = "_".join(path_parts[2:]).replace('/', '_').replace('\\', '_')  # 从相机部分开始
    if idx is not None:
        file_name = f"{base_name}_{idx + 1}.png"
    else:
        file_name = f"{base_name}.png"
    file_path = os.path.join(category_folder, file_name)

    # 如果数据是 BGR 通道，转换为 RGB
    if len(data.shape) == 3 and data.shape[2] == 3:
        data = data[..., ::-1]  # BGR 转 RGB

    # 保存图像
    image = Image.fromarray(data, mode='RGB' if len(data.shape) == 3 and data.shape[2] == 3 else 'L')
    image.save(file_path, format='PNG', compress_level=3)
    return file_path


def save_text(name, data, output_folder, path_parts):
    """保存非图像数据到文件夹"""
    # 根据数据路径命名
    base_name = path_parts[-1] if len(path_parts) > 0 else name
    file_name = f"{base_name}.txt"
    file_path = os.path.join(output_folder, "texts", file_name)

    # 保存数据
    create_folder(os.path.dirname(file_path))
    np.savetxt(file_path, data, fmt='%s')
    return file_path


def process_hdf5_file(hdf5_file_path, output_directory, extract_images=True, extract_data=True):
    """
    处理 HDF5 文件：递归读取内容，图像数据分类保存为 PNG 文件，其他数据保存为 TXT 文件。
    - extract_images: 是否提取图像数据
    - extract_data: 是否提取非图像数据
    """
    # 检查输入文件
    if not os.path.exists(hdf5_file_path):
        print(f"错误：文件 {hdf5_file_path} 不存在！")
        return

    create_folder(output_directory)

    image_count = 0
    data_count = 0

    try:
        with h5py.File(hdf5_file_path, 'r') as hdf:
            print(f"成功打开 HDF5 文件: {hdf5_file_path}")

            def traverse(name, obj):
                nonlocal image_count, data_count
                if isinstance(obj, h5py.Dataset):
                    data = np.array(obj)
                    shape = obj.shape
                    path_parts = name.split('/')

                    if extract_images and is_image_data(name, shape):
                        try:
                            if len(shape) == 4:  # 批量图像
                                for idx in range(shape[0]):
                                    file_path = save_image(name, data[idx], output_directory, path_parts, idx)
                                    image_count += 1
                                    print(f"已保存图像: {file_path}")
                            else:
                                file_path = save_image(name, data, output_directory, path_parts)
                                image_count += 1
                                print(f"已保存图像: {file_path}")
                        except Exception as e:
                            print(f"保存图像 {name} 时出错: {e}")
                    elif extract_data:
                        try:
                            file_path = save_text(name, data, output_directory, path_parts)
                            data_count += 1
                            print(f"已保存数据: {file_path}")
                        except Exception as e:
                            print(f"保存数据 {name} 时出错: {e}")

            hdf.visititems(traverse)

    except Exception as e:
        print(f"处理 HDF5 文件时出错: {e}")

    print(f"\n处理完成。共保存图像 {image_count} 张，数据集 {data_count} 个到 '{output_directory}' 目录下。")


if __name__ == "__main__":
    # 示例使用
    input_hdf5 = r"D:\BYD\git_ku\ACT_plus_plus-master\ACT_plus_plus-master\hdf5_file\save_dir\episode_23.hdf5"
    output_dir = r"D:\BYD\jieya"

    # 选择提取类型：提取图像和其他数据
    extract_images_only = True
    extract_data_only = True

    start_time = time.time()
    process_hdf5_file(input_hdf5, output_dir, extract_images=extract_images_only, extract_data=extract_data_only)
    end_time = time.time()

    print(f"总耗时: {end_time - start_time:.2f}秒")
