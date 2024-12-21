import h5py
import os
import numpy as np
import cv2
import io
import time


def create_folder(path):
    """创建文件夹"""
    os.makedirs(path, exist_ok=True)


def save_image(name, image, output_folder, path_parts, idx=None):
    """保存解压后的图像数据到文件夹"""
    # 生成基名称，避免创建过多层级的文件夹
    base_name = "_".join(path_parts).replace('/', '_').replace('\\', '_')

    # 为每个基名称创建一个单独的文件夹
    image_folder = os.path.join(output_folder, "images", base_name)
    create_folder(image_folder)

    # 生成唯一的文件名
    if idx is not None:
        file_name = f"{base_name}_{idx + 1}.png"
    else:
        file_name = f"{base_name}.png"
    file_path = os.path.join(image_folder, file_name)

    # 保存图像（OpenCV 使用 BGR 格式，转换为 RGB 后保存）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file_path, image_rgb)
    return file_path


def save_text(name, data, output_folder, path_parts):
    """保存非图像数据到文件夹"""
    # 根据数据路径命名
    base_name = "_".join(path_parts).replace('/', '_').replace('\\', '_') if len(path_parts) > 0 else name
    file_name = f"{base_name}.txt"
    file_path = os.path.join(output_folder, "texts", file_name)

    # 创建存放文本的目录
    create_folder(os.path.dirname(file_path))

    # 保存数据
    np.savetxt(file_path, data, fmt='%s')
    return file_path


def load_compressed_image_cv2(compressed_data):
    """
    使用 OpenCV 将压缩的图像数据转换为图像。
    """
    try:
        # 将 uint8 数组转换为字节流
        image_np = np.frombuffer(compressed_data, dtype=np.uint8)
        # 使用 OpenCV 解码图像（保持 BGR 格式）
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV 无法解码图像数据")
        return image
    except Exception as e:
        print(f"加载压缩图像失败: {e}")
        return None


def process_dataset(name, obj, hdf, output_directory, compress_len_right_wrist, compress_len_top, image_count,
                    data_count):
    """
    处理单个数据集，根据路径判断其类型并进行相应处理。
    """
    if isinstance(obj, h5py.Dataset):
        path_parts = name.split('/')

        # 处理图像数据
        if 'images' in path_parts:
            # 确定是 right_wrist 还是 top
            if 'right_wrist' in path_parts:
                dataset_type = 'right_wrist'
                compress_len_array = compress_len_right_wrist
            elif 'top' in path_parts:
                dataset_type = 'top'
                compress_len_array = compress_len_top
            else:
                print(f"未知的图像子类型: {name}")
                return image_count, data_count

            try:
                image_dataset = hdf[name]
                num_frames = image_dataset.shape[0]
                print(f"开始处理图像数据集: {name}，共 {num_frames} 帧。")

                for i in range(num_frames):
                    current_compress_len = compress_len_array[i]
                    if current_compress_len <= 0 or current_compress_len > image_dataset.shape[1]:
                        print(f"警告：帧 {i + 1} 的 compress_len 无效（{current_compress_len}），跳过。")
                        continue

                    # 提取压缩图像数据
                    compressed_data = image_dataset[i][:current_compress_len]

                    # 解压缩图像
                    decompressed_image = load_compressed_image_cv2(compressed_data)
                    if decompressed_image is not None:
                        file_path = save_image(name, decompressed_image, output_directory, path_parts, idx=i)
                        image_count += 1
                        print(f"已保存图像: {file_path}")
                    else:
                        print(f"加载图像 {name} 的第 {i + 1} 帧失败，compress_len={current_compress_len}")
            except Exception as e:
                print(f"处理图像数据集 {name} 时出错: {e}")

        # 处理文本数据
        else:
            try:
                data = np.array(obj)
                file_path = save_text(name, data, output_directory, path_parts)
                data_count += 1
                print(f"已保存数据: {file_path}")
            except Exception as e:
                print(f"保存数据 {name} 时出错: {e}")

    return image_count, data_count


def process_hdf5_file(hdf5_file_path, output_directory, extract_images=True, extract_data=True):
    """
    处理 HDF5 文件：递归读取内容，图像数据解压保存为 PNG 文件，其他数据保存为 TXT 文件。
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

            # 读取 compress_len 数据集
            if 'compress_len' in hdf:
                compress_len = hdf['compress_len'][:]
                right_wrist_shape = hdf['observations/images/right_wrist'].shape[0]
                top_shape = hdf['observations/images/top'].shape[0]
                if compress_len.shape != (2, right_wrist_shape):
                    print("错误：compress_len 的形状与 right_wrist 数据集的帧数不匹配！")
                    return
                compress_len_right_wrist = np.round(compress_len[0]).astype(int)
                compress_len_top = np.round(compress_len[1]).astype(int)
                print("成功读取 compress_len 数据集。")
            else:
                print("错误：compress_len 数据集不存在！")
                return

            # 遍历所有数据集
            def traverse(name, obj):
                nonlocal image_count, data_count
                image_count, data_count = process_dataset(
                    name, obj, hdf, output_directory,
                    compress_len_right_wrist, compress_len_top,
                    image_count, data_count
                )

            hdf.visititems(traverse)

    except Exception as e:
        print(f"处理 HDF5 文件时出错: {e}")

    print(f"\n处理完成。共保存图像 {image_count} 张，数据集 {data_count} 个到 '{output_directory}' 目录下。")


if __name__ == "__main__":
    # 示例使用
    input_hdf5 = r"D:\BYD\git_ku\ACT_plus_plus-master\ACT_plus_plus-master\hdf5_file\save_dir\episode_55.hdf5"
    output_dir = r"D:\BYD\jieya"

    # 选择提取类型：提取图像和其他数据
    extract_images_only = True
    extract_data_only = True

    start_time = time.time()
    process_hdf5_file(input_hdf5, output_dir, extract_images=extract_images_only, extract_data=extract_data_only)
    end_time = time.time()

    print(f"总耗时: {end_time - start_time:.2f}秒")
