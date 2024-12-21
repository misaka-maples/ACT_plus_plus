import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def read_single_txt_file(folder_path):
    """
    读取指定文件夹中的唯一一个TXT文件，并返回一个DataFrame。
    """
    txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
    if len(txt_files) == 0:
        print(f"错误：文件夹 {folder_path} 中没有TXT文件。")
        return pd.DataFrame()
    elif len(txt_files) > 1:
        print(f"警告：文件夹 {folder_path} 中有多个TXT文件，仅读取第一个：{os.path.basename(txt_files[0])}")

    file = txt_files[0]
    try:
        data = pd.read_csv(file, sep='\s+', header=None)
        if data.shape[1] != 7:
            print(f"警告：文件 {os.path.basename(file)} 的列数不是7，实际列数为 {data.shape[1]}，将被跳过。")
            return pd.DataFrame()
        return data
    except Exception as e:
        print(f"错误：无法读取文件 {os.path.basename(file)}，错误信息：{e}")
        return pd.DataFrame()

def plot_individual_charts(data_qpos, data_action, output_folder):
    """
    为每个指标绘制单独的折线图，并保存为独立的图片文件。
    """
    if data_qpos.empty or data_action.empty:
        print("错误：其中一个数据集为空，无法绘图。")
        return

    # 在qpos数据前填充一行0000000
    # zero_row = pd.DataFrame([[0]*data_qpos.shape[1]], columns=data_qpos.columns)
    data_qpos_padded = pd.concat([ data_qpos], ignore_index=True)
    print("已在qpos数据前填充一行0000000。")

    # 对action数据集进行-2操作


    data_action_adjusted = data_action - 2
    data_action_adjusted_0 = pd.DataFrame([[0] * data_action_adjusted.shape[1]], columns=data_action_adjusted.columns)
    data_action_adjusted = pd.concat([data_action_adjusted_0, data_action_adjusted], ignore_index=True)
    print("已对action数据进行-2操作。")

    # 确保两个数据集的行数一致，若不一致则截断到最小行数
    min_length = min(len(data_qpos_padded), len(data_action_adjusted))
    if min_length == 0:
        print("错误：数据集在填充和截断后没有数据行。")
        return
    data_qpos_padded = data_qpos_padded.iloc[:min_length].reset_index(drop=True)
    data_action_adjusted = data_action_adjusted.iloc[:min_length].reset_index(drop=True)
    print(f"数据行数（截断后）：{min_length}")

    # 打印数据示例
    print("qpos_padded 前5行：")
    print(data_qpos_padded.head())
    print("action_adjusted 前5行：")
    print(data_action_adjusted.head())

    # 生成横轴数据（行数）
    x_values = np.arange(1, min_length + 1)
    print(f"x_values 前10个：{x_values[:10]} ...")

    # 动态设置横轴刻度
    max_xticks = 20  # 期望的最大刻度数量
    step = 1 if min_length <= max_xticks else max(1, min_length // max_xticks)
    xticks = np.arange(1, min_length + 1, step)
    print(f"设置的xticks: {xticks}")

    # 创建单独的图表
    for i in range(data_qpos_padded.shape[1]):
        plt.figure(figsize=(15, 5))
        plt.plot(x_values, data_qpos_padded.iloc[:, i], label='qpos', color='blue')
        plt.plot(x_values, data_action_adjusted.iloc[:, i], label='action (减2)', color='orange')
        plt.title(f'轴 {i + 1} 对比')
        plt.ylabel(f'值 {i + 1}')
        plt.xlabel('行数')
        plt.xticks(xticks, rotation=45, ha='right')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # 保存单独的图表
        plot_file = os.path.join(output_folder, f'comparison_lineplot_axis_{i+1}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存对比折线图：{plot_file}")

def main(folder_qpos, folder_action, output_folder):
    """
    主函数，读取两个文件夹的TXT数据并生成对比折线图。
    """
    print(f"读取qpos文件夹的数据：{folder_qpos}")
    data_qpos = read_single_txt_file(folder_qpos)
    print(f"qpos的数据量：{len(data_qpos)} 行")

    print(f"读取action文件夹的数据：{folder_action}")
    data_action = read_single_txt_file(folder_action)
    print(f"action的数据量：{len(data_action)} 行")

    if data_qpos.empty and data_action.empty:
        print("错误：两个文件夹中都没有有效的数据。程序终止。")
        return
    elif data_qpos.empty:
        print("错误：qpos文件夹中没有有效的数据。程序终止。")
        return
    elif data_action.empty:
        print("错误：action文件夹中没有有效的数据。程序终止。")
        return

    print("开始绘制对比折线图...")
    plot_individual_charts(data_qpos, data_action, output_folder)
    print("所有对比折线图已生成。")

if __name__ == "__main__":
    import argparse

    # 设置命令行参数
    parser = argparse.ArgumentParser(description='读取两个文件夹中的TXT数据并生成对比折线图。')
    parser.add_argument('folder_qpos', type=str, nargs='?', default=r"D:\BYD\HDF5_make\qpos",
                        help='qpos文件夹的路径，例如 "D:\\BYD\\HDF5_make\\qpos"')
    parser.add_argument('folder_action', type=str, nargs='?', default=r"D:\BYD\HDF5_make\action",
                        help='action文件夹的路径，例如 "D:\\BYD\\HDF5_make\\action"')
    parser.add_argument('--output', type=str, default=r"D:\BYD\HDF5_make\out\02",
                        help='输出图表的文件夹路径，例如 "D:\\BYD\\HDF5_make\out"')

    args = parser.parse_args()

    # 检查文件夹路径是否存在
    if not os.path.isdir(args.folder_qpos):
        print(f"错误：qpos文件夹路径 '{args.folder_qpos}' 不存在或不是一个文件夹。")
        sys.exit(1)
    if not os.path.isdir(args.folder_action):
        print(f"错误：action文件夹路径 '{args.folder_action}' 不存在或不是一个文件夹。")
        sys.exit(1)

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        print(f"已创建输出文件夹：{args.output}")

    # 调用主函数
    main(args.folder_qpos, args.folder_action, args.output)
