import torch
import argparse
import pathlib
from torch import nn
import torchvision
import os
import time
import h5py
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np

import IPython
e = IPython.embed

def chunks(lst, n):
    """将列表 lst 分割成大小为 n 的子列表."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def expand_greyscale(t):
    """将灰度图像扩展为三通道."""
    return t.expand(3, -1, -1)

def main(args):
    #################################################
    batch_size = 256  # 批量大小
    #################################################

    # 从命令行参数中读取检查点路径和数据集目录
    ckpt_path = args.ckpt_path
    dataset_dir = args.dataset_dir
    ckpt_name = pathlib.PurePath(ckpt_path).name  # 获取检查点文件的名字
    dataset_name = ckpt_name.split('-')[1]  # 数据集名称
    repr_type = ckpt_name.split('-')[0]  # 表示类型（如 co-training）
    seed = int(ckpt_name.split('-')[-1][:-3])  # 随机种子

    if 'cotrain' in ckpt_name:
        repr_type += '_cotrain'  # 如果是协同训练，更新表示类型

    # 获取数据集中所有 episode 的索引，假设文件名为 'episode_0.hdf5', 'episode_1.hdf5', ...
    episode_idxs = [int(name.split('_')[1].split('.')[0]) for name in os.listdir(dataset_dir) if ('.hdf5' in name) and ('features' not in name)]
    episode_idxs.sort()
    assert len(episode_idxs) == episode_idxs[-1] + 1  # 确保没有缺失的 episode
    num_episodes = len(episode_idxs)  # 计算总的 episode 数量

    feature_extractors = {}  # 用于存储每个摄像头的特征提取模型

    for episode_idx in range(num_episodes):  # 遍历所有的 episode

        # 加载图像数据
        print(f'loading data')
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            image_dict = {}
            camera_names = list(root[f'/observations/images/'].keys())  # 获取摄像头名称
            print(f'Camera names: {camera_names}')
            for cam_name in camera_names:
                image = root[f'/observations/images/{cam_name}'][:]  # 加载图像数据
                uncompressed_image = []
                for im in image:
                    im = np.array(cv2.imdecode(im, 1))  # 解压缩图像
                    uncompressed_image.append(im)
                image = np.stack(uncompressed_image, axis=0)  # 将所有图像堆叠成一个数组

                image_dict[cam_name] = image  # 将每个摄像头的图像数据存储到字典中

        # 加载 ResNet18 模型并准备提取特征
        print(f'loading model')
        if not feature_extractors:  # 如果尚未加载过特征提取器（即模型）
            for cam_name in camera_names:
                resnet = torchvision.models.resnet18(pretrained=True)  # 加载预训练的 ResNet18 模型
                loading_status = resnet.load_state_dict(torch.load(ckpt_path.replace('DUMMY', cam_name)))  # 加载特定摄像头的模型
                print(cam_name, loading_status)
                resnet = nn.Sequential(*list(resnet.children())[:-1])  # 去掉最后的全连接层
                resnet = resnet.cuda()  # 将模型移至 GPU
                resnet.eval()  # 设置为评估模式
                feature_extractors[cam_name] = resnet  # 存储每个摄像头的特征提取器

        # 使用 ResNet 提取特征
        feature_dict = {}  # 用于存储提取的特征
        for cam_name, images in image_dict.items():  # 遍历每个摄像头的图像数据
            # 图像预处理
            image_size = 120  # 注意：图像分辨率已减少
            transform = transforms.Compose([
                transforms.Resize(image_size),  # 调整图像大小
                transforms.CenterCrop(image_size),  # 中心裁剪
                transforms.ToTensor(),  # 转换为张量
                transforms.Lambda(expand_greyscale),  # 如果是灰度图像，则扩展为三通道
                transforms.Normalize(
                    mean=torch.tensor([0.485, 0.456, 0.406]),  # 归一化参数（ResNet的标准值）
                    std=torch.tensor([0.229, 0.224, 0.225])),
            ])
            processed_images = []  # 存储处理后的图像
            for image in tqdm(images):  # 进度条显示图像处理进度
                image = Image.fromarray(image)  # 转换为 PIL 图像
                image = transform(image)  # 应用图像预处理
                processed_images.append(image)
            processed_images = torch.stack(processed_images).cuda()  # 将所有图像堆叠成一个批次，并移至 GPU

            # 使用模型提取特征
            all_features = []
            with torch.inference_mode():  # 禁用梯度计算
                for batch in chunks(processed_images, batch_size):  # 按批次处理
                    print('inference')
                    features = feature_extractors[cam_name](batch)  # 获取特征
                    features = features.squeeze(axis=3).squeeze(axis=2)  # 降维
                    all_features.append(features)
            all_features = torch.cat(all_features, axis=0)  # 合并所有批次的特征
            max_timesteps = all_features.shape[0]  # 获取特征的时间步数
            feature_dict[cam_name] = all_features  # 存储该摄像头的特征

        # 将提取的特征保存为 hdf5 文件
        dataset_path = os.path.join(dataset_dir, f'{repr_type}_features_seed{seed}_episode_{episode_idx}.hdf5')
        print(dataset_path)
        t0 = time.time()
        with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            features = root.create_group('features')  # 创建 'features' 数据组
            for cam_name, array in feature_dict.items():  # 遍历每个摄像头的特征
                cam_feature = features.create_dataset(cam_name, (max_timesteps, 512))  # 创建数据集
                features[cam_name][...] = array.cpu().numpy()  # 将特征保存为 numpy 数组
        print(f'Saving: {time.time() - t0:.1f} secs\n')  # 打印保存所花费的时间

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='cache features')
    parser.add_argument('--ckpt_path', type=str, required=True, help='ckpt_path')  # 模型路径
    parser.add_argument('--dataset_dir', type=str, required=True, help='dataset_dir')  # 数据集路径
    args = parser.parse_args()

    main(args)  # 调用主函数
