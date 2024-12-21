import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim

def compare_images(image_path1, image_path2, output_dir):
    """
    对比两张图像，使用像素差异、感知哈希（pHash）和结构相似性指数（SSIM）。

    参数:
    - image_path1 (str): 第一张图像的文件路径。
    - image_path2 (str): 第二张图像的文件路径。
    - output_dir (str): 保存对比结果的目录。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载图像
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None:
        print(f"无法加载图像: {image_path1}")
        return
    if img2 is None:
        print(f"无法加载图像: {image_path2}")
        return

    # 检查图像尺寸是否一致
    if img1.shape != img2.shape:
        print("图像尺寸不一致，无法进行对比。")
        return

    # 计算像素差异
    diff = cv2.absdiff(img1, img2)
    diff_percentage = np.sum(diff) / (diff.size * 255) * 100

    # 计算图像哈希（感知哈希）
    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    hash1 = imagehash.phash(img1_pil)
    hash2 = imagehash.phash(img2_pil)
    hash_diff = hash1 - hash2

    # 计算结构相似性（SSIM）
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_index, ssim_diff = ssim(img1_gray, img2_gray, full=True)

    # 打印对比结果
    print(f"像素差异百分比: {diff_percentage:.2f}%")
    print(f"感知哈希差异 (pHash): {hash_diff}")
    print(f"结构相似性指数 (SSIM): {ssim_index:.4f}")

    # 可视化对比
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title("图像1")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title("图像2")
    axs[1].axis("off")

    axs[2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    axs[2].set_title(f"像素差异\n({diff_percentage:.2f}%)")
    axs[2].axis("off")

    axs[3].imshow(ssim_diff, cmap='gray')
    axs[3].set_title(f"SSIM 差异\n(SSIM: {ssim_index:.4f})")
    axs[3].axis("off")

    # 在图中添加哈希差异
    fig.suptitle(f"感知哈希差异 (pHash): {hash_diff}", fontsize=16)

    # 保存对比结果
    image1_name = os.path.splitext(os.path.basename(image_path1))[0]
    image2_name = os.path.splitext(os.path.basename(image_path2))[0]
    comparison_path = os.path.join(output_dir, f"{image1_name}_vs_{image2_name}_comparison.png")
    plt.savefig(comparison_path)
    plt.close()

    print(f"对比结果已保存至: {comparison_path}")

if __name__ == "__main__":
    # 示例图像路径
    image1 = r'E:\color_images\color_0_1347346.png'  # 替换为第一张图像的路径
    image2 = r'E:\image_vs\frame_episode_5.jpg'  # 替换为第二张图像的路径

    # 保存对比结果的目录
    output_directory = r'D:\BYD\jieya'  # 替换为你希望保存对比结果的目录

    # 调用对比函数
    compare_images(image1, image2, output_directory)
