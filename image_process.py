import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import rotate

PATH_CARBON = r"D:\seadrive\陈显力\我的资料库\调研\碳纳米管薄膜气体温度场\实验数据\20241018\Air 15V×0.12A 0° 41cm 900mlmin"


def read_img_name_from_dir(directory):
    # 指定目录路径

    # 列出目录中的文件名
    file_names = os.listdir(directory)

    # 打印文件名
    for file_name in file_names:
        print(file_name)

    return file_names


# 假设你有一个函数来获取所有BMP文件的路径
def get_image_paths():
    # 这里应该是获取所有BMP文件路径的逻辑
    # 例如：return ['image1.bmp', 'image2.bmp', 'image3.bmp']
    return ["0.1.bmp"]


# 读取BMP文件并将其转换为numpy数组
def read_bmp_to_array(image_path: str) -> np.array:
    with Image.open(image_path) as img:
        return np.array(img)


# plot image
def plot_image(img: np.array) -> None:
    plt.imshow(img)


# Pattern recognition
def reco_specific_pattern_file(directory):

    # 使用正则表达式匹配以数字开头的文件
    pattern = r'[0-9].*'  # 匹配以数字开头的任意字符
    file_path_pattern = os.path.join(directory, pattern)

    # 列出目录中符合模式的文件名
    file_names = glob.glob(file_path_pattern)
    return file_names

    # 打印文件名
    for file_name in file_names:
        # 使用os.path.basename获取完整路径中的文件名
        print(os.path.basename(file_name))

def stack_img(images):
    images_3d = np.stack(images, axis=2)
    return images_3d

# 主函数
def main():
    images = get_images(PATH_CARBON)

    # 使用numpy的stack函数将2D图像堆叠成3D数组
    # axis=2表示沿着第三个维度堆叠
    images_3d = stack_img(images)

    # 旋转3D数组
    rotated_images_3d = rotate(images_3d, angle=0, axes=(0, 1), reshape=True)

    # 显示旋转后的切面
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    rotated_images_3d_1 = rotate(images_3d[288, :, :], angle=90, axes=(0, 1), reshape=True)
    ax[0].imshow(rotated_images_3d_1, cmap='gray')  # 显示原始图像的第一个切面
    ax[0].set_title('Original Slice')
    ax[1].imshow(rotated_images_3d[:, :, 0], cmap='gray')  # 显示旋转后的第一个切面
    ax[1].set_title('Rotated Slice')
    plt.show()

    print(images_3d.shape)  # 输出3D数组的形状，例如：(height, width, num_images)


def get_images(path):
    image_paths = reco_specific_pattern_file(path)  # TODO: 获取所有BMP文件的路径
    images = []  # 用于存储所有图像的数组
    for path in image_paths:
        image_array = read_bmp_to_array(path + r"\\" + path)
        images.append(image_array)  # 将每个图像添加到列表中
    return images


if __name__ == "__main__":
    main()
