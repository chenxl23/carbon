import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 被测试的代码
from image_process import read_bmp_to_array, main, plot_image, read_img_name_from_dir, \
    reco_specific_pattern_file, get_images, stack_img  # 确保导入路径正确
from imshow3d import imshow3d
from imshow3d_slice import imshow3d_slice

PATH_CARBON = r"D:\seadrive\陈显力\我的资料库\调研\碳纳米管薄膜气体温度场\实验数据\20241018\Air 15V×0.12A 0° 41cm 900mlmin"


class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        # 创建测试用的BMP文件
        self.test_image_path = 'test.bmp'
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        img.save(self.test_image_path)

    def tearDown(self):
        # 删除测试用的BMP文件
        os.remove(self.test_image_path)

    def test_read_bmp_to_array(self):
        # 测试read_bmp_to_array函数
        image_array = read_bmp_to_array(self.test_image_path)
        self.assertIsInstance(image_array, np.ndarray)
        self.assertEqual(image_array.shape, (100, 100, 3))

    def test_read_from_dir(self):
        y = read_img_name_from_dir(PATH_CARBON)
        x = 1

    def test_plot_img(self):
        list_path = read_img_name_from_dir(PATH_CARBON)
        img = read_bmp_to_array(list_path[0])
        img_partly = img[200:400, 150:300, :]
        # img_partly = img
        plot_image(img_partly)
        plt.show()

    def test_pattern(self):
        reco_specific_pattern_file(PATH_CARBON)

    def test_main(self):
        # 测试main函数
        # 因为main函数依赖于get_image_paths函数，我们需要模拟这个函数的返回值
        def mock_get_image_paths():
            return [self.test_image_path]

        # 将mock函数赋值给模块
        get_image_paths = mock_get_image_paths

        # 运行main函数
        main()

        # 这里我们不直接测试main函数的输出，因为它涉及到打印和文件操作
        # 我们可以通过检查函数是否成功运行以及是否有异常来测试它
        # 如果需要检查输出，可以引入临时文件和路径操作

    def test_3d_plot(self):
        import matplotlib.pyplot as plt
        import numpy as np

        # 创建一个新的figure
        fig = plt.figure()

        # 添加一个3D子图
        ax = fig.add_subplot(111, projection='3d')

        # 生成一些3D数据
        X, Y, Z = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))

        # 你可以在这里替换X, Y, Z为你的3D数组数据
        # 例如：X = np.random.rand(10, 10, 10)

        # 绘制3D表面图
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # 设置标签
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # 显示图像
        plt.show()

    def test_plot_3d_from_web(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set(xlabel="x", ylabel="y", zlabel="z")
        # 设置 50:50:500 的比例
        ax.set_box_aspect([1, 1, 10])  # X:Y:Z 比例为 50:50:500

        nx, ny, nz = 50, 50, 500
        data_xy = np.arange(ny * nx).reshape(ny, nx) + 15 * np.random.random((ny, nx))
        data_yz = np.arange(nz * ny).reshape(nz, ny) + 10 * np.random.random((nz, ny))
        data_zx = np.arange(nx * nz).reshape(nx, nz) + 8 * np.random.random((nx, nz))

        imshow3d(ax, data_xy)
        imshow3d(ax, data_yz, value_direction='x', cmap='gray')
        imshow3d(ax, data_zx, value_direction='y', pos=ny, cmap='plasma')

        plt.show()

    def test_plot_3d_carbon(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set(xlabel="x", ylabel="y", zlabel="z")

        images = get_images(PATH_CARBON)

        # 使用numpy的stack函数将2D图像堆叠成3D数组
        # axis=2表示沿着第三个维度堆叠
        images_3d = stack_img(images)

        images_3d = images_3d[280:320, 175:510, :, 1]

        nx, ny, nz = images_3d.shape  # 512, 712, 51
        # ax.set_box_aspect([zz, xx, yy]) # 512, 712, 51
        ax.set_box_aspect([nx, ny, nz])  # X:Y:Z 比例为 50:50:500  # X:Y:Z 比例为 50:50:500

        data_xy = images_3d[:, :, 1]
        # data_xy = rotate(data_xy, angle=90, axes=(0, 1), reshape=False)
        data_xy = data_xy.T
        data_yz = images_3d[10, :, :].T
        # data_yz = rotate(data_yz, angle=90, axes=(0, 1), reshape=False)
        # data_xy = rotate(data_xy, angle=90, axes=(0, 1), reshape=False)
        data_zx = images_3d[:, 289, :]

        cmaps = ['hot', 'inferno', 'plasma', 'magma', 'coolwarm']

        my_cmap = cmaps[1]

        imshow3d(ax, data_xy, cmap=my_cmap)
        imshow3d(ax, data_yz, value_direction='x', cmap=my_cmap)
        imshow3d(ax, data_zx, value_direction='y', pos=ny, cmap=my_cmap)

        plt.show()

    def test_plot_3d_slide(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set(xlabel="x", ylabel="y", zlabel="z")

        images = get_images(PATH_CARBON)

        # 使用numpy的stack函数将2D图像堆叠成3D数组
        # axis=2表示沿着第三个维度堆叠
        images_3d = stack_img(images)

        images_3d = images_3d[280:320, 175:510, :, 1]

        nx, ny, nz = images_3d.shape  # 512, 712, 51
        # ax.set_box_aspect([zz, xx, yy]) # 512, 712, 51
        ax.set_box_aspect([nx, ny, nz])  # X:Y:Z 比例为 50:50:500  # X:Y:Z 比例为 50:50:500

        data_xy = images_3d[:, :, 1]

        cmaps = ['hot', 'inferno', 'plasma', 'magma', 'coolwarm']

        my_cmap = cmaps[1]

        n_planes_k = 6
        for i in range(n_planes_k):
            z = i * 10  # Calculate z position of the plane
            imshow3d_slice(ax, data_xy, myzi=z, cmap=my_cmap)

        plt.show()

    def test_plot_3d_slide_by_imshow3d_py(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set(xlabel="x", ylabel="y", zlabel="z")

        images = get_images(PATH_CARBON)

        # 使用numpy的stack函数将2D图像堆叠成3D数组
        # axis=2表示沿着第三个维度堆叠
        images_3d = stack_img(images)

        slide_w = 10
        list_width = np.array([280, 320])
        list_width -= slide_w
        images_3d = images_3d[list_width[0]:list_width[1], 175:510, :, 1]

        nx, ny, nz = images_3d.shape  # 512, 712, 51
        # ax.set_box_aspect([zz, xx, yy]) # 512, 712, 51
        ax.set_box_aspect([nx, ny, nz])  # X:Y:Z 比例为 50:50:500  # X:Y:Z 比例为 50:50:500

        data_xy = images_3d[:, :, 1]
        data_xy = data_xy.T

        cmaps = ['hot', 'inferno', 'plasma', 'magma', 'coolwarm']

        my_cmap = cmaps[1]

        n_planes_k = 16
        for i in range(n_planes_k):
            z = i * 5  # Calculate z position of the plane
            imshow3d(ax, data_xy, pos=z, cmap=my_cmap)

        plt.show()


if __name__ == '__main__':
    unittest.main()

    # test = TestImageProcessing()
    # test.setUp()
