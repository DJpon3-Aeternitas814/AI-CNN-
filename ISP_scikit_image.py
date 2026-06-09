# scikit-image 是一个基于 Python 的图像处理库，它提供了广泛的算法和工具，用于图像处理、分析和可视化。scikit-image 建立在 NumPy 数组上，这使得它非常适合进行科学计算和图像分析。
# pip install scikit-image
# import numpy as np
# import skimage
# import matplotlib.pyplot as plt
# from skimage import data
from skimage import io, color, filters, exposure, morphology, measure, segmentation, restoration, feature, transform
# from skimage.color import label2rgb
# # img = io.imread('./data/img01.jpg')
# img = data.astronaut()

# # print(type(img)) #显示类型
# # print('Image shape: ',img.shape) #显示尺寸
# # print('Image width: ', img.shape[0]) #图片宽度
# # print('Image height: ', img.shape[1]) #图片高度
# # print('Image channel num: ', img.shape[2]) #图片通道数
# # print('Pixel numbers: ', img.size) #显示总像素个数
# # print('Pixel, max value', img.max()) #最大像素值
# # print('Pixel, min value', img.min()) #最小像素值
# # print('Pixel, average value', img.mean()) #像素平均值

# img_gray = color.rgb2gray(img)
# resize = transform.resize(img, (32, 32))  # 缩放图像
# rotated = transform.rotate(img, 45)  # 旋转45度
# labels = measure.label(img_gray > 0.5, connectivity=2)
# binary = (labels > 0).astype(np.uint8) * 255

# plt.subplot(141)
# plt.imshow(img)

# plt.subplot(142)
# plt.imshow(resize)

# plt.subplot(143)
# plt.imshow(rotated)

# plt.subplot(144)
# plt.imshow(binary, cmap='gray')


########################################
# scikit-image 是一个基于 Python 的图像处理库，它提供了广泛的算法和工具，用于图像处理、分析和可视化。scikit-image 建立在 NumPy 数组上，这使得它非常适合进行科学计算和图像分析。
import numpy as np
from skimage import data
import matplotlib.pyplot as plt

img = io.imread('./data/img03.jpg')

# camera = data.astronaut() #内置数据集
# camera[:10] = 0
# mask = camera < 87
# camera[mask] = 255
# inds_x = np.arange(len(camera))
# inds_y = (4 * inds_x) % len(camera)
# camera[inds_x, inds_y] = 0 #一系列点涂黑 这些点是直线

# l_x, l_y = camera.shape[0], camera.shape[1]
# X, Y = np.ogrid[:l_x, :l_y]
# outer_disk_mask = (X - l_x / 2) ** 2 + (Y - l_y / 2) ** 2 > (l_x / 2) ** 2
# camera[outer_disk_mask] = 0

l_x, l_y = img.shape[0], img.shape[1]
X, Y = np.ogrid[:l_x, :l_y]
mask = (X - l_x / 2) ** 2 + (Y - l_y / 2) ** 2 > (l_x / 2) ** 2
img[mask]=0

plt.figure(figsize=(4, 4))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

