# torchvision.transforms
# https://pytorch.org/vision/0.11/transforms.html
# Most transformations accept both PIL images and tensor images, although some transformations are PIL-only and some are tensor-only. The Conversion Transforms may be used to convert to and from PIL images.

from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from PIL import Image
img = Image.open("./data/img05.jpg")

# 单一图像处理操作
# num_output_channels =1 灰度图，  =3 ：r g b
# img1 = transforms.Grayscale(num_output_channels=1)(img)

# 多个图像处理操作组合在一起
transform = transforms.Compose([
        transforms.Resize((32,32)),        # 调整图像大小，短边为28像素（保持长宽比）
        # transforms.CenterCrop(28),    # 从中心裁剪为28x28像素（确保尺寸一致）
        transforms.Grayscale(),       # 转换为单通道灰度图（MNIST是灰度数据集）
        # transforms.ColorJitter(contrast=(3, 3)),  # 增强对比度3倍（使数字更清晰）
        transforms.RandomInvert(p=1)
])
img3 = transform(img)

plt.subplot(141)
plt.imshow(img, cmap = 'gray')
plt.subplot(142)
img1 = transforms.Resize((32,32))(img)
plt.imshow(img1, cmap = 'gray')
plt.subplot(143)
img2 = transforms.Grayscale(num_output_channels=1)(img1)
plt.imshow(img2,cmap='gray')
plt.subplot(144)
plt.imshow(img3, cmap = 'gray')

plt.show()