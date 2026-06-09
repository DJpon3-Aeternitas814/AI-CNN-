from PIL import Image
im = Image.open("./data/img_rzn.jpg")
im2 = Image.open("./data/img01.jpg")
# print(im.format, im.size, im.mode) #显示图像格式、尺寸彩色模式
# im.show() #使用操作系统默认的图片浏览器显示图像

# gray = im.convert("L")
# gray.show()

import matplotlib.pyplot as plt # plt 用于显示图片的控制
from PIL import ImageEnhance
enh = ImageEnhance.Contrast(im)
# enh.enhance(1.3).show("30% more contrast") #提高图像的对比度



###########################################
# Using Matplotlib — Matplotlib 3.9.2 documentation https://matplotlib.org/stable/users/index.html
plt.subplot(2,3,1) #1行5列的绘图区第1列
plt.imshow(im)  
box = (0, 0, 32, 32)
region = im.crop(box) #裁剪图像的局部区域
plt.subplot(2,3,2)
plt.imshow(region)

gray = region.convert("L")
plt.subplot(2,3,3)
plt.imshow(gray,cmap="gray")


region2 = im2.crop(box)
plt.subplot(2,3,4)
plt.imshow(region2)

gray2 = im2.convert("L")
plt.subplot(2,3,5)
plt.imshow(gray2,cmap="gray")

rotate = im2.rotate(90) # degrees counter-clockwise
plt.subplot(2,3,6)
plt.imshow(rotate)



# resize = im.resize((32, 32))
# plt.subplot(2,4,3)
# plt.imshow(resize)

# rotate = im.rotate(45) # degrees counter-clockwise
# plt.subplot(2,4,4)
# plt.imshow(rotate)

# r, g, b = im.split()
# plt.subplot(2,4,5)
# plt.imshow(r)
# plt.subplot(2,4,6)
# plt.imshow(g)
# plt.subplot(2,4,7)
# plt.imshow(b)




# 🌟 必须在最末尾加上这两行！
plt.tight_layout()  # 让格子排版更整齐，不挤压
plt.show()          # 正式激活并弹出画布窗口