# PIL图像与Numpy数组的相互转换
from PIL import Image 
import numpy 
import matplotlib.pyplot as plt

# im = Image.open('./data/ustc2.jpg').resize((32, 32)) 
# n = numpy.array(im)
# #n = numpy.asarray(im)
# print(n.shape)
# n_r = numpy.copy(n)
# n_g = numpy.copy(n)
# n_b = numpy.copy(n)
# n_r[:, :, 0] = 0 
# n_g[:, :, 1] = 0 
# n_b[:, :, 2] = 0 
# image_r = Image.fromarray(n_r)
# image_g = Image.fromarray(n_g)
# image_b = Image.fromarray(n_b)
# plt.subplot(131)
# plt.imshow(image_r)
# plt.subplot(132)
# plt.imshow(image_g)
# plt.subplot(133)
# plt.imshow(image_b)
# plt.show()
# new_image = Image.fromarray(n)
# new_image.save("./ustc2_new.jpg") # 保存新图像

################################################
# PIL图像转为Tensor，Numpy数组转为Tensor
import cv2
import torch
from PIL import Image
from torchvision import transforms

# img_pil = Image.open("./ustc2_new.jpg") # pil 加载的图像默认是 RGB
# print(img_pil)
# img_cv = cv2.imread("./ustc2_new.jpg") #cv2加载的图像默认是 BGR
# print(img_cv.shape)

# plt.subplot(131)
# plt.imshow(img_pil)
# plt.subplot(132)
# plt.imshow(img_cv)

# # 必须进行转换才行
# img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).transpose(2,0,1)
# plt.subplot(133)
# plt.imshow(img_cv)

img = Image.open("./data/img04.jpg")
resized = img.resize((4,4))
resized.save("./data/img04_4x4.jpg")
img_4x4_pil = Image.open("./data/img04_4x4.jpg")
raw_img_4x4_cv = cv2.imread("./data/img04_4x4.jpg")
img_4x4_cv = cv2.cvtColor(raw_img_4x4_cv, cv2.COLOR_BGR2RGB)


plt.subplot(121),plt.imshow(img_4x4_pil)
plt.subplot(122),plt.imshow(img_4x4_cv)
plt.show()

pil_tensor = transforms.functional.pil_to_tensor(img_4x4_pil)
cv_tensor = torch.tensor(img_4x4_cv.transpose(2,0,1))

import torch

# 🚀 核心安全步：在计算前，统一转换成 float32 浮点型，防止均值计算报错
pil_tensor_f = pil_tensor.float()
cv_tensor_f = cv_tensor.float()

print("================ PIL Tensor 统计结果 ================")
print(f"最大值 (Max): {torch.max(pil_tensor_f).item():.2f}")
print(f"最小值 (Min): {torch.min(pil_tensor_f).item():.2f}")
print(f"平均值 (Mean): {torch.mean(pil_tensor_f).item():.2f}")

print("\n================ OpenCV Tensor 统计结果 ================")
print(f"最大值 (Max): {torch.max(cv_tensor_f).item():.2f}")
print(f"最小值 (Min): {torch.min(cv_tensor_f).item():.2f}")
print(f"平均值 (Mean): {torch.mean(cv_tensor_f).item():.2f}")

print("\n================ 两者直接对比（两者的差值统计） ================")
diff_tensor = pil_tensor_f - cv_tensor_f
print(f"最大偏差 (Max Diff): {torch.max(torch.abs(diff_tensor)).item():.2f}")
print(f"平均偏差 (Mean Diff): {torch.mean(torch.abs(diff_tensor)).item():.2f}")
#把两个转换到tensor进行测试
# pil_tensor = transforms.functional.pil_to_tensor(img_pil)
# cv_tensor = torch.tensor(img_cv)
# difference = torch.abs(pil_tensor - cv_tensor).sum()
# print("两种方式的输出结果的绝对差之和:", difference.item())
#print(pil_tensor)
#print(cv_tensor)
