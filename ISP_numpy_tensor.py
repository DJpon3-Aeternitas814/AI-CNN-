import cv2
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy 


resized = Image.open("./data/img04.jpg")

# resized = img.resize((32,32))
n_resized = numpy.array(resized)
n_resized[:,:,0] = 0
# n_resized[:,:,1] = 0
# n_resized[:,:,2] = 0


img_r = Image.fromarray(n_resized)
plt.subplot(121)
plt.imshow(resized)
plt.subplot(122)
plt.imshow(img_r)
plt.show()
# img_pil = Image.open("./data/ustc2.jpg") # pil 加载的图像默认是 RGB
# print(img_pil)
# img_cv = cv2.imread("./data/ustc2.jpg") #cv2加载的图像默认是 BGR
# print(img_cv.shape)

# plt.subplot(131)
# plt.imshow(img_pil)
# plt.subplot(132)
# plt.imshow(img_cv)

# # 必须进行转换才行
# # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB).transpose(2,0,1)
# img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
# img_cv = img_rgb.transpose(2,0,1)
# plt.subplot(133)
# plt.imshow(img_rgb)
# plt.show()

# #把两个转换到tensor进行测试
# pil_tensor = transforms.functional.pil_to_tensor(img_pil)
# cv_tensor = torch.tensor(img_cv)
# difference = torch.abs(pil_tensor - cv_tensor).sum()
# print("两种方式的输出结果的绝对差之和:", difference.item())
# # print(pil_tensor)
# # print(cv_tensor)
