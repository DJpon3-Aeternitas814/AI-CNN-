# 使用OpenCV库示例
# 打开Anaconda Prompt，执行以下命令来安装opencv-python（使用中国科技大学的镜像源）：
# pip install opencv-python -i https://pypi.mirrors.ustc.edu.cn/simple
# pip install opencv-contrib-python --user -i https://pypi.mirrors.ustc.edu.cn/simple

import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片的控制

# img = cv2.imread('./data/img01.jpg') #读取本地的图片文件
# # cv2.imshow('image', img) #会打开一个窗口显示图片
# # cv2.waitKey(0)
# img2 = img[:,:,::-1] # transform image to rgb
# plt.subplot(1,2,1)
# plt.imshow(img2) # 显示图片


# def imread_url(url):
#     response = requests.get(url)
#     image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
#     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#     return image

# # 使用示例
# url = "https://www.ustc.edu.cn/images/19/08/05/1bv2xhbf6r/img06.jpg"
# image = imread_url(url) #读取网络上的图片
# cv2.imshow("Image", image) #会打开一个窗口显示图片
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# image2 = image[:,:,::-1] # transform image to rgb
# plt.subplot(1,2,2)
# plt.imshow(image2)


#################################################
# import numpy as np
# import cv2 as cv

# img = cv.imread('./data/img_rzn.jpg')
# assert img is not None, "file could not be read, check with os.path.exists()"

# height, width = img.shape[:2]
# res = cv.resize(img,(64, 64), interpolation = cv.INTER_CUBIC)
# #res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.subplot(1,2,2)
# plt.imshow(res)


###################################
#OpenCV: Changing Colorspaces
#https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('./data/img02.jpg')
# img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)//导入的图实际上是RBG不是RGB
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# plt.subplot(1,4,1)
# plt.imshow(img)
# plt.subplot(1,4,2)
# plt.imshow(img_rgb)
# plt.subplot(1,4,3)
# plt.imshow(gray)
# plt.subplot(1,4,4)
# plt.imshow(gray,cmap='gray')




#################################################
# OpenCV: Plotting Histograms
# https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# img = cv.imread('./data/img02.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# plt.hist(img.ravel(),256,[0,256]); plt.show()



########################################################
# OpenCV: Canny Edge Detection in OpenCV
# https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# img = cv.imread('./data/img02.jpg', cv.IMREAD_GRAYSCALE)
# assert img is not None, "file could not be read, check with os.path.exists()"
# edges = cv.Canny(img,100,200)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

raw_img = cv.imread('./data/img02.jpg')
assert raw_img is not None, "file could not be read, check with os.path.exists()"
img = cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)


cropped = img[0:32,0:32]

img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
edges = cv.Canny(img_gray,100,200)

plt.subplot(1,4,1),plt.imshow(cropped)
plt.subplot(1,4,2),plt.hist(img.ravel(),256,[0,256]),plt.subplot(1, 4, 2).set_box_aspect(1)
plt.subplot(1,4,3),plt.imshow(img_gray,cmap='gray')
plt.subplot(1,4,4),plt.imshow(edges,cmap='gray')



# import cv2 as cv
# import numpy as np

# img = cv.imread('./data/img02.jpg', cv.IMREAD_GRAYSCALE)

# # 1. 用内置函数算出 X 和 Y 方向的导数
# grad_x = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=3)
# grad_y = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=3)

# # 2. 过滤方向：比如我只要“从下往上”变亮的边缘（Y方向导数小于0）
# # 把不符合方向的梯度直接在矩阵里清零
# grad_y[grad_y > 0] = 0 

# # 3. 把过滤后的定向梯度直接喂给 Canny 升级版函数（cv.Canny dx dy 模式）
# # 这样闭着眼睛算出来的 Canny，就自带方向过滤属性了！
# directed_canny = cv.Canny(grad_x, grad_y, 100, 200)
# plt.subplot(1,1,1),plt.imshow(directed_canny,cmap='gray')


plt.show()