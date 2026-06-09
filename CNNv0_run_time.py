# 如何统计关键代码的执行时间
#Python中的time库是一个用于处理时间操作的模块，提供了多种函数来获取当前时间、计算时间差等。其中，time.time()函数是该库中一个非常重要的函数，它返回自纪元（1970年1月1日00:00:00 UTC）以来的秒数，通常用于获取当前时间的时间戳。

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# Define the transformation to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset with the specified transformation
mnist_pytorch = datasets.MNIST(root='data', train=True, download=True, transform=transform)

# Create a DataLoader to load the dataset in batches
train_loader_pytorch = torch.utils.data.DataLoader(mnist_pytorch, batch_size=1, shuffle=False)


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
#OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
# Create a figure to display the images
plt.figure(figsize=(15, 3))

start_time = time.time()  # 记录开始时间

# Print the first few images in a row
for i, (image, label) in enumerate(train_loader_pytorch):
    if i < 5:  # Print the first 5 samples
        plt.subplot(1, 5, i + 1)
        plt.imshow(image[0].squeeze(), cmap='gray')
        plt.title(f"Label: {label.item()}")
        plt.axis('off')
    else:
        break  # Exit the loop after printing 5 samples

end_time = time.time()  # 记录结束时间
print("执行时间（预测单张图片）：", end_time - start_time, "秒")

start_time = end_time  # 记录开始时间

plt.tight_layout()
plt.show()
end_time = time.time()  # 记录结束时间
print("执行时间（图像显示）：", end_time - start_time, "秒")
