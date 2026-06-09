import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms

# Define the transformation to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset with the specified transformation
# mnist_pytorch = datasets.MNIST(root='data', train=True, download=True, transform=transform)
#print(mnist_pytorch)
# Create a DataLoader to load the dataset in batches

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

train_loader_pytorch = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 如果出现一下错误，需要在代码中插入上2行
#OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
#OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
# Create a figure to display the images

import os
# matplotlib.pyplot.figure 函数说明
plt.figure(figsize=(15, 3))

# torchvision.transforms模块中，enumerate()函数用于遍历数据集中的每个图像，并对每个图像应用定义的转换操作。
counter = 0
for i, (image, label) in enumerate(train_loader_pytorch):
    
    current_label = label.item()

    if current_label == 0:
        counter += 1  # 计数器累加
        
        # 创建 1 行 3 列的子图，当前定位到第 counter 个
        plt.subplot(1, 3, counter)
        
        # 降维压平：将 (1, 1, 28, 28) 转换为 matplotlib 认识的 (28, 28) 二维矩阵
        plt.imshow(image.squeeze(), cmap='gray')
        
        # 打印子图标题和关闭坐标轴刻度
        plt.title(f"Image No.{counter} (Label: 0)")
        plt.axis('off')
        
        # 终止条件：集齐 3 张立刻拉闸
        if counter == 3:
            break

plt.tight_layout()
plt.show()
