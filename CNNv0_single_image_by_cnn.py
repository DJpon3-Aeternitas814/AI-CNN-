# 导入所需的库


import torch  # PyTorch深度学习框架，用于构建和训练神经网络
import torch.nn as nn  # PyTorch的神经网络模块，包含各种层和损失函数
from torch.utils.data import Dataset, DataLoader  # 数据集处理和数据加载工具
from torchvision import transforms, models  # 图像预处理工具和预训练模型
import cv2  # OpenCV库，用于图像读取和处理
import os  # 操作系统接口，用于文件路径操作
from PIL import Image
from torchsummary import summary
import torch.nn.functional as F


# 定义一个卷积神经网络（CNN）模型，用于手写数字识别
class CNN(nn.Module):
    def __init__(self):
        """
        初始化CNN网络结构
        包含两个卷积层、池化层、全连接层和Dropout层
        """
        super(CNN, self).__init__()
        
        # 第一个卷积层：输入1通道（灰度图），输出32通道，使用3x3卷积核，边缘填充1像素
        # 作用：提取图像的低级特征（如边缘、线条）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # 第二个卷积层：输入32通道，输出64通道，同样使用3x3卷积核
        # 作用：提取更复杂的特征（如形状、纹理）
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 最大池化层：窗口大小2x2，步长2
        # 作用：降低特征图尺寸，减少计算量，同时保留主要特征
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第一个全连接层：输入尺寸64*7*7（经过两次池化后的特征图大小），输出128个特征
        # 作用：将卷积层提取的特征进行组合，准备分类
        self.fc1 = nn.Linear(64*7*7, 128)
        
        # 第二个全连接层：输入128个特征，输出10个类别（对应数字0-9）
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout层：随机丢弃50%的神经元，防止模型过拟合（仅在训练时生效）
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        定义数据的前向传播过程
        x: 输入图像数据（形状：[batch_size, 1, 28, 28]）
        返回值：10个类别的预测概率
        """
        # 卷积层1 -> ReLU激活 -> 池化（输出形状：[batch_size, 32, 14, 14]）
        x = self.pool(torch.relu(self.conv1(x)))
        
        # 卷积层2 -> ReLU激活 -> 池化（输出形状：[batch_size, 64, 7, 7]）
        x = self.pool(torch.relu(self.conv2(x)))
        
        # 将特征图展平为一维向量（形状：[batch_size, 64*7*7=3136]）
        x = x.view(-1, 64*7*7)
        
        # 全连接层1 -> ReLU激活
        x = torch.relu(self.fc1(x))
        
        # 应用Dropout（训练时随机丢弃部分神经元，测试时不丢弃）
        x = self.dropout(x)
        
        # 全连接层2（输出形状：[batch_size, 10]）
        x = self.fc2(x)
        
        return x

def get_model_inf(model):
    print("\n" + "="*10 + " 运行 get_model_inf() 分析 conv2 " + "="*10)
    
    # 获取 conv2 层
    target_layer = model.conv2
    weight = target_layer.weight
    bias = target_layer.bias

    print(f'层名称: conv2')
    print(f'完整权重形状 [Out, In, H, W]: {list(weight.shape)}') # 应该是 [64, 32, 3, 3]
    print(f'偏置形状: {list(bias.shape)}') # 应该是 [64]

    # 打印前两个卷积核的局部参数（取第一个输入通道的 3x3 矩阵）
    # 使用 .detach().cpu().numpy() 让输出更整洁，不带 tensor 字样
    print(f'\n【神经元 1 (Filter 0) 局部权重】:\n{weight[0, 0].detach().cpu().numpy()}')
    print(f'\n【神经元 2 (Filter 1) 局部权重】:\n{weight[1, 0].detach().cpu().numpy()}')
    
    print(f'\n对应的偏置 (Bias) 前两个值: {bias[:2].detach().cpu().numpy()}')
    print("="*40 + "\n")

def main():
    # 选择设备：优先使用GPU（如果可用），否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型文件路径
    model_path = "./model/mnist_pretrained_cnn.pth"
    # 创建CNN实例，并将模型移动到设备
    model = CNN().to(device)
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path))
    # 模型的加载和保存，参阅 https://pytorch.org/tutorials/beginner/saving_loading_models.html

    get_model_inf(model)
    # 定义图像预处理步骤（顺序非常重要！）
    test_transform = transforms.Compose([
        #transforms.ToPILImage(),     # 将numpy数组转换为PIL图像格式
        transforms.Resize(28),        # 调整图像大小，短边为28像素（保持长宽比）
        transforms.CenterCrop(28),    # 从中心裁剪为28x28像素（确保尺寸一致）
        transforms.Grayscale(),       # 转换为单通道灰度图（MNIST是灰度数据集）
        transforms.ColorJitter(contrast=(3, 3)),  # 增强对比度3倍（使数字更清晰）
        transforms.Lambda(lambda x: transforms.functional.invert(x)),  # 颜色反转（白底黑字 -> 黑底白字）
        transforms.ToTensor(),        # 转换为Tensor格式并归一化到[0.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化（使用MNIST的均值和标准差）
    ])
    image_path = "./data/numbersbyHAND/3.png"
    img_pil=Image.open(image_path).convert("RGB")
    # 对图像预处理
    input_img=test_transform(img_pil)
    print(input_img.shape)
    input_img=input_img.unsqueeze(0).to(device)
    print(input_img.shape)
    # 前向传播，得到所有类别的logit预测分数
    pred_logits=model(input_img)
    print(pred_logits)
    pred_softmax=F.softmax(pred_logits,dim=1)
    #print(pred_softmax.detach().numpy())

    # print('predicted label: ', pred_softmax.detach().numpy().argmax())
    # 将计算结果转回 CPU 并变成普通的 numpy 数组方便取值
    prob_array = pred_softmax.get_device()  # 保险起见先转到cpu
    prob_array = pred_softmax.detach().cpu().numpy()[0] 

    predicted_label = prob_array.argmax()  # 拿到预测标签
    confidence = prob_array[predicted_label]  # 用标签作为索引，精准抓出概率值

    print('\n' + '='*10 + ' 要求 (b) 预测报告 ' + '='*10)
    print(f'所用图片路径: {image_path}')
    print(f'模型预测到的类别标签: {predicted_label}')
    print(f'属于该类别的概率值 (置信度): {confidence * 100:.4f}%')

    # 对应要求 (c)：传入模型和输入的伪数据尺寸 (通道, 高, 宽)
    print("\n" + "="*10 + " 要求 (c) 预训练模型结构摘要 " + "="*10)
    summary(model, (1, 28, 28))


if __name__ == "__main__":
    main()