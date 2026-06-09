import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import torch
import torch.nn as nn

# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        """
        初始化网络结构：
        - 卷积层1：输入通道数为1（灰度图像），输出通道数为32，卷积核大小为3x3，填充为1。
        - 卷积层2：输入通道数为32，输出通道数为64，卷积核大小为3x3，填充为1。
        - 池化层：最大池化，窗口大小为2x2，步幅为2。
        - 全连接层1：输入特征数为64*7*7（MNIST图像经过两次池化后尺寸变为7x7），输出特征数为128。
        - 全连接层2：输入特征数为128，输出特征数为10（对应MNIST数据集的10个类别）。
        - Dropout层：随机丢弃50%的神经元，防止过拟合。
        """
        super(CNN, self).__init__()
        
        # 第一层卷积：输入通道数为1（灰度图像），输出通道数为32，卷积核大小为3x3，填充为1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # 第二层卷积：输入通道数为32，输出通道数为64，卷积核大小为3x3，填充为1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # 最大池化层：窗口大小为2x2，步幅为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第一层全连接：输入特征数为64*7*7（MNIST图像经过两次池化后尺寸变为7x7），输出特征数为128
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        
        # 第二层全连接：输入特征数为128，输出特征数为10（对应MNIST数据集的10个类别）
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        
        # Dropout层：随机丢弃50%的神经元，防止过拟合
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        定义前向传播过程：
        - 输入经过卷积层1和ReLU激活函数后，进行最大池化。
        - 再经过卷积层2和ReLU激活函数后，再次进行最大池化。
        - 将特征展平并通过两个全连接层，最后使用Dropout防止过拟合。
        """
        # print("数据初始维度", x.shape)

        # 第一层卷积 -> ReLU -> 最大池化
        x = self.pool(torch.relu(self.conv1(x)))  # 输出尺寸变为 32x14x14
        # print("第一层卷积 -> ReLU -> 最大池化: ", x.shape)
        
        # 第二层卷积 -> ReLU -> 最大池化
        x = self.pool(torch.relu(self.conv2(x)))  # 输出尺寸变为 64x7x7
        
        # 展平成一维向量
        x = x.view(-1, 64 * 7 * 7)                # 展平为 [batch_size, 64*7*7]
        
        # 第一层全连接 -> ReLU
        x = torch.relu(self.fc1(x))               # 输出尺寸为 [batch_size, 128]
        
        # Dropout层
        x = self.dropout(x)                       # 随机丢弃50%的神经元
        
        # 第二层全连接
        x = self.fc2(x)                           # 输出尺寸为 [batch_size, 10]
        
        return x

# 加载MNIST数据集
def load_data(batch_size=64):
    """
    加载MNIST数据集，并进行预处理：
    - ToTensor(): 将PIL图像或NumPy数组转换为张量。
    - Normalize(): 对图像数据进行标准化（减去均值，除以标准差）。
    - 返回训练集和测试集的数据加载器。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                    # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST数据集的统计值）
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 训练一个epoch
def train_epoch(model, device, train_loader, optimizer, criterion, writer, epoch):
    """
    训练模型一个epoch：
    - 使用交叉熵损失函数和Adam优化器。
    - 每10个batch记录一次训练损失到TensorBoard。
    - 返回该epoch的平均训练损失。
    """
    model.train()  # 切换到训练模式
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 将数据移动到GPU/CPU

        optimizer.zero_grad()  # 清空梯度
        output = model(data)   # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        
        running_loss += loss.item()
        global_step = epoch * len(train_loader) + batch_idx
        
        if batch_idx % 10 == 0:  # 每10个batch记录一次损失
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)  # 记录整个epoch的平均损失
    return avg_loss

# 测试一个epoch
def test_epoch(model, device, test_loader, criterion, writer, epoch):
    """
    测试模型一个epoch：
    - 在测试集上评估模型性能。
    - 计算测试损失和准确率，并将结果写入TensorBoard。
    - 返回测试损失和准确率。
    """
    model.eval()  # 切换到评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 累加损失
            pred = output.argmax(dim=1, keepdim=True)      # 获取预测类别
            correct += pred.eq(target.view_as(pred)).sum().item()  # 统计正确预测的数量

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)  # 计算准确率
    
    writer.add_scalar('Loss/test', test_loss, epoch)      # 记录测试损失
    writer.add_scalar('Accuracy/test', accuracy, epoch)   # 记录测试准确率
    
    return test_loss, accuracy

# 主函数
def main():
    """
    主程序入口：
    - 初始化TensorBoard日志记录器。
    - 加载MNIST数据集。
    - 初始化模型、优化器和损失函数。
    - 训练模型并记录训练和测试过程中的指标。
    - 提示用户如何使用TensorBoard查看训练过程。
    """

    modelpath = "./model"
    if not os.path.exists(modelpath):
        print("模型目录不存在，将自动新建！")
        os.makedirs(modelpath, exist_ok=True)

    # 初始化TensorBoard日志记录器
    log_dir = f"runs/mnist_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # 日志目录名包含时间戳
    writer = SummaryWriter(log_dir)
    
    # 初始化设备（优先使用GPU，否则使用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    train_loader, test_loader = load_data(batch_size=64)
    
    # 初始化模型
    model = CNN().to(device)
    
    # 记录模型结构和数据样本
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    writer.add_graph(model, images.to(device))  # 记录模型结构
    writer.add_image('MNIST Sample', torchvision.utils.make_grid(images[:8]), 0)  # 记录样本图像
    
    # 初始化训练组件
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    
    # 训练参数
    epochs = 10#10  # 总共训练10个epoch
    
    # 开始训练和测试
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        start_time = time.time()  # 记录开始时间
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion, writer, epoch)
        end_time = time.time()  # 记录结束时间
        print("执行时间（epoch）：", end_time - start_time, "秒")
        test_loss, accuracy = test_epoch(model, device, test_loader, criterion, writer, epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    modelname = "mnist_cnn.pth"
    print(f"模型保存于{modelpath}， 名称{modelname}")
    torch.save(model.state_dict(), os.path.join(modelpath, modelname))
    #torch.save(model, './model/mnist_cnn_full.pth')
    writer.close()  # 关闭TensorBoard日志记录器

    print("要使用TensorBoard查看训练过程，可以在命令行运行： tensorboard --logdir=runs")

if __name__ == '__main__':
    # 进入环境后，输入 python CNNv1.py 运行程序。 
    main()