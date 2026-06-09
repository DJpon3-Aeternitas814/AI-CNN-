import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 工具

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    """
    设置随机种子，确保每次运行代码时生成的随机数相同。
    这对于实验的可重复性非常重要。
    """
    torch.manual_seed(seed)

# 数据预处理与加载
def load_data(batch_size=64, num_workers=2):
    """
    加载 CIFAR-10 数据集，并进行预处理。
    - ToTensor(): 将图像数据转换为 PyTorch 张量。
    - Normalize(): 对图像数据进行标准化（减去均值，除以标准差）。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL 图像或 NumPy 数组转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 标准化
    ])

    # 下载训练集trainset和测试集testset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "data")
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    

    # 下载训练集trainset和测试集testset
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # TODO 此处需要你补全

    def forward(self, x):
        # TODO 此处需要你补全
        
        return x


# 训练模型
def train_model(net, trainloader, criterion, optimizer, device, writer, epochs=10):
    """
    训练模型：
    - 使用交叉熵损失函数和SGD优化器。
    - 每个epoch结束后记录平均损失到 TensorBoard。
    """
    print("Starting Training...")
    for epoch in range(epochs):
        running_loss = 0.0
        net.train()  # 切换到训练模式
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到GPU/CPU

            optimizer.zero_grad()  # 清空梯度
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.3f}")
        writer.add_scalar('Training Loss', avg_loss, epoch)  # 将损失写入 TensorBoard

    print('Finished Training')

# 测试模型
def test_model(net, testloader, device, writer, epoch):
    """
    测试模型：
    - 在测试集上评估模型性能。
    - 将测试准确率写入 TensorBoard。
    """
    correct = 0
    total = 0
    net.eval()  # 切换到评估模式
    with torch.no_grad():  # 不计算梯度
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    writer.add_scalar('Test Accuracy', accuracy, epoch)  # 将准确率写入 TensorBoard

# 主函数
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    set_seed()  # 设置随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU可用

    # 加载数据
    trainloader, testloader = load_data()

    # 初始化模型、损失函数和优化器
    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 使用SGD优化器

    # 创建 TensorBoard 的日志记录器
    writer_dir = os.path.join(script_dir, "runs/cifar10_experiment")
    writer = SummaryWriter(writer_dir)  # 日志保存在 runs/cifar10_experiment 文件夹中

    # 训练模型
    train_model(net, trainloader, criterion, optimizer, device, writer, epochs=19)

    # 测试模型
    test_model(net, testloader, device, writer, epoch=19)  # 测试最后一个epoch的结果

    writer.close()  # 关闭 TensorBoard 日志记录器

    print("要使用TensorBoard查看训练过程，可以在命令行运行： tensorboard --logdir=runs")

if __name__ == '__main__':
    main()