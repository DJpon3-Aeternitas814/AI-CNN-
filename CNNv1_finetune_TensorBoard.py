import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import cv2  # OpenCV库，用于图像读取和处理
import os  # 操作系统接口，用于文件路径操作
import time
import torch
import torch.nn as nn
import numpy as np

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

# 自定义数据集类，用于加载和处理图像数据
class CustomImageDataset():
    def __init__(self, img_dir, transform=None):
        """
        img_dir: 图像文件夹路径
        transform: 图像预处理流程
        """
        self.img_dir = img_dir
        # 获取所有.png/.jpg/.jpeg结尾的文件名
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg', 'bmp', 'gif'))]
        # self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print (self.img_names)
        self.transform = transform
        # 验证文件名格式是否正确（文件名应形如"3_abc.jpg"，以数字开头）
        self._validate_filenames()
    
    def _validate_filenames(self):
        """检查文件名是否符合 数字_其他内容.扩展名 的格式"""
        for name in self.img_names:
            parts = name.split('.')
            # parts = name.split('_')
            print(parts)
            if len(parts) < 1 or not parts[0].isdigit():
                raise ValueError(f"文件名'{name}'格式错误！正确格式应为：数字_其他内容.扩展名")

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.img_names)
    
    def __getitem__(self, idx):
        """加载并返回索引为idx的样本（图像和标签）"""
        # 1. 规范化路径拼接，使用 os.path.join 并用 os.path.normpath 统一 Windows 的斜杠
        img_path = os.path.normpath(os.path.join(self.img_dir, self.img_names[idx]))
        
        # 2. 🌟 核心修正：规避 OpenCV 中文路径 Bug
        # 不直接使用 cv2.imread(img_path)
        try:
            # 先用 numpy 把图片读入为纯二进制字节流
            img_array = np.fromfile(img_path, dtype=np.uint8)
            # 再通过 OpenCV 从内存字节流中解码出图像
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"二进制流读取失败：{img_path}，错误原因: {e}")

        if image is None:
            raise ValueError(f"无法读取图像（解码失败，请检查文件是否损坏）：{img_path}")
        
        # 3. 将 BGR 格式转换为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 4. 从文件名中提取标签
        label = int(self.img_names[idx].split('.')[0])
        
        # 5. 应用预处理流程
        if self.transform:
            image = self.transform(image)
            
        return image, label

def deal_data(data_dir, batch_size=1):
    """
    构建数据预处理流程并创建数据加载器
    data_dir: 数据文件夹路径
    batch_size: 每次加载的样本数量
    """
    # 定义图像预处理步骤（顺序非常重要！）
    transform = transforms.Compose([
        transforms.ToPILImage(),     # 将numpy数组转换为PIL图像格式
        transforms.Resize(28),        # 调整图像大小，短边为28像素（保持长宽比）
        transforms.CenterCrop(28),    # 从中心裁剪为28x28像素（确保尺寸一致）
        transforms.Grayscale(),       # 转换为单通道灰度图（MNIST是灰度数据集）
        transforms.ColorJitter(contrast=(3, 3)),  # 增强对比度3倍（使数字更清晰）
        transforms.Lambda(lambda x: transforms.functional.invert(x)),  # 颜色反转（白底黑字 -> 黑底白字）
        transforms.ToTensor(),        # 转换为Tensor格式并归一化到[0.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化（使用MNIST的均值和标准差）
    ])
    
    # 创建数据集实例
    dataset = CustomImageDataset(
        img_dir=data_dir,
        transform=transform
    )
    
    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,       # 测试时不需要打乱数据顺序
        num_workers=0        # 使用2个子进程加载数据（加快数据读取）
    )
    
    return loader
    
# 加载MNIST数据集
def load_data(batch_size=64):

    # 微调时使用的训练数据（自己手写的图片）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    finetune_loader = deal_data(os.path.join(script_dir, "data/numbersbyHAND"))
    """
    加载MNIST数据集，并进行预处理：
    - ToTensor(): 将PIL图像或NumPy数组转换为张量。
    - Normalize(): 对图像数据进行标准化（减去均值，除以标准差）。
    - 返回训练集和测试集的数据加载器。
    """
    #微调过程使用的测试数据（仍然取自MNIST，可在训练过程汇总观察损失函数变化情况）
    transform = transforms.Compose([
        transforms.ToTensor(),                    # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化（MNIST数据集的统计值）
    ])

    test_dataset = torchvision.datasets.MNIST(
        root=os.path.join(script_dir, "data"), train=False, download=False, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return finetune_loader, test_loader

# 训练一个epoch
def train_epoch(model, device, train_loader, optimizer, criterion, writer, epoch):
    """
    训练模型一个epoch：
    - 使用交叉熵损失函数和Adam优化器。
    - 每10个batch记录一次训练损失到TensorBoard。
    - 返回该epoch的平均训练损失。
    """
    model.train()  # 切换到训练模式
    model.to(device)

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

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
# 主函数
def main():
    # 获取当前脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 创建带有时间戳的目录名
    dir_name = f"mnist_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # 组合成完整路径
    log_dir = os.path.join(script_dir, "runs", dir_name)
    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    print("create dir: ", log_dir)

    writer = SummaryWriter(log_dir)
    
    # 初始化设备（优先使用GPU，否则使用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    train_loader, test_loader = load_data(batch_size=64)
    
    # 初始化模型
    model = CNN().to(device)

    # 加载预训练权重
    model.load_state_dict(torch.load(os.path.join(script_dir, "model/mnist_cnn.pth")))
    #。。。。。。。。。。。。。。。。。。。。
    
    # 记录模型结构和数据样本
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    writer.add_graph(model, images.to(device))  # 记录模型结构
    writer.add_image('MNIST Sample', torchvision.utils.make_grid(images[:8]), 0)  # 记录样本图像

    # 冻结参数的梯度
    feature_extract = True
    set_parameter_requires_grad(model, feature_extract)
    # 修改模型
    num_ftrs = model.fc2.in_features
    model.fc2 = nn.Linear(in_features=128, out_features=10, bias=True)
    
    # 初始化训练组件
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    
    # 训练参数
    epochs = 10 #10  # 总共训练10个epoch
    
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

    # 模型文件路径
    modelpath = os.path.join(script_dir, "model")
    modelname = "mnist_finetune_cnn.pth"
    print(f"模型保存于{modelpath}， 名称{modelname}")
    torch.save(model.state_dict(), os.path.join(modelpath, modelname))
    #torch.save(model, './model/mnist_cnn_full.pth')
    writer.close()  # 关闭TensorBoard日志记录器

    print("要使用TensorBoard查看训练过程，可以在命令行运行： tensorboard --logdir=runs")

if __name__ == '__main__':
    # 进入环境后，输入 python CNNv1.py 运行程序。 
    main()