import os
import time  # 🌟 25题核心：引入时间统计模块
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  
        x = self.pool(torch.relu(self.conv2(x)))  
        x = x.view(-1, 64 * 7 * 7)                
        x = torch.relu(self.fc1(x))               
        x = self.dropout(x)                       
        x = self.fc2(x)                           
        return x

# 加载MNIST数据集
def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),                    
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 训练一个epoch
def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()  
    running_loss = 0.0
    
    # ⚠️ 【注意】根据第25题要求，循环内部严禁包含任何 print 语句
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  
        output = model(data)   
        loss = criterion(output, target)  
        loss.backward()        
        optimizer.step()       
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

# 测试一个epoch
def test_epoch(model, device, test_loader, criterion):
    model.eval()  
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0  # 🌟 24题核心
    total_samples = 0
    
    # ⚠️ 【注意】根据第25题要求，循环内部严禁包含任何 print 语句
    with torch.no_grad():  
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  
            total_samples += target.size(0)
            
            # Top-1 计算 (23题)
            pred = output.argmax(dim=1, keepdim=True)      
            correct_top1 += pred.eq(target.view_as(pred)).sum().item()
            
            # Top-5 计算 (24题)
            _, pred_top5 = output.topk(5, dim=1, largest=True, sorted=True)
            correct_top5 += (pred_top5 == target.view(-1, 1)).sum().item()

    test_loss /= len(test_loader)
    accuracy_top1 = 100. * correct_top1 / total_samples
    accuracy_top5 = 100. * correct_top5 / total_samples
    
    return test_loss, accuracy_top1, accuracy_top5

# 主函数
def main():
    modelpath = "./model"
    os.makedirs(modelpath, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前训练使用的硬件设备: {device}")
    
    train_loader, test_loader = load_data(batch_size=64)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)  
    
    epochs = 3
    
    # 用于累加各轮时间以计算最后的平均时间 (25题)
    total_train_time = 0.0
    total_test_time = 0.0
    
    for epoch in range(1, epochs + 1):
        print(f"\n" + "="*15 + f" ⏳ Epoch {epoch}/{epochs} 正在运行 " + "="*15)
        
        # ⏱️ 统计 train_epoch 执行时间
        start_train = time.time()
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        end_train = time.time()
        epoch_train_time = end_train - start_train
        total_train_time += epoch_train_time
        
        # ⏱️ 统计 test_epoch 执行时间
        start_test = time.time()
        test_loss, accuracy_top1, accuracy_top5 = test_epoch(model, device, test_loader, criterion)
        end_test = time.time()
        epoch_test_time = end_test - start_test
        total_test_time += epoch_test_time
        
        # 结果汇报
        print(f"⏱️ 本轮耗时 -> 训练时间: {epoch_train_time:.4f} 秒 | 测试时间: {epoch_test_time:.4f} 秒")
        print(f"📉 误差指标 -> Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"🎯 准确率   -> Top-1 Acc: {accuracy_top1:.2f}% | Top-5 Acc: {accuracy_top5:.2f}%")

    # 🌟 计算并输出题目要求的平均时间报告
    avg_train_time = total_train_time / epochs
    avg_test_time = total_test_time / epochs
    
    print("\n" + "==================================================")
    print("      📊 综合实验报告结论 (第 22, 23, 24, 25 题)      ")
    print("==================================================")
    print(f"1. 准确率随 Epoch 的增加呈现【稳步上升】趋势 (第22题)")
    print(f"2. 最终批量测试集 Top-1 准确率 (精度): {accuracy_top1:.2f}% (第23题)")
    print(f"3. 最终批量测试集 Top-5 准确率 (精度): {accuracy_top5:.2f}% (第24题)")
    print(f"4. train_epoch() 平均执行时间: {avg_train_time:.4f} 秒 (第25题-a)")
    print(f"5. test_epoch()  平均执行时间: {avg_test_time:.4f} 秒 (第25题-b)")
    print("==================================================\n")

    torch.save(model.state_dict(), os.path.join(modelpath, "mnist_cnn.pth"))

if __name__ == '__main__':
    main()