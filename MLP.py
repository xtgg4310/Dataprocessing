import torch
import torch.nn.functional as F
import numpy as np
import torch.utils.data

n_epochs = 10



class MLP(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self):
        super(MLP, self).__init__()  #
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(784, 512)  # 第一个隐含层
        self.fc2 = torch.nn.Linear(512, 128)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(128, 10)  # 输出层

    def forward(self, din):
        # 前向传播， 输入值：din, 返回值 dout
        din = din.view(-1, 28 * 28)  # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))  # 使用 relu 激活函数
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout


model = MLP()


def train():
    # 定义损失函数和优化器
    lossfunc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    # 开始训练
    for epoch in range(n_epochs):
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 得到预测值
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 训练集中不需要反向传播
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    return 100.0 * correct / total
