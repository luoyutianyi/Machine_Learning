import torchvision
from torchvision import transforms
import os
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.utils.data import DataLoader

BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000

#准备数据集
#分别为均值和标准差
def get_datalodaer(train=True,batch_size = BATCH_SIZE):
    dataset = MNIST(root="./data", train=train, download=False,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


#构建模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.fc1 = nn.Linear(1*28*28,28)
        self.fc2 = nn.Linear(28,10)

    def forward(self,input):
        """

        :param input:[batch_size,1,28,28]
        :return:
        """
        #1、修改形状
        x = input.view([input.size(0),1*28*28])
        #2、进行全连接的操作
        x = self.fc1(x)
        #3、进行激活函数处理
        x = F.relu(x)
        #4、全连接处理
        out = self.fc2(x)
        return F.log_softmax(out,dim=-1)


model =MnistModel()
if os.path.exists("./model/model.pkl"):
    model.load_state_dict(torch.load("./model/model.pkl"))
optimizer = Adam(model.parameters(),lr=0.001)
if os.path.exists("./model/optimizer.pkl"):
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))


def train(epoch):
    """实现训练过程"""
    data_loader = get_datalodaer()
    for idx,(input,target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(input) #调用模型得到预测值
        loss = F.nll_loss(output,target)
        loss.backward() #反向传播
        optimizer.step() #梯度更新
        if idx%10==0:
            print(epoch,idx,loss.item())

        # 模型的保存
        if idx%100 ==0:
            torch.save(model.state_dict(),"./model/model.pkl")
            torch.save(optimizer.state_dict(),"./model/optimizer.pkl")

def test():
    loss_list = []
    acc_list = []
    test_dataloader = get_datalodaer(train=False,batch_size=TEST_BATCH_SIZE)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss)
            #计算真确率
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率，平均损失：",np.mean(acc_list),np.mean(loss_list))


if __name__ == '__main__':
    #for i in range(3): #训练三轮
    #    train(i)
    test()
