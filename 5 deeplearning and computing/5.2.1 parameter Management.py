import torch
from torch import nn

# 一个具有单隐藏层的多层感知机
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

print(net[2].state_dict())  #访问第三层的权重及偏置参数

print(type(net[2].bias))    #访问目标参数
print(net[2].bias)
print(net[2].bias.data)

print(net[2].weight.grad == None ) #参数梯度处于初始状态

# ---一次性访问所有参数---
print("---一次性访问所有参数---")
print(*[(name, param.shape) for name, param in net[0].named_parameters()]) # named_parameters()会返回一个迭代器对象
print(*[(name, param.shape) for name, param in net.named_parameters()])

# ---另一种访问网络参数的方式---
print("---另一种访问网络参数的方式---")
print(net.state_dict()['2.bias'].data)
print("\n")



print("***从嵌套块收集参数***")
# 定义两个生成块的函数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())  #生成名称为block i，内容为block1()的子模块
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print("---网络是如何工作的---")
print(rgnet)
print("---访问第一个主要的块中、第二个子块的第一层的偏置项---")
print(rgnet[0][1][0].bias.data)
