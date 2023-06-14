import torch
from torch import nn
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

# 使用内置初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_normal)    # apply方法会自动将模块对象net作为参数传递给调用函数
print("---w服从(0,0.01)的正太分布，b全为0---")
print(net[0].weight.data[0], net[0].bias.data[0])

print("*************")
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print("---将所有参数初始化为给定的常数，比如初始化为1---")
print(net[0].weight.data[0], net[0].bias.data[0])
print("*************")


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


net[0].apply(init_xavier)   # 使用Xavier初始化方法初始化第一个神经网络层
net[2].apply(init_42)       # 将第三个神经网络层初始化为常量值42
print('---第一层的W---')
print(net[0].weight.data[0])
print('---第三层的W---')
print(net[2].weight.data)


# 自定义初始化

def my_init(m):
    if type(m) == nn.Linear:
        # [(name, param.shape) for name, param in m.named_parameters()]是一个列表生成式
        # 当使用 * 运算符时，它会将列表中的元素展开作为多个参数传递给函数
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5  # 绝对值>=5的W被输出

net.apply(my_init)
print('---自定义初始化，只有绝对值>=5的W被输出---')
print(net[0].weight[:2])


print('**************')
print('×××参数绑定×××')
# 参数绑定

shared = nn.Linear(8, 8) # 我们需要给共享层一个名称，以便可以引用它的参数
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print('---共享层参数是否相同---')
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100  # 神经元第一个输入到第一个输出的权重被置为100
# 确保它们实际上是同一个对象，而不只是有相同的值
print('---修改其中一层后，参数是否仍然相等---')
print(net[2].weight.data[0] == net[4].weight.data[0])