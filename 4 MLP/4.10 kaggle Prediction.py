import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]  # 读取指定数据集的url和哈希值
    os.makedirs(cache_dir, exist_ok=True)  # 创建data文件夹
    fname = os.path.join(cache_dir, url.split('/')[-1]) # 文件路径./data ， 文件名 url的/后的最后一列，即指定数据集名
    if os.path.exists(fname):   # 如果文件存在
        sha1 = hashlib.sha1()   # 初始化哈希值
        with open(fname, 'rb') as f:    # 以只读方式打开文件
            while True:
                data = f.read(1048576)  # 读取1M的数据
                if not data:            # 如果没有数据了就跳出
                    break
                sha1.update(data)       # 更新哈希值
        if sha1.hexdigest() == sha1_hash:   # 判断哈希值是否相等
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')   # 如果文件不存在
    r = requests.get(url, stream=True, verify=True)    # 从url读取文件，流读取，需要ssl证书
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

        def download_extract(name, folder=None):  # @save
            """下载并解压zip/tar文件"""
            fname = download(name)
            base_dir = os.path.dirname(fname)
            data_dir, ext = os.path.splitext(fname)
            if ext == '.zip':
                fp = zipfile.ZipFile(fname, 'r')
            elif ext in ('.tar', '.gz'):
                fp = tarfile.open(fname, 'r')
            else:
                assert False, '只有zip/tar文件可以被解压缩'
            fp.extractall(base_dir)
            return os.path.join(base_dir, folder) if folder else data_dir

        def download_all():  # @save
            """下载DATA_HUB中的所有文件"""
            for name in DATA_HUB:
                download(name)




#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/' # 指定网址首页

DATA_HUB['kaggle_house_train'] = (  #@save       DATA_HUB[键]=数据集网址，哈希值
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))    # pd.read_csv返回一个Dataframe类型的对象
test_data = pd.read_csv(download('kaggle_house_test'))

print('---训练集形状---')
print(train_data.shape)
print('---测试集形状---')
print(test_data.shape)
print('---前四个和最后两个特征---')
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 训练集第零列是ID，最后一列是标签，测试集是用于预测后提交kaggle判断成绩故没有标签
# iloc切片操作是不包含结束索引，故训练集最后一列不包含
# 拼接：训练集第一到倒数第二列+测试集第一到最后一列
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
#   获得包含所有数值型特征的索引列表
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index # all_features.dtypes筛选bool值为true的列
#   所有索引为数字类的特征执行标准化操作
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)   #fillna自动将缺失值赋0
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)  # get_dummy将分类变量变成多列，用0和1代表不属于和属于哪一类
print('---预处理完后的全部特征---')
print(all_features.shape)


n_train = train_data.shape[0]   # 获取行数
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)   # 取前n_train行
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)    # 取后n_train行
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)                # 转化为(n,1)的张量

# 训练
loss = nn.MSELoss()     # 实例化MSELoss类
in_features = train_features.shape[1]   # 获取列数，即特征维数

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))     # 设置1的下限，无穷的上线，防止取log下溢
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()  # 将一个元素的张量返回为python数字

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0     # 分别创建了两个空列表，用于存储每个轮次的训练集和测试集上的对数均方根误差
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64

# k交叉验证，k_fold输入(折数，训练集的特征矩阵和标签向量，训练的轮数，学习率，权重衰减系数，每个批次的样本数量)
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

