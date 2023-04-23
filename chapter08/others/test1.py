#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tushare as ts
import pandas as pd
import torch
from torch import nn
import time


DAYS_FOR_TRAIN = 20


class LSTM_Regression(nn.Module):
    """
        使用LSTM进行回归
        
        参数：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape     # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x


def create_dataset(data, days_for_train=10):
    """
        根据给定的序列data，生成数据集
        
        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。

        若给定序列的长度为d，将输出长度为(d-days_for_train+1)个输入/输出对
    """
    dataset_x, dataset_y= [], []
    for i in range(len(data)-days_for_train):
        _x = data[i:(i+days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i+days_for_train])
    return (np.array(dataset_x), np.array(dataset_y))


import os
# 对应文件夹的名称
data_dir_name = "data"
# data_file_name = "LBMA-GOLD.csv"
data_file_name = "BCHAIN-MKPRU.csv"

# 获取当前文件所在文件夹的路径
dir_path = os.path.dirname(__file__)
print("dir_path = ", dir_path)

# 获取数据所在文件夹的路径
data_path = os.path.join(dir_path, data_dir_name)

# 获取数据所在文件夹下所有文件的文件名
try:
    files = os.listdir(data_path)
except:
    raise FileNotFoundError("directory {} does not exist under the path {}"\
        .format(data_dir_name, dir_path))

# 检查数据所在文件夹下是否存在 "face.txt" 和 "vertex.txt" 文件, 并获取相应文件的绝对路径
try:
    assert data_file_name in files
    # assert vertex_data_file_name in files
except:
    raise AssertionError("file {} and {} should be under the directory {}"\
        .format(data_file_name, data_file_name, data_path))
finally:
    data_file_path = os.path.join(data_path, data_file_name)
    # vertex_data_path = os.path.join(data_path, data_file_name)

print("data_file_path : ", data_file_path)

if __name__ == '__main__':
    t0 = time.time()

    df=pd.read_csv(data_file_path)
    df=df.dropna()
    data=df.drop(columns=['Date'])
    data_close=data  #.loc[1:1000]
    #data_close = ts.get_k_data('000001', start='2019-01-01', index=True)['close']# 取上证指数的收盘价
    #(df)
    # print(data_close)
    # data_close.to_csv('000001.csv', index=False) #将下载的数据转存为.csv格式保存
    # data_close = pd.read_csv('000001.csv') #读取文件
    # df_sh = ts.get_k_data('sh', start='2019-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d'))
    # #print(df_sh.shape)
    data_close = data_close.astype('float32').values  # 转换数据类型
    plt.plot(data_close)
    plt.savefig('data.png', format='png', dpi=200)
    plt.close()


    # 将价格标准化到0~1
    data_real = data_close
    max_value = np.max(data_close)
    min_value = np.min(data_close)
    data_close = (data_close - min_value) / (max_value - min_value)

    dataset_x, dataset_y = create_dataset(data_close, DAYS_FOR_TRAIN)

    # 划分训练集和测试集，75%作为训练集
    print("data_close.shape = ", data_close.shape)
    print("dataset_x.shape = ",dataset_x.shape ) # (1806, 20, 1)
    train_size = int(len(dataset_x) * 0.75)

    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]

    # 将数据改变形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_y = train_y.reshape(-1, 1, 1)

    # 转为pytorch的tensor对象
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    print("train_x.shape = ", train_x.shape)  # [1354, 1, 20]
    print("train_y.shape = ", train_y.shape)  # [1354, 1, 1]

    model = LSTM_Regression(DAYS_FOR_TRAIN, hidden_size = 8, output_size=1, num_layers=2) # 导入模型并设置模型的参数输入输出层、隐藏层等

	
    model_total = sum([param.nelement() for param in model.parameters()]) # 计算模型参数
    print("Number of model_total parameter: %.8fM" % (model_total/1e6))


    train_loss = []
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    for i in range(20):
        out = model(train_x)  
        # print(out.shape)   # # [735, 1, 1]
        loss = loss_function(out, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss.append(loss.item())
		
		# 将训练过程的损失值写入文档保存，并在终端打印出来
        with open('log.txt', 'a+') as f:
            f.write('{} - {}\n'.format(i+1, loss.item()))
        if (i+1) % 1 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.item()))

    # 画loss曲线
    plt.figure()
    plt.plot(train_loss, 'b', label='loss')
    plt.title("Train_Loss_Curve")
    plt.ylabel('train_loss')
    plt.xlabel('epoch_num')
    plt.savefig('loss.png', format='png', dpi=200)
    plt.close()


    # torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用
    t1=time.time()
    T=t1-t0
    print('The training time took %.2f'%(T/60)+' mins.')

    tt0=time.asctime(time.localtime(t0))
    tt1=time.asctime(time.localtime(t1))
    print('The starting time was ' , tt0)
    print('The finishing time was ', tt1)


    # for test
    model = model.eval() # 转换成测试模式
    # model.load_state_dict(torch.load('model_params.pkl'))  # 读取参数

    # 注意这里用的是全集 模型的输出长度会比原数据少DAYS_FOR_TRAIN 填充使长度相等再作图
    dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
    dataset_x = torch.from_numpy(dataset_x)

    pred_test = model(dataset_x) # 全量训练集
    # 的模型输出 (seq_size, batch_size, output_size)
    print("pred_test.shape = ",pred_test.shape)  # torch.Size([1806, 1, 1])
    pred_test = pred_test.view(-1).data.numpy()
    pred_test = np.concatenate((np.zeros(DAYS_FOR_TRAIN), pred_test))  # 填充0 使长度相同
    assert len(pred_test) == len(data_close)

    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(data_close, 'b', label='real')
    plt.plot((train_size, train_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    plt.legend(loc='best')
    plt.savefig('chapter09_test_figure.png', format='png', dpi=200)
    plt.close()
    pred_test=pred_test*(max_value - min_value)+min_value

pred=pd.DataFrame(pred_test)
real=pd.DataFrame(data_real)   
j=pd.concat([pred,real],axis=1,join='outer',sort=True)  
j.to_csv('GLOD1-200.csv')