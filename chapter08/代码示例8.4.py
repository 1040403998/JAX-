


"""
代码示例8.4 :
    LSTM 模型的测试

赛题: https://www.jingsailian.com/zlk/41859.html 
"""


""" 循环神经网络模型 """
import jax
import jax.numpy as jnp
import optax
import haiku as hk

def unroll_net(inputs: jnp.ndarray, hidden_size=10, output_size=1):
    """ Unrolls an LSTM over inputs """
    (T, B, N) = inputs.shape
    core = hk.LSTM(hidden_size=hidden_size)
    initial_state = core.initial_state(batch_size=B)
    outs, hidden_state_final = hk.dynamic_unroll(core, inputs, initial_state)
    outputs = hk.BatchApply(hk.Linear(output_size=output_size))(outs)
    return outputs, hidden_state_final

LSTMModel = hk.transform(unroll_net)

"""  数据的读取  """
import os
import pandas as pd

# 设置文件名称
data_dir_name  = "data"                 # 数据文件夹名称
data_file_name =  "BCHAIN-MKPRU.csv"    # 数据文件名称
exception = "directory/file \"{}\" does not exist under the path \"{}\"" # 报错提示

dir_path = os.path.dirname(__file__)    # 获取当前文件所在文件夹的路径
data_path = os.path.join(dir_path, data_dir_name) # 获取数据所在文件夹的绝对路径
print("dir_path = ", dir_path)
print("data_path = ", data_path)

# 获取数据所在文件夹下，所有文件的文件名称, 保存在 files_list 这一列表当中 
try: files_list = os.listdir(data_path)
except: raise FileNotFoundError(exception.format(data_dir_name, dir_path))

# 检查文件夹下是否存在所需文件，读取相应文件的绝对路径
try: assert data_file_name in files_list
except: raise FileNotFoundError(exception.format(data_file_name, data_path))
finally: data_file_path = os.path.join(data_path, data_file_name)
print("data_file_path = {}\n".format(data_file_path))
print("number of files under data_file_path = {}\n".format(len(files_list)))

# 数据的获取
df = pd.read_csv(data_file_path)
df = df.dropna(axis = 0, how = "any")  # 去除所有包含缺失数据的条目
data_original = df.drop(columns = ["Date"]).to_numpy().reshape(-1)
print("original data's type  = ", type(data_original))  # float64 
print("original data's shape = ", data_original.shape)  # (1826,)  

# import numpy as np
# data_original = np.sin(np.arange(1826))
# print("original data's type  = ", type(data_original))  # float64 
# print("original data's shape = ", data_original.shape)  # (1826,)  

"""  数据的预处理  """
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)

# 超参数设置
training_ratio = 0.95     # 输入数据作为训练集的比例
training_days  = 20       # 为了预测下一天数据，所需要的输入数据的天数

# 数据的预处理及数据集的构造
def create_dataset(data, training_days):
    """ 将原本按照时间排列的数据重新打包，生成数据集

    输入：
        data: numpy格式的一维数组 (按时间顺序排列), 长度 length 
        training_days: 训练数据集序列的长度
    输出：
        长度为 length - traning_days + 1 条有效数据
    """
    data_list, labels_list = [], []
    for idx in range(len(data) - training_days):
        data_list.append(data[idx:idx + training_days])
        labels_list.append(data[idx + training_days])
    return jnp.array(data_list), jnp.array(labels_list)

## 数据的预处理 —— 归一化(normalization)，将数据缩放到区间[0,1]
maximum = jnp.max(data_original)
minimum = jnp.min(data_original)
data_normalized = (data_original - minimum) / (maximum - minimum)

## 数据集的构造与划分
data_total, labels_total = create_dataset(data_normalized, training_days)
print("data_total.shape = ", data_total.shape)  # (1806, 20)
train_size = int(len(data_total) * training_ratio)

train_data = data_total[: train_size].reshape(-1, 1, training_days) # 训练集数据
test_data  = data_total[train_size :].reshape(-1, 1, training_days) # 测试集数据
train_labels = labels_total[ : train_size].reshape(-1, 1, 1)        # 训练集标签
test_labels  = labels_total[train_size : ].reshape(-1, 1, 1)        # 测试集标签

print(train_data.shape)    # (1715, 1, 20)
print(train_labels.shape)  # (1715, 1,  1)
print(test_data.shape)     # ( 91 , 1, 20)
print(test_labels.shape)   # ( 91 , 1,  1)

""" 训练模型 """

model = LSTMModel

def train_model(model, train_data, train_labels, test_data, test_labels):

    rng = jax.random.PRNGKey(0)
    opt = optax.adam(learning_rate=0.01)

    @jax.jit
    def loss(params, x, y):
        pred, _ = model.apply(params, None, x)
        return jnp.mean(jnp.square(pred-y))
    
    @jax.jit
    def update(step, params, opt_state, x, y):
        value, grads = jax.value_and_grad(loss)(params, x, y)
        grads, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return value, params, opt_state

    # 初始化模型和优化器的状态

    params = model.init(rng, train_data)
    opt_state = opt.init(params)

    for step in range(10001):
        train_loss, params, opt_state = update(step, params, opt_state, 
                                            x = train_data, y = train_labels)
        if (step + 1) % 100 == 0:
            test_loss = loss(params, x=test_data, y = test_labels)
            print("step {}: train_loss = {:.5f}, test_loss = {:.5f}"\
                    .format(step+1, train_loss, test_loss))
    
    return params

model_params = train_model(model = model,
                    train_data = train_data, train_labels = train_labels, 
                    test_data  = test_data , test_labels  = test_labels)



""" 模型的测试 """

def predict(model, model_params, train_data):
    # 预热
    outputs, state = model.apply(params=model_params, rng=None, inputs=train_data)

    # 开始预测
    output_list = list(outputs.reshape(-1))
    output_list = [float(item) for item in output_list]
    get_input = lambda: jnp.array(output_list[-20:]).reshape(1,1,-1)
    for _ in range(len(test_data)):
        print(_, get_input())
        y, state = model.apply(params=model_params, rng=None, 
                           inputs=get_input(), initial_state = state)
        output_list.append(float(y[-1]))
    return output_list

pred = predict(model=model, model_params=model_params, 
        train_data=train_data, number_pred=test_data.shape[0])
pred = list(jnp.zeros(training_days)) + pred
print(len(pred))
print(len(data_original))
assert len(pred) == len(data_original)


import matplotlib.pyplot as plt

# plt.plot(data_normalized[train_size:], label = "original",  c = "b")
# plt.plot(pred[train_size:], label = "prediction", c = "red",linewidth=2, linestyle = "-.",)
plt.plot(data_normalized, label = "original",  c = "b")
plt.plot(pred, label = "prediction", c = "red",linewidth=2, linestyle = "-.",)
plt.plot((train_size, train_size), (0.2, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出

plt.xlabel("Days")
plt.ylabel("Price")
plt.legend(loc = "lower right")
plt.savefig('test_Result.png', format='png', dpi=200)
plt.close()

# import matplotlib.pyplot as plt
# x = jnp.arange(len(pred))
# plt.plot(x[train_size:], data_normalized[train_size:], label = "original",  c = "b")
# plt.scatter(x[train_size:], pred[train_size:], label = "prediction", c = "red",s=3)
# # plt.plot((train_size, train_size), (0.1, 1.05), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出

# plt.xlabel("Time (Days)")
# plt.ylabel("Value")
# # plt.ylim((-0.1, 1.1))

# plt.legend(loc = "lower right")
# plt.savefig('test_Result.png', format='png', dpi=200)
# plt.close()