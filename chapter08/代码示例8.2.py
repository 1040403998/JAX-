


"""
赛题: https://www.jingsailian.com/zlk/41859.html 

"""


""" 循环神经网络模型 """
import jax
import jax.numpy as jnp
import jax.random as random

from abc import abstractmethod
from typing import Tuple, Any, NamedTuple

class RNNCore(object):
    @abstractmethod
    def __call__(self, inputs, prev_state, params) -> Tuple[Any, Any]:
        """ 执行一步 RNN 的计算

        输入参数:
            input : 用于训练模型的 2 维数组 
                形状为 (批大小 B, 单一时间步内输入数据的维数 N) 
            prev_state : 前一时刻循环网络的隐状态
            params : 模型中所需的参数
        
        函数返回: 
            一个形如 (output, next_state) 的元组
            output: 循环神经网络隐藏层的输出，形状为 (批大小, 隐藏层的维数)
            next_state: 神经网络在下一时刻的隐状态, 形状与 prev_state 完全相同
        """

    @abstractmethod
    def initial_state(self, batch_size: int):
        """ 为当前的 RNN核 构造初始状态
    
        输入参数：
            batch_size: 用于确定输入数据的批大小
        
        返回：
            提供一组
        """    

class VanillaRNN(RNNCore):
    def __init__(self, hidden_size:int):
        self.hidden_size = hidden_size # 隐藏层中元素的数目
    
    def __call__(self, inputs, prev_state, params) -> Tuple[Any, Any]:
        r"""
        math::
            v_t = tanh(W_h u_t + M v_{t-1} + b_h)

        模型输入：
            input      : shape = (B, N)
            prev_state : shape = (B, hidden_size)
            params     : 包含3个 jax 的数组 (W, M, b)
                W.shape = (hidden_size, N)
                M.shape = (hidden_size, hidden_size) 
                b.shape = (hidden_size, )
                这里的N是单一时间步内输入数据的维数

        模型输出：
            output     : shape = (B, hidden_size)
            next_state : shape = (B, hidden_size)
        """
        W, M, b = params
        Wu = jnp.einsum("in, bn -> bi", W, inputs)
        Mv = jnp.einsum("ij, bj -> bi", M, prev_state)
        output = jax.nn.tanh(Wu + Mv + b)  # bi, bi, i -> bi
        next_state = output
        return output, next_state

    def initial_state(self, batch_size: int):
        """  初始化模型的隐状态 """
        state = jnp.zeros([batch_size, self.hidden_size])
        return state
    


""" 构造循环神经网络的隐藏层 """

def static_unroll(core, input_sequence, initial_state, params) -> Tuple[Any, Any]:
    """ 将循环网络的核递归地"展开", 从而构造出隐藏层完整的计算图
    (编译过后, 网络中的循环结构将不再保留)

    函数输入:
        core           : 一个基于 RNNCore 构造的基类
        input_sequence : 隐藏层的输入, 是一个3维的数组
        initial_state  : 隐藏层的初始状态
        params         : 隐藏层的参数
    
    函数输出：
        output_sequence : 隐藏层的输出
        final_state     : 隐藏层的最终状态
    
    注：
        input_sequence .shape = (T, B, N)
        output_sequence.shape = (T, B, hidden_size)
    """
    output_sequence = []
    time_steps = input_sequence.shape[0]
    state = initial_state
    for t in range(time_steps):
        inputs = input_sequence[t]
        outputs, state = core(inputs, state, params)
        output_sequence.append(outputs)
    output_sequence = jnp.stack(output_sequence, axis=0)
    return output_sequence, state

def dynamic_unroll(core, input_sequence, initial_state, params) -> Tuple[Any, Any]:
    """ 将循环网络的核递归地"展开", 从而构造出隐藏层完整的计算图
    (编译过后, 网络中的循环结构将被保留)

    函数输入:
        core           : 一个基于 RNNCore 构造的基类
        input_sequence : 隐藏层的输入, 是一个3维的数组
        initial_state  : 隐藏层的初始状态
        params         : 隐藏层的参数
    
    函数输出：
        output_sequence : 隐藏层的输出
        final_state     : 隐藏层的最终状态
    
    注：
        input_sequence .shape = (T, B, N)
        output_sequence.shape = (T, B, hidden_size)
    """
    def scan_fun(prev_state, inputs):
        outputs, next_state = core(inputs, prev_state, params)
        return next_state, outputs

    final_state, output_sequence = jax.lax.scan(
        f = scan_fun,
        init = initial_state,
        xs = input_sequence,)

    return output_sequence, final_state


""" 构造完整的循环神经网络模型 """   
class RNNModel(object):
    """
    math::
        v_t = tanh(W_h u_t + M v_{t-1} + b_h)
        o_t = W_o v_t + b_o
    """
    def __init__(self, hidden_size:int, output_size: int):
        self.hidden_size = hidden_size  # 隐藏层中元素的数目
        self.output_size = output_size  # 神经网络输出的维数

    def init(self, rng, inputs):
        """ 返回模型中的所有初始参数 """
        (T, B, N) = inputs.shape
        Wh_key, M_key, bh_key, Wo_key, bo_key = random.split(rng, num=5)

        # 隐藏层参数
        Wh = random.normal(key=Wh_key, shape = (self.hidden_size, N))
        M  = random.normal(key=M_key , shape = (self.hidden_size, self.hidden_size))
        bh = random.normal(key=bh_key, shape = (self.hidden_size, ))
        hidden_layer_params = (Wh, M, bh)

        # 输出层参数
        Wo = random.normal(key=Wo_key, shape = (self.output_size, self.hidden_size))
        bo = random.normal(key=bo_key, shape = (self.output_size, ))
        output_layer_params = (Wo, bo)
        return (hidden_layer_params, output_layer_params)

    def apply(self, params, rng, inputs, initial_state=None):
        """ 通过模型, 计算神经网络的输出及隐藏层的最终状态 """
        (T, B, N) = inputs.shape
        hidden_layer_params, output_layer_params = params

        # 隐藏层前向传播
        core = VanillaRNN(hidden_size=self.hidden_size)
        if not isinstance(initial_state, jnp.ndarray):
            initial_state = core.initial_state(batch_size=B)
        hidden_outputs, hidden_state_final = dynamic_unroll(core=core, 
                input_sequence=inputs, initial_state=initial_state, 
                params=hidden_layer_params,)
        
        # 输出层前向传播
        Wo, bo = output_layer_params
        outputs = jnp.einsum("mi, tbi -> tbm", Wo, hidden_outputs) + bo
        
        return outputs, hidden_state_final



"""  数据的读取  """
import os

# 设置文件名称
data_dir_name  = "data"                 # 数据文件夹名称
# data_file_name = "LBMA-GOLD.csv"        # 数据文件名称 1
data_file_name =  "BCHAIN-MKPRU.csv"    # 数据文件名称 2
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
print("number of files under data_path = {}\n".format(len(files_list)))


"""  数据的预处理  """
import pandas as pd
import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)

# 超参数设置
training_ratio = 0.95    # 输入数据作为训练集的比例
training_days = 20       # 为了预测下一天数据，所需要的输入数据的天数

# 数据的获取
df = pd.read_csv(data_file_path)
df = df.dropna(axis = 0, how = "any")  # 去除所有包含缺失数据的条目
data_original = df.drop(columns = ["Date"]).to_numpy().reshape(-1)
print("original data's type  = ", type(data_original))  # float64 
print("original data's shape = ", data_original.shape)  # (1826,)  


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

print(train_data.shape)    # (1354, 1, 20)
print(train_labels.shape)  # (1354, 1, 1)
print(test_data.shape)     # (452 , 1, 20)
print(test_labels.shape)   # (452 , 1, 1)

""" 训练模型 """

model = RNNModel(hidden_size=10, output_size=1)

def train_model(model, train_data, train_labels, test_data, test_labels):
    import optax

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

    for step in range(9001):
        train_loss, params, opt_state = update(step, params, opt_state, 
                                            x = train_data, y = train_labels)
        if (step + 1) % 100 == 0:
            test_loss = loss(params, x=test_data, y = test_labels)
            print("step {}: train_loss = {}, test_loss = {}"\
                    .format(step+1, train_loss, test_loss))
    
    return params

model_params = train_model(model = model,
                    train_data = train_data, train_labels = train_labels, 
                    test_data  = test_data , test_labels  = test_labels)



""" 模型的测试 """

def predict(model, model_params, train_data, number_pred):
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
x = jnp.arange(len(pred))
plt.plot(x, data_normalized, label = "original",  c = "b")
plt.scatter(x, pred, label = "prediction", c = "red",s=3)
plt.plot((train_size, train_size), (0.1, 1.05), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出

plt.xlabel("Time (Days)")
plt.ylabel("Value")
plt.ylim((-0.1, 1.1))

plt.legend(loc = "lower right")
plt.savefig('test_Result.png', format='png', dpi=200)
plt.close()
