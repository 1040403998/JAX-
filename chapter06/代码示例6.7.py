"""
代码示例 6.7 :
    扫描学习率参数
"""

import jax
import jax.numpy as jnp
from jax import grad, vmap
from typing import Tuple
import torchvision.transforms
import matplotlib.pyplot as plt
# pip install GPUtil 安装GPUtil
# import GPUtil # 如果使用GPU环境，则需要导入GPUtil

# 数据的获取
train_dataset = torchvision.datasets.MNIST(
    root="./data/mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_dataset = torchvision.datasets.MNIST(
    root="./data/mnist",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

train_data = jnp.array(train_dataset.train_data.numpy())
train_labels = jnp.array(train_dataset.train_labels.numpy())
test_data = jnp.array(test_dataset.test_data.numpy())
test_labels = jnp.array(test_dataset.test_labels.numpy())

print(train_data.shape)  # (60000, 28, 28)
print(train_labels.shape)  # (60000,)
print(test_data.shape)  # (10000, 28, 28)
print(test_labels.shape)  # (10000,)

# 数据的可视化
def draw(image):
    plt.rcParams["figure.figsize"] = (13.0, 13.0)  # 设置图片尺寸
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation="nearest")

    x = y = jnp.arange(0, 28, 1)
    x, y = jnp.meshgrid(x, y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        x_val, y_val = int(x_val), int(y_val)
        gray_scale = image[y_val][x_val]
        ax.text(x_val, y_val, gray_scale, va="center", ha="center")

    plt.savefig("mnist_test.png")
    plt.show()


index = 7
data = train_data[index]
print(train_labels[index])  # 3
draw(data)

# 数据的预处理
train_data = (train_data / 255).reshape(-1, 784)  # 训练数据的归一化
test_data = (test_data / 255).reshape(-1, 784)  # 测试数据的归一化
train_labels = jnp.eye(10)[train_labels]  # 训练标签转化为one-hot数组
test_labels = jnp.eye(10)[test_labels]  # 测试标签转化为one-hot数组

print(train_data.shape)  # (60000, 784)
print(train_labels.shape)  # (60000, 10)
print(test_data.shape)  # (10000, 784)
print(test_labels.shape)  # (10000, 10)

# softmax 测试
def softmax(o):
    return jnp.exp(o) / jnp.sum(jnp.exp(o))


key = jax.random.PRNGKey(2)
r_arr = jax.random.normal(key, shape=(10,)) * 2.2
y_arr = softmax(r_arr)

# print(r_arr)
# [ 1.5197709  -1.0723704  -2.5427358   0.26638603 -0.4311657
#  -1.1173288   2.0145104   3.7612953  -0.80848736  0.31494498]
# print(y_arr)
# [0.07670916 0.00574241 0.00131985 0.02190328 0.01090351
#  0.00548996 0.12580848 0.7216538  0.00747649 0.02299313]

from jax.scipy.special import logsumexp

from typing import Any, Dict


class ArrayType:
    def __getitem__(self, idx):
        return Any


f32 = ArrayType()


def predict(
    params: Tuple[f32[(10, 728)], f32[(10,)]], image: f32[(728,)]) -> f32[(10,)]:
    w, b = params
    logits = jnp.sum(w * image, axis=-1) + b
    return logits - logsumexp(logits)


batched_predict = vmap(predict, in_axes=(None, 0))


def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=-1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=-1)
    return jnp.mean(predicted_class == target_class)


v_accuracy = vmap(accuracy, in_axes=(0, None, None))


def init_network_params(seed, scale=1e-2):
    key = jax.random.PRNGKey(seed)
    w_key, b_key = jax.random.split(key, 2)
    w = scale * jax.random.normal(w_key, shape=(10, 784))
    b = scale * jax.random.normal(b_key, shape=(10,))
    return (w, b)


@jax.jit
def update(params, x, y, lr):
    w, b = params
    dw, db = grad(loss)(params, x, y)
    return (w - lr * dw, b - lr * db)


v_update = vmap(update, in_axes=(0, None, None, 0))


def data_loader(images, labels, batch_size):
    pos = 0
    sample_nums = images.shape[0]
    while pos < sample_nums:
        images_batch = images[pos : pos + batch_size]
        labels_batch = labels[pos : pos + batch_size]
        pos += batch_size
        yield (images_batch, labels_batch)


# 超参数设置
n_trail_step_size = 10
step_size = jnp.linspace(0, 1, n_trail_step_size)
num_epochs = 20
batch_size = 100
noise_scale = 1e-5

key = jax.random.PRNGKey(0)
seed = jax.random.uniform(key, shape=(n_trail_step_size,)) * 100
params_array = vmap(init_network_params)(seed=seed.astype(int))


import time

t_list = []
# 模型训练
for epoch in range(num_epochs):
    start_time = time.time()
    training_generator = data_loader(train_data, train_labels, batch_size)
    for x, y in training_generator:
        # 添加噪声
        _, key = jax.random.split(key, 2)
        x += noise_scale * jax.random.normal(key, x.shape)
        params_array = v_update(params_array, x, y, step_size)

    # GPUtil.showUtilization()

    # 输出信息
    epoch_time = time.time() - start_time
    t_list.append(epoch_time)

print(sum(t_list[1:]) / (len(t_list) - 1))
