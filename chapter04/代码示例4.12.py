


"""
代码示例 4.12 :
    全连接神经网络
"""


from tkinter import W
import jax
import jax.numpy as jnp
from jax import grad
import torchvision.transforms
import matplotlib.pyplot as plt


# 数据的获取
train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, 
                transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, 
                transform=torchvision.transforms.ToTensor(), download=True)

train_data = jnp.array(train_dataset.train_data.numpy())
train_labels = jnp.array(train_dataset.train_labels.numpy())
test_data = jnp.array(test_dataset.test_data.numpy())
test_labels = jnp.array(test_dataset.test_labels.numpy())

print(train_data.shape)    # (60000, 28, 28)
print(train_labels.shape)  # (60000,)
print(test_data.shape)     # (10000, 28, 28)
print(test_labels.shape)   # (10000,)

# 数据的可视化
def draw(image):
    # plt.rcParams['figure.figsize'] = (13.0, 13.0)  # 设置图片尺寸

    fig, ax = plt.subplots()
    fig.set_size_inches(15.0, 15.0)
    ax.imshow(image, interpolation='nearest')

    x = y = jnp.arange(0, 28, 1)
    x, y = jnp.meshgrid(x, y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        x_val, y_val = int(x_val), int(y_val)
        gray_scale = image[y_val][x_val]
        if int(gray_scale) > 50:
            ax.text(x_val, y_val, gray_scale, va='center', ha='center')
        else: ax.text(x_val, y_val, gray_scale, va='center', ha='center', color="white")

    # plt.savefig("mnist_test.png")
    plt.savefig(r"/mnt/c/Users/lenovo/Desktop/代码/chapter04/mnist_test.png", dpi=1000)
    plt.show()

index = 7
data = train_data[index]
print(train_labels[index])  # 3
draw(data)
print("drawing complete")

# 数据的预处理
train_data = (train_data / 255).reshape(-1, 784)    # 训练数据的归一化
test_data = (test_data / 255).reshape(-1, 784)      # 测试数据的归一化
train_labels = jnp.eye(10)[train_labels]   # 训练标签转化为one-hot数组
test_labels = jnp.eye(10)[test_labels]     # 测试标签转化为one-hot数组

print(train_data.shape)    # (60000, 784)
print(train_labels.shape)  # (60000, 10)
print(test_data.shape)     # (10000, 784)
print(test_labels.shape)   # (10000, 10)

# softmax 测试
def softmax(o):
    return jnp.exp(o) / jnp.sum(jnp.exp(o))

key = jax.random.PRNGKey(1)
r_arr = jax.random.normal(key, shape=(10,)) * 2.2
y_arr = softmax(r_arr)

print(r_arr)  
# [ 1.5197709  -1.0723704  -2.5427358   0.26638603 -0.4311657  
#  -1.1173288   2.0145104   3.7612953  -0.80848736  0.31494498]
print(y_arr)
# [0.07670916 0.00574241 0.00131985 0.02190328 0.01090351 
#  0.00548996 0.12580848 0.7216538  0.00747649 0.02299313]

from jax.scipy.special import logsumexp

def batched_predict(params, images):
    w, b = params
    images = jnp.expand_dims(images, 1)
    logits = jnp.sum(w * images, axis=-1) + b
    return logits - logsumexp(logits)

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=-1)
    predicted_class = jnp.argmax(
        batched_predict(params, images), axis=-1)
    return jnp.mean(predicted_class == target_class)

def init_network_params(key, scale=1e-2):
    w_key, b_key = jax.random.split(key, 2)
    w = scale * jax.random.normal(w_key, shape=(10, 784))
    b = scale * jax.random.normal(b_key, shape=(10, ))
    return (w, b)

key = jax.random.PRNGKey(0)
params = init_network_params(key)

@jax.jit
def update(params, x, y, lr):
    w, b = params
    dw, db = grad(loss)(params, x, y)
    return (w - lr * dw, b - lr * db)

def data_loader(images, labels, batch_size):
    pos = 0
    sample_nums = images.shape[0]
    while pos < sample_nums:
        images_batch = images[pos : pos + batch_size]
        labels_batch = labels[pos : pos + batch_size]
        pos += batch_size
        yield (images_batch, labels_batch)

# 超参数设置
step_size = 0.01
num_epochs = 10
batch_size = 100
noise_scale = 1e-5

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

        # 训练参数
        params = update(params, x, y, step_size)
    
    # 输出信息
    epoch_time = time.time() - start_time
    t_list.append(epoch_time)
    train_acc = accuracy(params, train_data, train_labels)
    test_acc = accuracy(params, test_data, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch+1, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
# print(t_list)
# print(sum(t_list))
# print(sum([2.88, 1.96, 2.05, 2.05, 2.04, 2.08, 2.08, 1.98, 2.00,2.04,]))