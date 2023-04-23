

import torchvision.transforms
import jax
import jax.numpy as jnp

# 数据的获取
train_dataset = torchvision.datasets.MNIST(root="./data/mnist",
                                           train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root="./data/mnist",
                                          train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)
train_data = jnp.array(train_dataset.train_data.numpy())
train_labels = jnp.array(train_dataset.train_labels.numpy())
test_data = jnp.array(test_dataset.test_data.numpy())
test_labels = jnp.array(test_dataset.test_labels.numpy())

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

import matplotlib.pyplot as plt
index = 7
data = train_data[index]
label = train_labels[index]

plt.rcParams['figure.figsize'] = (13.0, 13.0) # 设置figure_size尺寸
fig, ax = plt.subplots()
ax.imshow(data, interpolation='nearest')

x = y = jnp.arange(0, 28, 1)
x, y = jnp.meshgrid(x, y)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    x_val, y_val = int(x_val), int(y_val)
    c = data[y_val][x_val]
    ax.text(x_val, y_val, c, va='center', ha='center')

plt.savefig("mnist_test.png")
plt.show()


