


"""
代码示例 4.10 :
    最小二乘法
"""

import jax
import jax.numpy as jnp
from jax import grad

# 损失函数
def loss(params, x_sample, y_sample):
    w, b = params
    return jnp.mean((w * x_sample + b - y_sample) ** 2)

@jax.jit
def update(params, x_sample, y_sample, lr):
    w, b = params
    dw, db = grad(loss)(params, x_sample, y_sample)
    return w - lr * dw, b - lr * db

# 数据读入
x_array = jnp.arange(10) + 1
y_array = jnp.array([49.300, 53.070, 58.210, 62.540, 67.080, \
                     71.423, 75.970, 80.340, 84.705, 89.190])

# 超参数设置
steps = 20000        # 行走步数
learning_rate = 1E-2  # 步长（学习率）
params = (0., 30.)    # 起始位置

# 梯度下降
for step in range(steps):
    params = update(params, x_array, y_array, lr=learning_rate)

    if (step + 1) % 1000 == 0:
        err = loss(params, x_array, y_array)
        print("step = {}, loss = {}".format(step+1, err))

print(params)
dw, db = grad(loss)(params, x_array, y_array)
# dw = 7.6089054e-06
# db = -0.00018997397



""" 算法测试 """

w_opt = 4.45901818181818
b_opt = 44.6582000000000
w_err_list = []
b_err_list = []
loss_err_list = []
# 学习率测试
lr_array = jnp.logspace(-1, -5, 201)
steps = 5000

for learning_rate in lr_array:
    print(learning_rate)
    params = (4.45, 44.65)
    for step in range(steps):
        params = update(params, x_array, y_array, lr=learning_rate)

        if (step + 1) % 1000 == 0:
            err = loss(params, x_array, y_array)
            print("step = {}, loss = {}".format(step+1, err))
    w, b = params
    w_err = jnp.abs(w - w_opt)
    b_err = jnp.abs(b - b_opt)
    loss_err = jnp.abs(loss(params, x_array, y_array) - loss((w_opt, b_opt), x_array, y_array))

    w_err_list.append(w_err)
    b_err_list.append(b_err)
    loss_err_list.append(loss_err)

import matplotlib.pyplot as plt

log_lr = jnp.log10(lr_array)
w_err_list = jnp.log(jnp.array(w_err_list))
b_err_list = jnp.log(jnp.array(b_err_list))
loss_err_list = jnp.log(jnp.array(loss_err_list))

# plt.figure(dpi=1500)

plt.xlabel(r"log($\alpha$)")
plt.ylabel("log(err)")

plt.plot(log_lr, w_err_list, label = "err w", linestyle = "--")
plt.plot(log_lr, b_err_list, label = "err b", linestyle = ":")
plt.plot(log_lr, loss_err_list, label = "err loss", linestyle = "-")
plt.grid()
plt.legend(loc = "lower left")
plt.savefig("lstsq_err.png")
# plt.savefig(r"/mnt/c/Users/lenovo/Desktop/fig4.6.png")

from jax import jacfwd, jacrev
def hessian(f): return jacfwd(jacrev(f))
print(hessian(loss)(params, x_array, y_array))
print(grad(loss)(params, x_array, y_array))

# (DeviceArray(4.459082, dtype=float32, weak_type=True), DeviceArray(44.657753, dtype=float32, weak_type=True))
# (DeviceArray(7.6089054e-06, dtype=float32),            DeviceArray(-0.00018997, dtype=float32))
# {a0: 44.6582000000000, a1: 4.45901818181818}
print(39.350 * 4.459082 * 2)

print(331.45 * jnp.sqrt((1+22.7/273.15)*(1+0.3192*2809.1*0.25/1.01E5)))