


"""
代码示例 4.11 :
    使用多项式 Pn(x) 拟合数据 (L1正则化)
"""

import jax
import jax.numpy as jnp
from jax import grad

n = 15

# 定义用于拟合的多项式
def Pn(x, params):
    assert len(params) == n+1
    res = 0.
    for i in range(n+1):
        res += params[i] * x ** i
    return res    

# 损失函数
def loss(params, x_sample, y_sample):
    return jnp.mean((Pn(x_sample, params) - y_sample) ** 2)

@jax.jit
def update(params, x_sample, y_sample, lr):
    dparams = grad(loss)(params, x_sample, y_sample)
    return params - dparams * lr

# 数据读入
x_array = jnp.arange(10) + 1
y_array = jnp.array([49.300, 53.070, 58.210, 62.540, 67.080, \
                     71.423, 75.970, 80.340, 84.705, 89.190])

# 超参数设置
steps = 200000         # 行走步数
learning_rate = 1E-19  # 步长（学习率）
params = jnp.array([44.6582,4.45901818181818,0,0,0,0,0,0,0,0,0,0,0,0,0,0])    # 起始位置
print(len(params))
# 梯度下降
for step in range(steps):
    params = update(params, x_array, y_array, lr=learning_rate)

    if (step + 1) % 1000 == 0:
        err = loss(params, x_array, y_array)
        print("step = {}, loss = {}".format(step+1, err))

print(params)

"""
[ 4.4658199e+01  4.4590182e+00 -1.2832770e-15  2.5089999e-15
  2.0478641e-14  7.8951304e-14  1.8825361e-13 -2.9393821e-13
 -9.5938197e-12 -1.1090036e-10  2.3014845e-11  4.0654646e-11
  5.1432702e-11  4.2215530e-11  2.4852372e-11  3.4902144e-11]
""" 
