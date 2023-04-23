


"""
代码示例7.5 :
    函数极小值点的寻找
"""

import jax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

def momentum(alpha, beta):

  def init(x0):
    """ 初始化函数 """
    v0 = jnp.zeros_like(x0)
    return (x0, v0)

  def update(g, opt_state):
    """ 参数更新函数 """
    x, v = opt_state
    v_new = beta * v - alpha * g  # 更新速度
    x_new = x + v_new             # 更新位置
    return (x_new, v_new)

  def get_params(opt_state):
    """ 参数读取函数 """
    x, _ = opt_state
    return x
  
  return init, update, get_params


"""  优化器测试  """

# 优化器测试
alpha, beta = 0.001, 0.8                # 超参数设置
init_fun, update_fun, get_params_fun = momentum(alpha, beta)

def Himmelblau(params):
    x, y = params
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

@jax.jit
def step(opt_state):
  params = get_params_fun(opt_state)    # 使用参数读取函数，返回模型参数
  value, g = jax.value_and_grad(Himmelblau)(params)
  opt_state = update_fun(g, opt_state)  # 使用参数更新函数，更新优化器状态
  return value, opt_state

# 算法迭代步骤
theta_init = jnp.array([4.3, 5.0])      # 起点设置
opt_state = init_fun(theta_init)        # 使用初始化函数，初始化优化器状态

for i in range(1000):
  value, opt_state = step(opt_state)

  if (i+1) % 100 == 0:
    print("step = {}, value = {}".format(i+1, value))


