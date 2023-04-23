


"""
代码示例7.7 :
    adagrad 优化器
"""

import jax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

"""  adagrad 优化器  """ 
def adagrad(alpha, eps=1e-8):

  def init(x0):
    s0 = jnp.zeros_like(x0)
    return x0, s0

  def update(g, state):
    x, s = state
    s += jnp.square(g)
    x = x - alpha / jnp.sqrt(eps + s) * g
    return x, s

  def get_params(state):
    x, _ = state
    return x

  return init, update, get_params


"""  RMSprop 优化器  """ 
def RMSprop(alpha, gamma=0.9, eps=1e-8):

  def init(x0):
    s0 = jnp.zeros_like(x0)
    return x0, s0

  def update(g, state):
    x, s = state
    s = s * gamma + jnp.square(g) * (1. - gamma)
    x = x - alpha / jnp.sqrt(eps + s) * g
    return x, s

  def get_params(state):
    x, _ = state
    return x

  return init, update, get_params



"""  优化器测试  """

alpha = 0.01                            # 超参数设置
init_fun, update_fun, get_params_fun = adagrad(alpha)

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


