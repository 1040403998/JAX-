


"""
代码示例7.8 :
    adadelta 优化器
"""

import jax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

"""  adadelta 优化器  """
def adadelta(gamma = 0.9, eps=1e-8):

  def init(x0):
    s0 = jnp.zeros_like(x0)
    t0 = jnp.zeros_like(x0)
    x_old = jnp.zeros_like(x0)
    return x0, s0, t0, x_old

  def update(g, state):
    x, s, t, x_old = state
    s = gamma * s + (1. - gamma) * jnp.square(g)
    t = gamma * t + (1. - gamma) * jnp.square(x - x_old)
    x_old = x
    x = x - (eps + jnp.sqrt(t)) / (eps + jnp.sqrt(s)) * g
    return x, s, t, x_old 

  def get_params(state):
    x, _, _, _ = state
    return x

  return init, update, get_params


"""  优化器测试  """

init_fun, update_fun, get_params_fun = adadelta()

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

for i in range(100):
  value, opt_state = step(opt_state)
  print("step = {}, value = {}".format(i+1, value))

  # if (i+1) % 10 == 0:
  #   print("step = {}, value = {}".format(i+1, value))


