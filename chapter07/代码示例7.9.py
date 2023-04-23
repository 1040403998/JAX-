


"""
代码示例7.9 :
    adam 优化器
"""

import jax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

def adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8):

  def init(x0):
    k0 = 1  # 迭代次数
    v0 = jnp.zeros_like(x0)
    s0 = jnp.zeros_like(x0)
    return x0, v0, s0, k0

  def update(g, state):
    x, v, s, k = state
    v = b1 * v + (1. - b1) * g
    s = b2 * s + (1. - b2) * jnp.square(g)
    v_hat = v / (1. - jnp.asarray(b1, v.dtype) ** k)
    s_hat = s / (1. - jnp.asarray(b2, s.dtype) ** k)
    x = x - learning_rate * v_hat / (eps + jnp.sqrt(s_hat))
    return x, v, s, k+1 

  def get_params(state):
    x, _, _, _ = state
    return x

  return init, update, get_params

init_fun, update_fun, get_params_fun = adam(learning_rate=0.01)


"""  优化器测试  """

init_fun, update_fun, get_params_fun = adam(learning_rate=0.01)

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

for i in range(50000):
  value, opt_state = step(opt_state)

  if (i+1) % 100 == 0:
    print("step = {}, value = {}".format(i+1, value))


