


"""
代码示例7.2.3 :
    nesterov 动量法优化器
"""

import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

def nesterov(alpha, beta):

  def init(x0):
    """ 初始化函数 """
    v0 = jnp.zeros_like(x0)
    return (x0, v0)

  def update(g, opt_state):
    """ 参数更新函数 """
    x, v = opt_state
    v_new = beta * v - alpha * g           # 更新速度
    x_new = x + beta * v_new - alpha * g   # 更新位置
    return (x_new, v_new)

  def get_params(opt_state):
    """ 参数读取函数 """
    x, _ = opt_state
    return x
  
  return init, update, get_params

# 普通梯度下降法优化器
def gradient_descent(alpha):
  def init(x0)      : return x0
  def update(g, x0) : return x0 - alpha * g
  def get_params(x0): return x0
  return init, update, get_params

