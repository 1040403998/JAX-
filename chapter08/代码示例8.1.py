


"""
代码示例8.1 :
    梯度裁剪
"""


# 无梯度裁剪
def gradient_descent(alpha):
  def init(x0)      : return x0
  def update(g, x0) : return x0 - alpha * g
  def get_params(x0): return x0
  return init, update, get_params


# 梯度裁剪
import jax
import jax.numpy as jnp

def gradient_descent(alpha=0.01, theta=10.0):
    def init(x0): return x0
    def get_params(x0): return x0

    def update(g, x0): 
        g_norm = jnp.sqrt(jnp.sum(g**2))
        g = jax.lax.cond(
            pred = (g_norm < theta),
            true_fun  = (lambda _g:_g), 
            false_fun = (lambda _g:_g/g_norm*theta), 
            operand = g)
        return x0 - alpha * g
    
    return init, update, get_params



"""  优化器测试  """

init_fun, update_fun, get_params_fun = gradient_descent()

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


