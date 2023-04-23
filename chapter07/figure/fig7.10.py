



import jax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

def momentum(alpha, beta):

  def init(x0):
    """ 初始化函数 """
    v0 = jnp.zeros_like(x0)
    return (x0, v0)

  @jax.jit
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

def Himmelblau(params):
    x, y = params
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def get_trajectory(theta_init, alpha, beta):

    init_fun, update_fun, get_params_fun = momentum(alpha, beta)

    @jax.jit
    def step(opt_state):
        params = get_params_fun(opt_state)    # 使用参数读取函数，返回模型参数
        value, g = jax.value_and_grad(Himmelblau)(params)
        opt_state = update_fun(g, opt_state)  # 使用参数更新函数，更新优化器状态
        return value, opt_state

    # 算法迭代步骤
    opt_state = init_fun(theta_init)        # 使用初始化函数，初始化优化器状态

    params_list = [theta_init, ]
    for i in range(1000):
        value, opt_state = step(opt_state)
        if (i+1) % 100 == 0:
            print(opt_state)
        params_list.append(get_params_fun(opt_state))
    
    params_array = jnp.array(params_list)
    return params_array


"""  可视化输出  """
import matplotlib.pyplot as plt

iteration_num = 250
params_array1 = get_trajectory(theta_init = jnp.array([4.5, 5.0]), alpha=0.001, beta=0.0)
z_value1 = Himmelblau(params_array1.T)
plt.plot(jnp.arange(iteration_num), jnp.log(z_value1[:iteration_num]), c = "purple", linestyle="--", label = r"gradient descent ($\beta = 0.00$)", zorder=3)

params_array2 = get_trajectory(theta_init = jnp.array([4.5, 5.0]), alpha=0.001, beta=0.5)
z_value2 = Himmelblau(params_array2.T)
plt.plot(jnp.arange(iteration_num), jnp.log(z_value2[:iteration_num]), c = "red", linestyle="-", label = r"momentum ($\beta = 0.50$)", zorder=3)

params_array3 = get_trajectory(theta_init = jnp.array([4.5, 5.0]), alpha=0.001, beta=0.75)
z_value3 = Himmelblau(params_array3.T)
plt.plot(jnp.arange(iteration_num), jnp.log(z_value3[:iteration_num]), c = "orange", linestyle="-.", label = r"momentum ($\beta = 0.75$)", zorder=3)
# params_array = get_trajectory(theta_init = jnp.array([4.5, 5.0]), alpha=0.001, beta=0.9)

xytext = (10,-10)
plt.annotate(text=r"$f(\vec{\theta})=(\theta_1^2+\theta_2-11)^2+(\theta_1+\theta_2^2-7)^2$",  xy=(0, -37), xytext=xytext, textcoords = "offset points")
plt.annotate(text=r"$y = log(f(\vec{\theta}_k))$",  xy=(60, -42), xytext=xytext, textcoords = "offset points")
plt.legend(loc="lower left")
plt.grid("-")
plt.xlabel("number of iteration k")
plt.ylabel("value of Himmelblau function")
plt.savefig("fig7.10.png")