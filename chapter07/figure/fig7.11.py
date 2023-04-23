



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

def draw_Himmelblau():
    x = jnp.arange(-6.5,6.5,0.1)
    y = jnp.arange(-6.5,6.5,0.1)
    x, y = jnp.meshgrid(x,y)
    z = Himmelblau((x,y))

    maximum = (-0.270845, -0.923039)
    minima1 = (      3.0,       2.0)
    minima2 = (-2.905118,  3.131312)
    minima3 = (-3.779310, -3.283186)
    minima4 = ( 3.584428, -1.848126)
    
    fig = plt.figure()
    plt.scatter(*minima1, s=80, c="b", marker="p", zorder=10) # , label="minima")
    plt.scatter(*minima2, s=80, c="b", marker="p", zorder=10)
    plt.scatter(*minima3, s=80, c="b", marker="p", zorder=10)
    plt.scatter(*minima4, s=80, c="b", marker="p", zorder=10)

    log_levels = jnp.array([2.5, 3.2, 3.8, 4.4, 4.7, 5.0, 5.1849, 5.45, 5.8, 6.2, 6.5])
    CP = plt.contour(x,y,z, levels=jnp.exp(log_levels),cmap="Blues", zorder=-10)

draw_Himmelblau()



# update params

plt.annotate(text=r"$\theta_{init}$", 
            xy=(4.5, 5.0), xytext=(1, 1), textcoords = "offset points")
# params_array = get_trajectory(theta_init = jnp.array([4.5, 5.0]), alpha=0.001, beta=0.8)
# plt.plot(params_array[:,0], params_array[:,1], c = "purple", linestyle="--", label = r"$\beta = 0.80$", zorder=3)
# params_array = get_trajectory(theta_init = jnp.array([4.5, 5.0]), alpha=0.001, beta=0.9)
# plt.plot(params_array[:,0], params_array[:,1], c = "orange", linestyle="-.", label = r"$\beta = 0.90$",zorder=2)
# params_array = get_trajectory(theta_init = jnp.array([4.5, 5.0]), alpha=0.001, beta=0.95)
# plt.plot(params_array[:,0], params_array[:,1], c = "red", linestyle="-", label = r"$\beta = 0.95$", zorder=1)
params_array = get_trajectory(theta_init = jnp.array([4.5, 5.0]), alpha=0.001, beta=1.00)
plt.plot(params_array[:500,0], params_array[:500,1], c = "red", linestyle="--", label = r"$\beta = 1.00$", zorder=3)

plt.legend(loc="lower right")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
# plt.savefig("fig7.11.png")
plt.savefig("fig7.11.2.png")