

"""
代码示例7.10 :
    adam 优化器
"""

import jax
import optax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

def Himmelblau(params):
    x,y = params
    return (x ** 2 + y - 11) ** 2 + (x + y** 2 - 7) ** 2

theta_init = jnp.array([5.0, -4.0])
optimizer = optax.adam(learning_rate=0.01)

params = theta_init
opt_state = optimizer.init(params)

@jax.jit
def step(params, opt_state):
    value, grads = jax.value_and_grad(Himmelblau)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value

params_list = [theta_init, ]

for i in range(1000):
    params, opt_state, value = step(params, opt_state)

    # 算法测试及输出
    params_list.append(params)
    if (i+1) % 100 == 0:
        print(value)

print(params)
print("opt_state = ", opt_state)
value, grads = jax.value_and_grad(Himmelblau)(params)

print(grads)

""" 测试 """
import matplotlib.pyplot as plt

def draw_Himmelblau_update(params_list):
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
    plt.scatter(*minima1, s=80, c="b", marker="p", label="minima")
    plt.scatter(*minima2, s=80, c="b", marker="p", )
    plt.scatter(*minima3, s=80, c="b", marker="p", )
    plt.scatter(*minima4, s=80, c="b", marker="p", )

    log_levels = jnp.array([2.5, 3.2, 3.8, 4.4, 4.7, 5.0, 5.1849, 5.45, 5.8, 6.2, 6.5])
    CP = plt.contour(x,y,z, levels=jnp.exp(log_levels),cmap="Blues")

    # update params
    params_array =jnp.array(params_list)
    plt.plot(params_array[:,0], params_array[:,1], )


    plt.legend(loc="lower right")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.savefig("7.13 (optax).png")

draw_Himmelblau_update(params_list)
