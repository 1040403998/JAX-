import jax.numpy as jnp
from jax.config import config
from jax import grad
config.update('jax_enable_x64', True)

import matplotlib.pyplot as plt


def Himmelblau(x, y):
    # x, y = params
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def draw_Himmelblau_forcefield():
    x = jnp.arange(-6.5,6.5,0.1)
    y = jnp.arange(-6.5,6.5,0.1)
    x, y = jnp.meshgrid(x,y)
    z = Himmelblau(x,y)

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
    plt.scatter(*maximum, s=80, c="r", marker="*", label="maximum")

    log_levels = jnp.array([2.5, 3.2, 3.8, 4.4, 4.7, 5.0, 5.1849, 5.45, 5.8, 6.2, 6.5])
    CP = plt.contour(x,y,z, levels=jnp.exp(log_levels), cmap="Blues")


    # 力场绘制
    f = lambda x, y: jnp.sum(Himmelblau(x, y))
    dx, dy = grad(f, argnums=(0, 1))(x, y)
    dx = dx.at[Himmelblau(x, y)>665].set(0)  # 去除边缘的力场线
    dy = dy.at[Himmelblau(x, y)>665].set(0)
    plt.streamplot(x, y, -dx, -dy, density=0.8)
    
    plt.legend(loc="lower right")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.savefig("fig7.9.png")

if __name__ == "__main__":
    draw_Himmelblau_forcefield()