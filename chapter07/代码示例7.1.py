


"""
代码示例7.1 :
    Himmelblau 函数的绘制
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

def Himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y** 2 - 7) ** 2


"""Himmelblau.png"""
def draw_Himmelblau():
    x = jnp.arange(-6.0, 6.1, 0.1)
    y = jnp.arange(-6.0, 6.2, 0.1)
    x, y = jnp.meshgrid(x,y)
    z = Himmelblau(x,y)

    fig = plt.figure()  # 定义三维坐标轴
    ax = plt.axes(projection = '3d')
    ax.view_init(40, -120)
    ax.plot_surface(x, y, z, cmap='rainbow')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    plt.savefig("Himmelblau.png")

"""Himmelblau_contour.png"""
def draw_Himmelblau_contour():
    x = jnp.arange(-6.5,6.5,0.1)
    y = jnp.arange(-6.5,6.5,0.1)
    x, y = jnp.meshgrid(x,y)
    z = Himmelblau(x,y)

    fig = plt.figure()
    log_levels = jnp.array([0.1, 1.8, 2.7, 3.2, 3.8, 4.4, 4.7, 5.0, 5.1849, 5.45, 5.8, 6.2, 6.5])
    CP = plt.contour(x,y,z, levels=jnp.exp(log_levels),cmap="rainbow")
    plt.clabel(CP, inline=1, fontsize=7)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.savefig("Himmelblau_contour.png")

"""Himmelblau_maxmin.png"""
def draw_Himmelblau_maxmin():
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
    plt.scatter(*maximum, s=100, c="r", marker="*", label="maxinum")
    plt.scatter(*minima1, s=80, c="b", marker="p", label="minima")
    plt.scatter(*minima2, s=80, c="b", marker="p", )
    plt.scatter(*minima3, s=80, c="b", marker="p", )
    plt.scatter(*minima4, s=80, c="b", marker="p", )

    plt.annotate(text="({:.3f},{:.3f})".format(*maximum), xy=maximum, 
                 xytext=(-25, 10), textcoords = "offset points")
    plt.annotate(text="({:.3f},{:.3f})".format(*minima1), xy=minima1, 
                 xytext=(-15,-20), textcoords = "offset points")
    plt.annotate(text="({:.3f},{:.3f})".format(*minima2), xy=minima2, 
                 xytext=(-15,-20), textcoords = "offset points")
    plt.annotate(text="({:.3f},{:.3f})".format(*minima3), xy=minima3, 
                 xytext=(-15,-20), textcoords = "offset points")
    plt.annotate(text="({:.3f},{:.3f})".format(*minima4), xy=minima4, 
                 xytext=(-15,-20), textcoords = "offset points")

    log_levels = jnp.array([2.5, 3.2, 3.8, 4.4, 4.7, 5.0, 5.1849, 5.45, 5.8, 6.2, 6.5])
    CP = plt.contour(x,y,z, levels=jnp.exp(log_levels),cmap="Blues")
    plt.legend(loc="lower right")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.savefig("Himmelblau_maxmin.png")

if __name__ == "__main__":
    draw_Himmelblau()
    draw_Himmelblau_contour()
    draw_Himmelblau_maxmin()
