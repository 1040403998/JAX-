


"""
代码示例 3.17
	Himmelblau 函数的绘制 及 其他

"""


import jax.numpy as jnp


"""   语义广播    """
ls = [1,2,3]
print(ls * 2)             # [1, 2, 3, 1, 2, 3]
print(jnp.array(ls) * 2)  # [2 4 6]

a = jnp.zeros((8,6,5))
b = jnp.zeros((1,5))
jnp.can_cast(b, a)  # True

# 外积
a = jnp.array([0.0, 10.0, 20.0, 30.0])
b = jnp.array([1.0, 2.0, 3.0])
print(a[:, jnp.newaxis] + b)
# [[ 1.  2.  3.]
#  [11. 12. 13.]
#  [21. 22. 23.]
#  [31. 32. 33.]]


"""  代码示例3.17  Himmelblau 函数的绘制"""

import jax.numpy as jnp

def Himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y** 2 - 7) ** 2

x = jnp.arange(-6,6,0.1)
y = jnp.arange(-6,6,0.1)
x, y = jnp.meshgrid(x,y)
z = Himmelblau(x,y)

import matplotlib.pyplot as plt

fig = plt.figure()  # 定义三维坐标轴
ax = plt.axes(projection = '3d')
ax.plot_surface(x, y, z, cmap='rainbow')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.savefig("Himmelblau.png")

