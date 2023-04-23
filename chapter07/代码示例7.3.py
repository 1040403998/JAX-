


"""
代码示例7.3 :
    共轭梯度算法求解方程组
"""

import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

# 通过共轭梯度算法，求解方程组 Ax = b
def conjugate_gradient_descent_solver(A, b, x_init):
    theta = x_init              # 初始化模型参数
    r = b - jnp.dot(A, theta)   # 初始化模型残差
    p = r                       # 初始化下降方向
    while jnp.linalg.norm(r) > 1E-8:
        Ap= jnp.dot(A,p)
        alpha = jnp.dot(r, r) / jnp.dot(p, Ap)       # 确定步长
        theta = theta + alpha * p                    # 更新模型参数
        r_old = r                                    # 保存上一步残差向量 
        r = r - alpha * Ap                           # 更新模型残差
        beta = jnp.dot(r, r) / jnp.dot(r_old, r_old) # 更新中间变量
        p = r + beta * p                             # 更新下降方向
    return theta

# 算法测试
A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
b = jnp.array([1.0, 1.0])
theta_init = jnp.array([-15.0, 25.0])
print(conjugate_gradient_descent_solver(A,b,theta_init))
# >> [0.33333333  0.33333333]


""" 可视化 """
import matplotlib.pyplot as plt

# 根据上述矩阵 A 定义优化函数
def f(x, y):
    return x**2 + x*y + y**2 - x - y


def draw_f():

    """ 绘制函数的轮廓 """
    x = jnp.arange(-30.5,10.5,0.1)
    y = jnp.arange(-10.5,30.5,0.1)
    x, y = jnp.meshgrid(x,y)
    z = f(x,y)

    fig = plt.figure()
    CP = plt.contour(x,y,z, levels=jnp.exp(jnp.linspace(0, 6.5, 21)),cmap="Blues", zorder=-10)
    plt.clabel(CP, inline=1, fontsize=7)

    """ 绘制共轭梯度算法的优化过程 """
    theta = theta_init          # 初始化模型参数
    r = b - jnp.dot(A, theta)   # 初始化模型残差
    p = r                       # 初始化下降方向

    theta_list0 = [theta, ]
    while jnp.linalg.norm(r) > 1E-8:
        Ap= jnp.dot(A,p)
        alpha = jnp.dot(r, r) / jnp.dot(p, Ap)       # 确定步长
        theta = theta + alpha * p                   # 更新模型参数
        r_old = r                                    # 保存上一步残差向量 
        r = r - alpha * Ap                           # 更新模型残差
        beta = jnp.dot(r, r) / jnp.dot(r_old, r_old) # 更新中间变量
        p = r + beta * p                             # 更新下降方向
        theta_list0.append(theta)


    point_array0 = jnp.array(theta_list0).T
    print("point_array.shape = {}".format(point_array0.shape))
    plt.scatter(point_array0[0], point_array0[1], c = "red", s=20)
    plt.plot(point_array0[0], point_array0[1], c = "green", label="conjugate gradient")


    """ 绘制最速下降法的优化过程 """
    theta = theta_init          # 初始化模型参数
    r = b - jnp.dot(A, theta)   # 初始化模型残差
    k = 1

    theta_list1 = [theta, ]
    while jnp.linalg.norm(r) > 1E-8 and k < 1000:
        Ar = jnp.dot(A,r)
        alpha = jnp.dot(r, r) / jnp.dot(r, Ar) # 确定步长
        theta = theta + alpha * r              # 更新模型参数
        r = r - alpha * Ar                     # 更新模型残差
        k = k + 1                              # 更新迭代次数
        theta_list1.append(theta)

    point_array1 = jnp.array(theta_list1).T
    print("point_array.shape = {}".format(point_array1.shape))
    # plt.scatter(point_array1[0][:7], point_array1[1][:7], c = "orange", s=20, label = r"$\theta_k$")
    plt.plot(point_array1[0][:7], point_array1[1][:7], c = "orange", label="steepest descent", linestyle="--")

    """ 图像注释 """
    plt.annotate(r"$\theta_{1}=x_{init}$", xy=point_array0[:,0], xytext = (-20,10), textcoords = "offset points")
    plt.annotate(r"$\theta_{2}$", xy=point_array0[:,1], xytext = (-20,-10), textcoords = "offset points")
    plt.annotate(r"$\theta_{3} = (0.3333, 0.3333)$", xy=point_array0[:,2], xytext = (-60,-20), textcoords = "offset points")



    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.xlim((-30, 5))
    plt.ylim((-10, 30))
    plt.legend(loc="lower left")
    plt.savefig("fig7.8.png")
        

if __name__ == "__main__":
    # draw_Himmelblau()
    draw_f()
    # draw_Himmelblau_maxmin()
