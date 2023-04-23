


"""
代码示例7.2 :
    最速下降法求解方程组
"""



import jax.numpy as jnp

# 通过最速下降算法，求解方程组 Ax = b
def steepest_gradient_descent_solver(A, b, theta_init, kmax=1000):
    theta = theta_init          # 初始化模型参数
    r = b - jnp.dot(A, theta)   # 初始化模型残差
    k = 1
    while jnp.linalg.norm(r) > 1E-8 and k < kmax:
        Ar = jnp.dot(A,r)
        alpha = jnp.dot(r, r) / jnp.dot(r, Ar) # 确定步长
        theta = theta + alpha * r              # 更新模型参数
        r = r - alpha * Ar                     # 更新模型残差
        k = k + 1                              # 更新迭代次数
    return theta

# 算法测试
A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
b = jnp.array([1.0, 1.0])
theta_init = jnp.array([-15.0, 25.0])
steepest_gradient_descent_solver(A,b,theta_init)
# >> [0.3333341  0.33333206]



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

    """ 绘制优化过程 """
    theta = theta_init          # 初始化模型参数
    r = b - jnp.dot(A, theta)   # 初始化模型残差
    k = 1

    theta_list = [theta, ]
    while jnp.linalg.norm(r) > 1E-8 and k < 1000:
        Ar = jnp.dot(A,r)
        alpha = jnp.dot(r, r) / jnp.dot(r, Ar) # 确定步长
        theta = theta + alpha * r              # 更新模型参数
        r = r - alpha * Ar                     # 更新模型残差
        k = k + 1                              # 更新迭代次数
        theta_list.append(theta)

    point_array = jnp.array(theta_list).T
    print("point_array.shape = {}".format(point_array.shape))
    plt.scatter(point_array[0][:7], point_array[1][:7], c = "red", s=20, label = r"$\theta_k$")
    plt.plot(point_array[0][:7], point_array[1][:7], c = "green", label=r"$\alpha_k \vec{p}_k$")

    """ 图像注释 """
    plt.annotate(r"$\theta_{1}=x_{init}$", xy=point_array[:,0], xytext = (-20,10), textcoords = "offset points")
    plt.annotate(r"$\theta_{2}$", xy=point_array[:,1], xytext = (-20,-10), textcoords = "offset points")
    plt.annotate(r"$\theta_{3}$", xy=point_array[:,2], xytext = (  5,  5), textcoords = "offset points")
    plt.annotate(r"$\theta_{4}$", xy=point_array[:,3], xytext = (-10,-10), textcoords = "offset points")
    plt.annotate(r"$\theta_{5}$", xy=point_array[:,4], xytext = (  0,  5), textcoords = "offset points")
    plt.annotate("...", xy=point_array[:,5], xytext = (0,-10), textcoords = "offset points")


    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.xlim((-30, 5))
    plt.ylim((-10, 30))
    plt.legend(loc="lower left")
    plt.savefig("fig7.7.png")


    """ 距离 """
    fig = plt.figure()
    point_num = 8
    theta_opt = jnp.array([1./3., 1./3.])
    index = jnp.arange(point_num)
    norm = jnp.einsum("ab,bc,ac->a", point_array.T - theta_opt,A,point_array.T - theta_opt,)[:point_num]

    eigvals = jnp.linalg.eigvals(A).real
    lambda_min = min(eigvals)
    lambda_max = max(eigvals)
    tau = jnp.log((lambda_max-lambda_min) / (lambda_max+lambda_min))
    
    x = jnp.linspace(0, point_num, 1001)
    y = jnp.exp(tau * x) * norm[0]

    equ = r"$y = (\frac{\lambda_{max} - \lambda_{min}}{\lambda_{max} + \lambda_{min}})^k$"
    plt.grid()
    plt.scatter(index, norm, c="red", label = r"$\Vert \theta_k-\theta^{*} \Vert_A$",s=20)
    plt.plot(x,y, c = "blue", label = "upper bound", linestyle="--")
                    
    plt.annotate(text="", arrowprops=dict(arrowstyle="-|>"),
                 xy=(x[150],y[150]), xytext = (x[150]+0.5, y[150]+30), fontsize = 10)
    plt.annotate(r"$\Vert \theta_k-\theta^{*} \Vert_A \leq \mu^{k} \Vert \theta_0-\theta^{*} \Vert_A$",
                 xy=(x[150]+0.5, y[150]+0.5), xytext = (3, 4), textcoords = "offset points", fontsize = 10)
    # plt.annotate(r"$\Vert \theta_k-\theta^{*} \Vert_A = e^{-k ln{2}} \Vert \theta_0-\theta^{*} \Vert_A$",
    #              xy=(x[150]+0.5, y[150]+0.5), xytext = (3, 4), textcoords = "offset points", fontsize = 10)

    plt.xlabel("k")
    plt.ylabel(r"$\Vert \theta_k-\theta^{*} \Vert_A$")
    plt.legend(loc = "upper right")
    plt.savefig("fig7.7.2.png")

    """ 对数距离 """
    fig = plt.figure()
    point_num = 8
    theta_opt = jnp.array([1./3., 1./3.])
    index = jnp.arange(point_num)
    norm = jnp.einsum("ab,bc,ac->a", point_array.T - theta_opt,A,point_array.T - theta_opt,)[:point_num]

    eigvals = jnp.linalg.eigvals(A).real
    lambda_min = min(eigvals)
    lambda_max = max(eigvals)
    tau = jnp.log((lambda_max-lambda_min) / (lambda_max+lambda_min))
    
    x = jnp.linspace(0, point_num, 1001)
    logy = jnp.log(jnp.exp(tau * x) * norm[0])

    plt.grid()
    plt.scatter(index, jnp.log(norm), c="red", label = r"$ln(\Vert \theta_k-\theta^{*} \Vert_A)$",s=20)
    plt.plot(x,logy, c = "blue", label = "upper bound", linestyle="--")
                    
    plt.annotate(text="", arrowprops=dict(arrowstyle="-|>"),
                 xy=(x[600],logy[600]), xytext = (x[600]+0.3, logy[600]+1.3), fontsize = 10)
    plt.annotate(r"$ln(\Vert \theta_k-\theta^{*} \Vert_A) = -k ln{2} + ln(\Vert \theta_0-\theta^{*} \Vert_A)$",
                 xy=(x[750]+0.5, logy[750]+2), xytext = (-120, 10), textcoords = "offset points", fontsize = 10)

    plt.xlabel("k")
    plt.ylabel(r"$ln(\Vert \theta_k-\theta^{*} \Vert_A)$")
    plt.legend(loc = "lower left")
    plt.savefig("figD.2.png")
        

if __name__ == "__main__":
    # draw_Himmelblau()
    draw_f()
    # draw_Himmelblau_maxmin()
