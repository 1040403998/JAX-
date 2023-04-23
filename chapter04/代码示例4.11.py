


"""
代码示例 4.11 :
    使用多项式 Pn(x) 拟合数据
"""


import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, simplify, diff, solve


def fit(n, linestyle = "-"):
    # 初始化n+1个变量
    a_tuple = symbols("a:{}".format(n+1))

    # 定义用于拟合的多项式
    def Pn(x):
        res = 0.
        for i in range(n+1):
            res += a_tuple[i] * x ** i
        return res

    # 定义损失函数
    def loss_fun(sample_num, x_sample, y_sample):
        res = 0.
        for i in range(sample_num):
            x, y = x_sample[i], y_sample[i]
            res += (Pn(x) - y) ** 2
            res = simplify(res)
        return res
    loss = loss_fun(10, x_sample=x_array, y_sample=y_array)
    
    # # L2 正则化
    # beta = 0.01
    # for i in range(n+1):
    #     loss += a_tuple[i]**2 * beta

    # 求解方程组
    equations = []
    for i in range(n+1):
        equations.append(diff(loss, a_tuple[i]))
    solution = solve(equations, list(a_tuple))
    print(solution)

    # 作图
    xarr = np.linspace(0.3,10.3,101) 
    yarr = np.array([Pn(float(x)).evalf(subs = solution) for x in xarr])
    plt.plot(xarr, yarr, label = "n = {}".format(n), linestyle=linestyle.get("{}".format(n), "-"))

x_array = np.arange(10) + 1
y_array = np.array([49.300, 53.070, 58.210, 62.540,\
                    67.080, 71.423, 75.970, 80.340, 84.705, 89.190],dtype=np.float128)  
plt.xlabel("x") 
plt.ylabel("y")
  
linestyle = {"0":"-", "1":"-", "5": ":", "10": "dashdot", "15":"--"}  
for n in [0, 1, 5, 10, 15]:  
    print("fitting...", n)
    fit(n, linestyle)  
  
plt.scatter(x_array, y_array, s = 20, marker = "p")  
plt.grid("-")  
plt.legend(loc = "lower center")  
plt.savefig("fitting_line.png")  