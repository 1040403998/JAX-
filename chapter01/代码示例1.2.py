

"""
代码示例 1.2 :
    数值微分的误差估计
"""

# 库的引入
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# 函数的定义
def f(x):
    return x ** 0.5

def err(h, x0, fcn: Callable, fcn_prime:Callable):
    # h 为步长； x0 为计算导数的点； fcn 为待求导的函数； fcn_prime 为函数的导函数
    return (fcn(x0+h) - fcn(x0)) / h - fcn_prime(x0)

# 步长的取值
##  从10^-17 到 10^-1 （对指数）等间距取49个点，包括首尾

h_list = np.logspace(-17, -1, 49).tolist() 


# 参数的设置
x = 0.1
fcn = math.sin
fcn_prime = math.cos

# 误差的计算
h_log_math = []
err_log_math = []
for h in h_list:
    h_log_math.append(math.log10(h))
    err_log_math.append(math.log10(abs(err(h,x,fcn,fcn_prime))))

# 可视化输出
plt.plot(h_log_math, err_log_math, label = "float")
plt.legend(loc = "lower right")
plt.xlabel("log(h)")
plt.ylabel("log(err)")
plt.grid()
plt.savefig(fname = "math_err")  # 保存图片
plt.show()
