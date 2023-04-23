

"""
代码示例 1.3.0 :
    数值微分的程序实现
"""

from typing import Callable

# 定义函数 grad, 用于求取输入函数fun的导函数
def grad(fun: Callable, step_size=1E-5)-> Callable:
    def grad_f(x):
        return (fun(x + step_size) - fun(x)) / step_size
    return grad_f

# 定义函数 value_and_grad，它将会同时计算输入函数grad的值和导函数
def value_and_grad(fun: Callable, step_size=1E-5)-> Callable:
    def value_and_grad_f(x):
        value = fun(x)
        grad = (fun(x + step_size) - value) / step_size
        return value, grad
    return value_and_grad_f


# 测试
import math
f = math.sin
df = grad(f)
ddf = grad(df)
dddf = grad(ddf)
print(df(0.))    # 返回：0.9999999999833332
print(ddf(0.))   # 返回：-1.000000082740371e-05
print(dddf(0.))  # 返回：-0.999996752071297

