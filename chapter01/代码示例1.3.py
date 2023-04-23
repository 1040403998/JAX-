

"""
代码示例 1.3 :
    数值微分的程序实现(简单函数)
"""


# 库的引入
import math
from typing import Callable

def value_and_grad(fun: Callable, step_size=1E-5
                   )-> Callable:
    '''
    构造一个方程，它能够同时计算函数 fun 的值和它的梯度
       fun: 被微分的函数。它的输入返回值需要为一个数（而非数组）；
       step_size: 数值微分所特有，用于描述微分之中所选取的步长；

    返回：
       一个和fun具有相同输入结构的函数，这个函数能够同时计算fun的值和它的导函数
    '''
    def value_and_grad_f(*arg):
        # 输入检查
        if len(arg) != 1:
            raise TypeError(f"函数仅允许有一个变量的输入, 但收到了{len(arg)}个")
        x = arg[0]

        # 计算函数的值和导函数
        value = fun(x)
        grad = (fun(x + step_size)-fun(x)) / step_size
        return value, grad
    # 将函数value_and_grad_f返回
    return value_and_grad_f

def grad(fun: Callable, step_size=1E-5)-> Callable:
    '''
    构造一个方程，它仅计算函数 fun 的导数
       fun: 被微分的函数。它的输入返回值需要为一个数（而非数组）；
       step_size: 数值微分所特有，用于描述微分之中所选取的步长；

    返回：
       一个和fun具有相同输入结构的函数，这个函数能够计算函数fun导函数
    '''
    value_and_grad_f = value_and_grad(fun, step_size)
    def grad_f(*arg):
        # 仅仅返回导数
        _, g = value_and_grad_f(*arg)
        return g
    # 将函数value_and_grad_f返回
    return grad_f

# 测试
f = math.sin
df = grad(f)
ddf = grad(df)
dddf = grad(ddf)
print(df(0.))
print(ddf(0.))
print(dddf(0.))



