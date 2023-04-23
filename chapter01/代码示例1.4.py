

"""
代码示例 1.4 :
    数值微分的程序实现(任意函数)
"""

import numpy as np
from copy import deepcopy
from typing import Callable, Union, Sequence

def value_and_grad(fun: Callable, argnums: Union[int, Sequence[int]] = (0,),
                   has_aux: bool = False, step_size=1E-5,
                   )-> Callable:
    '''
    构造一个方程，它能够同时计算函数 fun 的值和它的梯度
        fun: 被微分的函数。需要被微分的位置由参数argnums指定， 
            而函数fun返回的第一个值需要为一个数（而非数组），
            如果函数fun有另外的输出, 则需令has_aux参数为True；
        argnums: 可选参数，可以为整数int或者一个整数的序列, 用于指定微分的对象；
        has_aux: 可选参数，bool类型，用于显式的声明函数fun是否存在除整数以外的输出；
        step_size: 数值微分所特有，用于描述微分之中所选取的步长；

    返回：
       一个和fun具有相同输入结构的函数，这个函数能够同时计算fun的值和指定位置的导函数
    '''
    if isinstance(argnums, int): 
        argnums = (argnums,)

    def value_and_grad_f(*args):
        # 输入检查
        max_argnum = argnums if isinstance(argnums, int) else max(argnums)
        if max_argnum >= len(args):
            raise TypeError(f"对参数 argnums = {argnums}微分需要至少 "
                            f"{max_argnum+1}个位置的参数作为变量被传入，"
                            f"但只收到了{len(args)}个参数")

        # 构造求导所需的输入
        diff_arg_list = []
        for num in argnums:
            temp_args = deepcopy(list(args))
            temp_args[num] += step_size * np.ones_like(args[num], dtype=np.float64)
            diff_arg_list.append(temp_args)

        # 计算函数的值和导函数
        if not has_aux:
            value = fun(*args)
            g = [(fun(*diff_args)-value) / step_size for diff_args in diff_arg_list]
        else:
            value, aux = fun(*args)
            g = [(fun(*diff_args)[0]-value) / step_size for diff_args in diff_arg_list]
        
        # 程序输出
        g = g[0] if len(argnums)==1 else tuple(g)
        if not has_aux:
            return value, g
        else:
            return (value, aux), g

    # 将函数value_and_grad_f返回
    return value_and_grad_f

def grad(fun: Callable, argnums: Union[int, Sequence[int]] = (0,),
         has_aux: bool = False, step_size=1E-5,
         )-> Callable:
    '''
    构造一个方程，它仅计算函数 fun 的梯度
        fun: 被微分的函数。需要被微分的位置由参数argnums指定， 
            而函数fun返回的第一个值需要为一个数（而非数组），
            如果函数fun有另外的输出, 则需令has_aux参数为True；
        argnums: 可选参数，可以为整数int或者一个整数的序列, 用于指定微分的对象；
        has_aux: 可选参数，bool类型，用于显式的声明函数fun是否存在除整数以外的输出；
        step_size: 数值微分所特有，用于描述微分之中所选取的步长；

    返回：
       一个和fun具有相同输入结构的函数，这个函数能够计算函数fun的梯度
    '''
    value_and_grad_f = value_and_grad(fun=fun, argnums=argnums,
                                      has_aux=has_aux ,step_size=step_size)
    
    def grad_f(*arg):
        # 仅仅返回导数
        _, g = value_and_grad_f(*arg)
        return g

    def grad_f_aux(*arg):
        # 返回导数，以及原函数输出的其他参数
        (_, aux), g = value_and_grad_f(*arg)
        return g, aux
    return grad_f_aux if has_aux else grad_f


# 测试
def f(x,y):
    aux = "function called"
    return np.sin(x+2*y), aux

x = np.array([0.,0.,np.pi])
y = np.array([0.,np.pi,0.])
df1  = grad(f, argnums=0,      step_size=1E-5, has_aux=True)
df2  = grad(f, argnums=1,      step_size=1E-5, has_aux=True)
df12 = grad(f, argnums=(0,1),  step_size=1E-5, has_aux=True)

print(f(x,y))
print(df1 (x,y))
print(df2 (x,y))
print(df12(x,y))