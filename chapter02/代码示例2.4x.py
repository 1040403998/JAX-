


"""
代码示例 2.5 :
    前向模式高阶导数的尝试（未完成）
"""


"""   代码示例 2.1.1   """
import math
from typing import Callable, List

class Variable(object):
    _is_leaf = True
    def __init__(self, value, dot=0):
        self.value = value
        self.dot = dot

    def __add__(self, other):
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        res = self.to_variable(self.value + other.value)
        res.dot = self.to_variable(self.dot + other.dot)
        res._is_leaf = False
        return res

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        res = self.to_variable(self.value - other.value)
        res.dot = self.to_variable(self.dot - other.dot)
        res._is_leaf = False
        return res

    def __rsub__(self, other):
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        res = self.to_variable(other.value - self.value)
        res.dot = self.to_variable(other.dot - self.dot)
        res._is_leaf = False
        return res


    def __mul__(self, other):
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        res = self.to_variable(self.value * other.value)
        res.dot = self.to_variable(other.value * self.dot + self.value * other.dot)
        res._is_leaf = False
        return res

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        if other.value == 0:
            raise ZeroDivisionError("division by zero")

        res = Variable(self.value / other.value)
        res.dot = Variable(1. / other.value * self.dot \
                  - 1 / (other.value ** 2) * self.value * other.dot)
        res._is_leaf = False
        return res

    def __pow__(self, other):
        if not isinstance(other, Variable):
            other = self.to_variable(other)

        # 指数和底数出现0的情况
        if other.value == 0 or self.value == 0:
            if not (self.value == 0 and other.value ==0):
                res = self.to_variable(1.)
            else:
                raise ValueError("0^0 occurred during calculation")
        
        # 指数为整数的情况
        elif int(other.value) == other.value and other._is_leaf:
            res = self.to_variable(self.value ** other.value)
            res.dot = self.to_variable(other.value * self.value ** (other.value - 1) * self.dot)
        
        # 一般情况
        else:
            if self.value < 0:
                raise ValueError("Can't take the power of a negative number currently,"
                                 " may be implemented later")
            res = self.to_variable(self.value ** other.value)
            res.dot = self.to_variable(other.value * self.value ** (other.value - 1) * self.dot \
                                      + self.value ** other.value * math.log(self.value) * other.dot)
        res._is_leaf = False
        return res

    def __str__(self):
        if isinstance(self.value, Variable):
            return str(self.value)
        return "Variable({})".format(self.value)


    @staticmethod
    def to_variable(obj):
        if isinstance(obj, Variable): 
            return obj
        try:
            return Variable(obj)
        except:
            raise TypeError("Object {} is of type {}, which can not be "
                            "interpreted as Variables".format(obj, type(obj)))



"""   代码示例 2.1.2   """

def value_and_grad(fun: Callable, 
                argnum: int = 0,)-> Callable:
    '''
    构造一个方程，它能够同时计算函数 fun 的值和它的梯度
        fun: 被微分的函数。需要被微分的位置由参数argnums指定, 函数的返回只能为一个数（不能为数组）
        argnum: 可选参数，只能为整数, 用于指定微分的对象；不指定则默认对第一个参数求导
    
    返回:
       一个和fun具有相同输入结构的函数，这个函数能够同时计算fun的值和指定位置的导函数
    '''

    def value_and_grad_f(*args):
        # 输入检查
        if argnum >= len(args):
            raise TypeError(f"对参数 argnums = {argnum}微分需要至少 "
                            f"{argnum+1}个位置的参数作为变量被传入，"
                            f"但只收到了{len(args)}个参数")

        # 构造求导所需的输入
        args_new: List[Variable] = []
        for arg in args:
            if not isinstance(arg, Variable):
                arg_new = Variable.to_variable(arg)
                arg_new.dot = 0.
            else:
                arg_new = arg
            
            args_new.append(arg_new)
        
        # 将待求导对象的 dot 值置为 1, 其余置为 0
        args_new[argnum].dot = 1.

        # 计算函数的值和导函数
        value = fun(*args_new)
        g = value.dot
        
        # 程序输出
        return value, g

    # 将函数value_and_grad_f返回
    return value_and_grad_f

def grad(fun: Callable, 
         argnum: int = 0,)-> Callable:
    '''
    构造一个方程，它仅计算函数 fun 的梯度
        fun: 被微分的函数。需要被微分的位置由参数argnums指定, 函数的返回只能为一个数（不能为数组）
        argnums: 可选参数，只能为整数, 用于指定微分的对象；不指定则默认对第一个参数求导

    返回:
       一个和fun具有相同输入结构的函数，这个函数能够计算函数fun的梯度
    '''
    value_and_grad_f = value_and_grad(fun=fun, argnum=argnum)
    
    def grad_f(*args):
        # 仅仅返回导数
        # print([str(arg) for arg in args])
        # print([str(arg.dot) for arg in args])
        _, g = value_and_grad_f(*args)
        return g

    return grad_f


"""   测试   """

def f(x,y):
    return (x + y) ** 2

x = 1.
y = 2.
# one = Variable(1, dot=1)
# x = Variable(2., dot=1)
# y = Variable(2.,)

df1  = grad(f, argnum=0,   )
df2  = grad(f, argnum=1,   )
ddf11 = grad(df1, argnum=0,)
ddf12 = grad(df1, argnum=1,)
ddf21 = grad(df2, argnum=0,)
ddf22 = grad(df2, argnum=1,)

"""  第零阶  """
print(f(x,y))       # >>  9.0

"""  第一阶  """
print(df1 (x,y))    # >>  Variable(6.0)
print(df2 (x,y))    # >>  Variable(6.0)

"""  第二阶  """
print(ddf11(x,y))   # >>  Variable(0.0)
print(ddf12(x,y))   # >>  Variable(0.0)
print(ddf21(x,y))   # >>  Variable(0.0)
print(ddf22(x,y))   # >>  Variable(0.0)


