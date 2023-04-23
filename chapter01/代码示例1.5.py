


"""
代码示例 1.5 :
    符号微分的简单实现
"""

class Expr(object):
    _operand_name = None

    def __init__(self, *args):
        for arg in args:
            if isinstance(arg, (int, float)):
                arg = Variable(str(arg))
        self._args = args

        """ 测试 """
        # print([str(item) for item in args], self._operand_name)

    # 加法
    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    # 减法
    def __sub__(self, other):
        return Sub(self, other)
    
    def __rsub__(self, other):
        return Sub(other, self)

    # 乘法
    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    # 除法
    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    # 乘方
    def __pow__(self, other):
        return Pow(self, other)
    
    def __rpow__(self, other):
        return Pow(other, self)
    
    def __str__(self):
        terms = [str(item) for item in self._args]
        operand = self._operand_name
        return "({})".format(operand.join(terms))


""" 数字 0 的单例 """
class Zero(Expr):
    _instance = None
    def __new__(cls, *args):
        if Zero._instance == None:
            obj = object.__new__(cls)
            obj.name = "0"
            return obj
        else:
            return Zero._instance

    def __str__(self):
        return "0"


""" 数字 1 的单例 """
class One(Expr):
    _instance = None
    def __new__(cls, *args):
        if One._instance == None:
            obj = object.__new__(cls)
            obj.name = "1"
            return obj
        else:
            return One._instance

    def __str__(self):
        return "1"

VARIABLE_NAME = "Variable"

class Variable(Expr):

    _operand_name = VARIABLE_NAME
    __slots__ = ("name",)

    def __init__(self, name: str):
        """ 构造函数 """
        try:
            assert(isinstance(name, str))
        except:
            raise TypeError("name parameters should be string, \
                  get type {} instead".format(type(name)))
        finally:
            self.name = name

    def __str__(self):
        return self.name

    def diff(self, var):
        if self.name == var.name:
            return One()
        else:
            return Zero()


class Add(Expr):
    _operand_name = " + "

    # def __str__(self):
    #     terms = [str(item) for item in self._args]
    #     return "({})".format(" + ".join(terms))

    def diff(self, var):
        terms = self._args
        terms_after_diff = [item.diff(var) for item in terms]
        return Add(*terms_after_diff)

class Sub(Expr):
    _operand_name = " - "

    def diff(self, var):
        terms = self._args
        terms_after_diff = [item.diff(var) for item in terms]
        return Sub(*terms_after_diff)    


class Mul(Expr):
    _operand_name = " * "

    def diff(self, var):
        terms = self._args
        if len(terms) != 2:
            raise ValueError("Mul operation takes only 2 parameters")
        terms_after_diff = [item.diff(var) for item in terms]

        return Add(*terms_after_diff)


class Pow(Expr):
    _operand_name = " ** "

    def diff(self, var):
        base = self._args[0]   # 底数
        pow = self._args[1]    # 幂
        dbase = base.diff(var)
        dpow = pow.diff(var)
        if isinstance(base, (int, float, np.ndarray)):
            return self * dpow * Log(base)
        elif isinstance(pow, (int, float, np.ndarray)):
            return self * dbase * pow / base
        else:
            return self * (dpow * Log(base) + dbase * pow / base)


class Div(Expr):
    _operand_name = " / "
    
    def diff(self, var):
        numer = self._args[0]     # 分子（被除数） numerator
        denom = self._args[1]     # 分母（除数）   denomenator
        d_numer = numer.diff(var)
        d_denom = denom.diff(var)
        if isinstance(numer, (int, float, np.ndarray)):
            # 如果分子是常数
            return Zero() - d_denom * numer / denom ** 2
        elif isinstance(denom, (int, float, np.ndarray)):
            # 如果分母是常数
            return d_numer / denom
        else:
            return d_numer / denom - d_denom * numer / denom ** 2


class Log(Expr):
    def __str__(self):
        return f"log({str(self._args[0])})"

    def diff(self, var):
        return self._args[0].diff(var) / self._args[0]

def diff(function, var):
    return function.diff(var)

def grad(fun, argnum = 0):
    '''
    构造一个方程，它仅计算函数 fun 的梯度
        fun: 被微分的函数。需要被微分的位置由参数 argnum 指定, 函数的返回只能为一个数（不能为数组）
        argnums: 可选参数，只能为整数, 用于指定微分的对象；不指定则默认对第一个参数求导

    返回:
       一个和fun具有相同输入结构的函数，这个函数能够计算函数fun的梯度
    '''
    def df(*args):
        namespace = []
        for i in range(len(args)):
            namespace.append("arg" + str(i))
        varlist = [Variable(name) for name in namespace]
        expr = str(diff(fun(*varlist), varlist[argnum]))
        for i in range(len(args)):
            exec("{} = {}".format(namespace[i], args[i]))
        return eval(expr)
    return df




"""  测试1 """

x = Variable("x")
y = Variable("y")

def f(x,y):
    return x + y**x

expr = str(diff(f(x,y), x))
print(expr)  
# >> (1 + ((y ** x) * ((1 * log(y)) + ((0 * x) / y))))

df = grad(f)
print(df(1.0, 2.0))
# >> 2.386294361119891

"""  测试2 """
def df(x, y):
    return eval(expr)

import numpy as np
import matplotlib.pyplot as plt
log = np.log


x_sample = np.linspace(0.1, 1, 901)
fx = f(x_sample, 2.)
dfx = df(x_sample, 2)
plt.plot(x_sample, fx, label = "fx")
plt.plot(x_sample, dfx, label = "dfx")
plt.legend(loc = "lower right")
plt.grid("-")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()



# print(Log(x) * y**2 + x/3)
# print(diff(Log(x ** y), x))