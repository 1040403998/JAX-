


"""
代码示例 2.2.1 :
    反向模式 的程序实现


    ~ and ~


代码示例 2.2.2 :
    基于反向模式的 grad 的实现
"""

import math
from abc import abstractmethod
from multiprocessing.sharedctypes import Value
from typing import Tuple, Callable, List


class Variable(object):

    __slots__ = ["value", "g"]

    _is_leaf = True         # 判断是否处于叶子节点
    _inherance: List = []   # 储存由该节点引出的节点
    
    def __init__(self, value, g = 0.):
        self.value = value
        self.g = g

    @staticmethod
    def _to_variable(obj):
        """ 将输入参数 obj 转换为 Variable 类 """
        if isinstance(obj, Variable): 
            return obj
        try:
            return Variable(obj)
        except:
            raise TypeError("Object {} is of type {}, which cannot be interpreted"
                            "as Variables".format(obj, type(obj).__name__))

    @property
    def is_leaf(self):
        return self._is_leaf

    @property    
    def ready_for_backward(self,):
        """ 一个结点在被反向传播之前，应该首先判断这个
        节点的所有父节点是否都已经完成了反向传播，因此
        需要定义这个函数来完成拓扑排序 """
        # print(self._inherance)
        for subvar in self._inherance:
            if subvar.is_leaf: continue
            assert hasattr(subvar, "_processed")
            if not subvar._processed: return False
        return True

    @abstractmethod
    def backward(*args):
        """ 继承该类的类应该定义 反向传播 时的计算方法 """
        pass

    def __str__(self):
        if isinstance(self.value, Variable):
            return str(self.value)
        return "Variable({})".format(self.value)


    def __add__(self, other):
        return Add(self, other)
    
    def __radd__(self, other):
        return Add(other, self)

    def __neg__(self):
        return Neg(self,)

    def __sub__(self, other):
        neg_other = Neg(other)
        return Add(self, neg_other)

    def __rsub__(self, other):
        neg_self = Neg(self)
        return Add(neg_self, other)
    
    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)
    
    def __pow__(self, other):
        return Pow(self, other)


class Add(Variable):

    """ 计算加法 """

    __slots__ = ["g", "_res", "_args", "_processed", "_is_leaf", "_inherance"]

    def __init__(self, v1, v2):
        """
        由两个已知的 Variable 节点 v1 和 v2, 构造出一个新的 Variable 节点.

        参数
        ----
        v1: Variable
            位于加法操作左侧的操作数
        v2: Variable
            位于加法操作右侧的操作数

        旁注
        ----
            输入的参数v1和v2, 要么属于 Variable 类，要么属于继承自Variable类的子类.

            self.g           用于储存反向传播时该点处积累的梯度
            self._res        用于储存反向传播时需要用到的参数的值
            self._args       用于标记计算图中该节点的父节点(们)
            self._processed  用于标记该节点是否已经经历了反向传播的运算

        """
        v1 = self._to_variable(v1)
        v2 = self._to_variable(v2)
        self.value = v1.value + v2.value
        self._is_leaf = False

        self.g = Variable(0.)
        self._res: Tuple = ()
        self._processed: bool = False
        self._args: Tuple[Variable] = (v1, v2, )
        
        # 更新继承关系
        self._inherance: List[Variable] = []
        v1._inherance.append(self)
        v2._inherance.append(self)

    def backward(self, ):
        # 更新父节点的梯度
        if isinstance(self._args[0], Variable):
            g = Variable(self.g.value)
            self._args[0].g += g
        if isinstance(self._args[1], Variable):
            g = Variable(self.g.value)
            self._args[1].g += g


        # 更新该节点的 _processed 参数
        self._processed = True

        return None


class Neg(Variable):

    """ 计算取反操作，即 a -> -a """

    __slots__ = ["g", "_res", "_args", "_processed", "_is_leaf", "_inherance"]

    def __init__(self, v):
        """
        参数
        ----
        v: Variable
            被取反的变量

        旁注
        ----
            输入的参数v, 要么属于 Variable 类，要么属于继承自Variable类的子类.

            self.g           用于储存反向传播时该点处积累的梯度
            self._res        用于储存反向传播时需要用到的参数的值
            self._args       用于标记计算图中该节点的父节点(们)
            self._processed  用于标记该节点是否已经经历了反向传播的运算
        """
        v = self._to_variable(v)
        self.value = -v.value
        self._is_leaf = False

        self.g = Variable(0.)
        self._res: Tuple = ()
        self._processed: bool = False
        self._args: Tuple[Variable] = (v,)

        # 更新继承关系
        self._inherance: List[Variable] = []
        v._inherance.append(self)
        
    def backward(self, ):
        # 更新父节点的梯度
        if isinstance(self._args[0], Variable):
            self._args[0].g -= self.g

        # 更新该节点的 _processed 参数
        self._processed = True

        return None


class Mul(Variable):

    """ 计算乘法 """

    __slots__ = ["g", "_res", "_args", "_processed", "_is_leaf", "_inherance"]

    def __init__(self, v1, v2):
        """
        由两个已知的 Variable 节点 v1 和 v2, 构造出一个新的 Variable 节点.

        参数
        ----
        v1: Variable
            位于乘法操作左侧的操作数
        v2: Variable
            位于乘法操作右侧的操作数

        旁注
        ----
            输入的参数v1和v2, 要么属于 Variable 类，要么属于继承自Variable类的子类.

            self.g           用于储存反向传播时该点处积累的梯度
            self._res        用于储存反向传播时需要用到的参数的值
            self._args       用于标记计算图中该节点的父节点(们)
            self._processed  用于标记该节点是否已经经历了反向传播的运算

        """
        v1 = self._to_variable(v1)
        v2 = self._to_variable(v2)
        self.value = v1.value * v2.value
        self._is_leaf = False

        self.g = Variable(0.)
        self._res: Tuple = (v1, v2)
        self._processed: bool = False
        self._args: Tuple[Variable] = (v1.value, v2.value, )

        # 更新继承关系
        self._inherance: List[Variable] = []
        v1._inherance.append(self)
        v2._inherance.append(self)

    def backward(self, ):
        # 取出更新父节点所需要的参数的值
        v1_value, v2_value = self._res

        # 为了求取高阶导数，需要重新初始化我们的v1
        v1 = Variable(v1_value)  
        v2 = Variable(v2_value)

        # 更新父节点的梯度
        if isinstance(self._args[0], Variable):
            self._args[0].g += v2 * self.g   # 更新 v1 的梯度
        if isinstance(self._args[1], Variable):
            self._args[1].g += v1 * self.g   # 更新 v2 的梯度

        # 更新该节点的 _processed 参数
        self._processed = True

        return None


class Div(Variable):

    """ 计算除法 """

    __slots__ = ["g", "_res", "_args", "_processed", "_is_leaf", "_inherance"]

    def __init__(self, numerator, denominator):
        """
        由两个已知的 Variable 节点 v1 和 v2, 构造出一个新的 Variable 节点.

        参数
        ----
        numerator: Variable
            分子(被除数)，位于操作符左侧
        denominator: Variable
            分母(除数)，位于操作符右侧

        旁注
        ----
            输入的参数 numerator 和 denominator, 要么属于 Variable 
            类，要么属于继承自Variable类的子类.

            self.g           用于储存反向传播时该点处积累的梯度
            self._res        用于储存反向传播时需要用到的参数的值
            self._args       用于标记计算图中该节点的父节点(们)
            self._processed  用于标记该节点是否已经经历了反向传播的运算

        """
        numerator = self._to_variable(numerator)
        denominator = self._to_variable(denominator)
        self.value = numerator.value / denominator.value
        self._is_leaf = False

        self.g = Variable(0.)
        self._res: Tuple = (numerator, denominator)
        self._processed: bool = False
        self._args: Tuple[Variable] = (numerator.value, denominator.value, )

        # 更新继承关系
        self._inherance: List[Variable] = []
        numerator._inherance.append(self)
        denominator._inherance.append(self)

    def backward(self, ):
        # 取出更新父节点所需要的参数的值
        numer_value, denom_value = self._res

        # 为了求取高阶导数，需要重新初始化我们的numer和denom
        numer = Variable(numer_value)
        denom = Variable(denom_value)

        # 更新父节点的梯度
        if isinstance(self._args[0], Variable):
            self._args[0].g += self.g / denom                     # 更新 分子numerator 的梯度
        if isinstance(self._args[1], Variable):
            self._args[1].g += - numer / denom**2 * self.g   # 更新 分母denominator 的梯度

        # 更新该节点的 _processed 参数
        self._processed = True
        return None
        

class Pow(Variable):

    """ 计算乘方 """

    __slots__ = ["g", "_res", "_args", "_processed", "_is_leaf", "_inherance"]

    def __init__(self, base, pow):
        """
        由两个已知的 Variable 节点 base 和 pow, 构造出一个新的 Variable 节点，描述 base ** pow.

        参数
        ----
        base: Variable
            乘方运算的 底数
        pow: Variable
            乘方运算的 幂

        旁注
        ----
            输入的参数 base 和 pow, 要么属于 Variable 类，要么属于继承自Variable类的子类.

            self.g           用于储存反向传播时该点处积累的梯度
            self._res        用于储存反向传播时需要用到的参数的值
            self._args       用于标记计算图中该节点的父节点(们)
            self._processed  用于标记该节点是否已经经历了反向传播的运算

        """
        base = self._to_variable(base)
        pow = self._to_variable(pow)
        self.value = base.value ** pow.value
        self._is_leaf = False

        self.g = Variable(0.)
        self._res: Tuple = (base.value, pow.value)
        self._processed: bool = False
        self._args: Tuple[Variable] = (base, pow)
        
        # 更新继承关系
        self._inherance: List[Variable] = []
        pow._inherance.append(self)
        base._inherance.append(self)

    def backward(self, ):
        # 取出更新父节点所需要的参数的值
        base_value, pow_value = self._res
        
        # 为了求取高阶导数，需要重新初始化我们的v1
        base =  Variable(base_value)
        pow = Variable(pow_value)
        self_new = Variable(self.value)

        # 更新父节点的梯度 x^y  = e^(ylnx)
        if isinstance(self._args[0], Variable):
            self._args[0].g += self.g * pow * base ** (pow - 1.)  # 更新底数 y*x^(y-1)
        if isinstance(self._args[1], Variable):
            self._args[1].g += self.g * self_new * Log(base)  # 更新指数 y*x^(y-1)

        # 更新该节点的 _processed 参数
        self._processed = True

        return None
    
class Log(Variable):

    """ 计算对数 """

    __slots__ = ["g", "_res", "_args", "_processed", "_is_leaf", "_inherance"]

    def __init__(self, x):
        """
        由已知的输入的 Variable 节点x, 构造出一个新的 Variable 节点，描述 log(x)

        参数
        ----
        x: Variable
            log 操作作用的对象

        旁注
        ----
            输入的参数 x, 要么属于 Variable 类，要么属于继承自Variable类的子类.

            self.g           用于储存反向传播时该点处积累的梯度
            self._res        用于储存反向传播时需要用到的参数的值
            self._args       用于标记计算图中该节点的父节点(们)
            self._processed  用于标记该节点是否已经经历了反向传播的运算

        """

        x = self._to_variable(x)
        self.value = math.log(x.value)
        self._is_leaf = False

        self.g = Variable(0.)
        self._res: Tuple = (x.value, )
        self._processed: bool = False
        self._args: Tuple[Variable] = (x,)
        
        # 更新继承关系
        self._inherance: List[Variable] = []
        x._inherance.append(self)

    def backward(self, ):
        # 取出更新父节点所需要的参数的值
        x_value = self._res

        # 为了求取高阶导数，需要重新初始化我们的v1
        x =  Variable(x_value)

        # 更新父节点的梯度 
        if isinstance(self._args[0], Variable):
            self._args[0].g += Variable(1.) / x * self.g    # 更新底数 y*x^(y-1)

        # 更新该节点的 _processed 参数
        self._processed = True

        return None


## 以拓扑序进行反向传播
def backward_pass(value: Variable):

    assert isinstance(value, Variable)

    arg_list: List[Variable] = []
    arg_list.append(value)
    
    n = 0
    while len(arg_list) != 0:
        # 从头开始遍历列表，找出可以进行反向传播的元素，进行反向传播
        for idx, arg in enumerate(arg_list):
            n += 1
            if isinstance(arg, Variable):
                print("[{}]:\n\t idx = {}, \n\t arg = {} \n\t arg_list = {}\
                    \n\t is_leaf = {} \n\t ready_for_backward = {}".\
                    format(n, idx, arg, arg_list, arg.is_leaf, arg.ready_for_backward))
            else:
                print("[{}]:\n\t idx = {}, \n\t arg = {} \n\t arg_list = {}"\
                    .format(n, idx, arg, arg_list))

            length_changed = False   # 用于判断循环后列表长度有无发生变化
            if not isinstance(arg, Variable):
                arg_list.pop(idx)
                length_changed = True
                continue

            if arg.is_leaf: 
                arg_list.pop(idx)
                length_changed = True
                continue

            if arg.ready_for_backward:
                arg_list.extend(arg._args) # 将与 arg 相邻的节点放入列表
                arg_list.pop(idx)  # 将 arg 从arg_list中取出
                arg.backward()     # 对 arg 参数进行反向传播
                length_changed = True

        # 调试用
        if not length_changed:
            if isinstance(arg, Variable):
                print("[{}]:\n\t idx = {}, \n\t arg = {} \n\t arg_list = {}\
                    \n\t is_leaf = {} \n\t ready_for_backward = {}".\
                    format(n, idx, arg, arg_list, arg.is_leaf, arg.ready_for_backward))
            else:
                print("[{}]:\n\t idx = {}, \n\t arg = {} \n\t arg_list = {}"\
                    .format(n, idx, arg, arg_list))

            print("\t arg._inherance = {}".format(arg._inherance))
            for subvar in arg._inherance:
                print("\t arg_isleaf = {}".format(subvar.is_leaf))
            raise ValueError("Some problem here...")



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

        # 将函数的输入转化为 Variable 类
        args_new: List[Variable] = []
        for arg in args:
            if not isinstance(arg, Variable):
                arg_new = Variable._to_variable(arg)
            else:
                arg_new = arg
            args_new.append(arg_new)

        # 计算函数的值和导函数
        value = fun(*args_new)
        value.g = Variable(1.)
        backward_pass(value)
        g = args_new[argnum].g
        
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
    # return (x + y) ** 2
    return x ** 3

x = 5.
y = 1.
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
# print(f(x,y))       # >>  9.0

"""  第一阶  """
print(df1 (x,y))    # >>  Variable(6.0)
# print(df2 (x,y))    # >>  Variable(6.0)

"""  第二阶  """
print(ddf11(x,y))   # >>  Variable(0.0)
# print(ddf12(x,y))   # >>  Variable(0.0)
# print(ddf21(x,y))   # >>  Variable(0.0)
# print(ddf22(x,y))   # >>  Variable(0.0)



