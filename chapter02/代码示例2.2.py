


"""
代码示例 2.2 :
    前向模式 的程序实现

"""


import math


class Variable(object):

    _is_leaf = True

    def __init__(self, value, dot=0):
        assert isinstance(value, (int, float, Variable))
        self.value = value
        self.dot = dot

    @staticmethod
    def to_variable(obj):
        if isinstance(obj, Variable): 
            return obj
        try:
            return Variable(obj)
        except:
            raise TypeError("Object {} is of type {}, which can not be interpreted "
                            "as Variables".format(type(obj).__name__, type(obj)))

    # 加法的重载
    def __add__(self, other):
        """ self * other """
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        res = self.to_variable(self.value + other.value)
        res.dot = self.dot + other.dot
        res._is_leaf = False
        return res

    def __radd__(self, other):
        """ other * self """
        return self.__add__(other)

    # 乘法的重载
    def __mul__(self, other):
        """ self * other"""
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        res = self.to_variable(self.value * other.value)
        res.dot = other.value * self.dot + self.value * other.dot
        res._is_leaf = False
        return res

    def __rmul__(self, other):
        """ other * self """
        return self.__mul__(other)

    # 取反操作的重载
    def __neg__(self):
        """ - self """
        self.value = - self.value
        self.dot = -self.dot
        return self

    # 减法的重载
    def __sub__(self, other):
        """ self - other """
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        other = - other  # 这里将用到重载的 __neg__
        return self.__add__(other)

    def __rsub__(self, other):
        """ other - self """
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        self = -self    # 这里将用到重载的 __neg__
        return self.__add__(other)

    # 除法的重载
    def __truediv__(self, other):
        """ self / other """
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        if other.value == 0:
            raise ZeroDivisionError("division by zero")

        res = Variable(self.value / other.value)
        res.dot = 1. / other.value * self.dot \
                  - 1 / (other.value ** 2) * self.value * other.dot
        res._is_leaf = False
        return res
    
    def __rtruediv__(self, other):
        """ other / self """
        if not isinstance(other, Variable):
            other = self.to_variable(other)
        if self.value == 0:
            raise ZeroDivisionError("division by zero")

        res = Variable(other.value / self.value)
        res.dot = 1. / self.value * other.dot \
                  - 1 / (self.value ** 2) * other.value * self.dot
        res._is_leaf = False
        return res

    # 乘方的重载
    def __pow__(self, other):
        """ self ** other"""
        if not isinstance(other, Variable):
            other = self.to_variable(other)

        # 指数和底数出现0的情况
        if other.value == 0 or self.value == 0:
            if self.value == 0 and other.value ==0:
                raise ValueError("0^0 occurred during calculation.")
            elif self.value == 0:  # 0^x
                res = self.to_variable(0.)
            elif other.value == 0: # x^0
                res = other.to_variable(1.)
            else:
                raise ValueError(" This Error should never have occurred.")
        
        # 指数为整数的情况
        elif int(other.value) == other.value and other._is_leaf:
            res = self.to_variable(self.value ** other.value)
            res.dot = other.value * self.value ** (other.value - 1) * self.dot
        
        # 一般情况
        else:
            if self.value < 0:
                raise ValueError("Can't take the power of a negative number currently,"
                                 " may be implemented later")
            res = self.to_variable(self.value ** other.value)
            res.dot = other.value * self.value ** (other.value - 1) * self.dot \
                    + self.value ** other.value * math.log(self.value) * other.dot
        res._is_leaf = False
        return res

    def __rpow__(self, other):
        """ other ** self """
        if not isinstance(other, Variable):
            other = self.to_variable(other)

        # 指数和底数出现0的情况
        if other.value == 0 or self.value == 0:
            if self.value == 0 and other.value ==0:
                raise ValueError("0^0 occurred during calculation.")
            elif other.value == 0:  # 0^x
                res = self.to_variable(0.)
            elif self.value == 0:   # x^0
                res = other.to_variable(1.)
            else:
                raise ValueError(" This Error should never have occurred.")
        
        # 一般情况
        else:
            if other.value < 0:
                raise ValueError("Can't take the power of a negative number currently,"
                                 " may be implemented later")
            res = self.to_variable(other.value ** self.value)
            res.dot = self.value * other.value ** (self.value - 1) * other.dot \
                    + other.value ** self.value * math.log(other.value) * self.dot
        res._is_leaf = False
        return res

    def __str__(self):
        if isinstance(self.value, Variable):
            return str(self.value)
        return "Variable({})".format(self.value)

