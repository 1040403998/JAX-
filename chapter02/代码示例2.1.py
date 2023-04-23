


"""
代码示例 2.1 :
    前向模式 的简单程序实现（加法、乘法）
"""

class Variable(object):

    def __init__(self, value, dot=0.):
        self.value = value
        self.dot = dot

    # 加法的重载
    def __add__(self, other):
        """ self * other """
        res = Variable(self.value + other.value)
        res.dot = self.dot + other.dot
        return res

    def __radd__(self, other): 
        return self.__add__(other)

    # 乘法的重载
    def __mul__(self, other):
        """ self * other """
        res = Variable(self.value * other.value)
        res.dot = other.value * self.dot + self.value * other.dot
        return res

    def __rmul__(self, other):
        """ other * self """
        return self.__mul__(other)

    def __str__(self,):
        return "Variable({}, {})".format(self.value, self.dot)


