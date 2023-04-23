


"""
代码示例 2.4 :
    高阶导数实现的难点
"""

class Variable(object):

    def __init__(self, value, dot=0.):
        self.value = value
        self.dot = Variable(dot)

    # 加法的重载
    def __add__(self, other):
        res = Variable(self.value + other.value)
        res.dot = Variable(self.dot + other.dot)
        return res

    def __radd__(self, other): 
        return self.__add__(other)

    
    def __str__(self):
        if isinstance(self.value, Variable):
            return str(self.value)
        return "Variable({})".format(self.value)

def grad(fun):
    def grad_f(x:Variable):
        x.dot = Variable(1.)
        return fun(x).dot
    return grad_f

def f(x):
    return x + x 


x = Variable(1.)

df = grad(f)
ddf = grad(df)

print(f(x))
print(df(x))   # 报错
print(ddf(x))  # 报错
