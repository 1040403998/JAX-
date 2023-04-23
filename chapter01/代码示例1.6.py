


"""
代码示例 1.6 :
    符号微分与厄米多项式
"""

import math
from sympy import symbols, simplify, expand, diff

e = math.e
x = symbols("x")

# 计算诶米多项式的表达式
def H_expr(order: int):
    assert isinstance(order, int) and order >= 0
    expr = e ** (-x**2)
    
    for _ in range(order):
        expr = diff(expr, x)

    expr *= e ** (x**2) * (-1) ** order
    
    return expand(simplify(expr))

# 对前15个表达式进行打印
for i in range(15):
    print("H{:<2.0f}(x) = {}".format(i, H_expr(i)))

# 一个可以直接调用的厄米多项式函数
def H(x, n):
    """ 计算Hn(x) """
    return eval(str(H_expr(n)))
