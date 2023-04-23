


"""
代码示例 1.7 :
    符号微分与路径积分
"""


import math
from sympy import symbols, eye, Matrix, simplify, expand , integrate, exp, oo, print_latex

# A = Matrix([[9, -4, 0, 0, 0, 0, 0, -4],
#             [-4, 9, -4, 0, 0, 0, 0, 0],
#             [0, -4, 9, -4, 0, 0, 0, 0],
#             [0, 0, -4, 9, -4, 0, 0, 0],
#             [0, 0, 0, -4, 9, -4, 0, 0],
#             [0, 0, 0, 0, -4, 9, -4, 0],
#             [0, 0, 0, 0, 0, -4, 9, -4],
#             [-4, 0, 0, 0, 0, 0, -4, 9]]) / 4
#
# u = Matrix([[x0, x1, x2, x3, x4, x5, x6, x7 ]])
# S = sympify(u * A * u.T)
# print(S)

x0, x1, x2, x3, x4, x5, x6, x7 = symbols("x0, x1, x2, x3, x4, x5, x6, x7")
S = x0*(9*x0/4 - x1 - x7) + x1*(-x0 + 9*x1/4 - x2) + \
    x2*(-x1 + 9*x2/4 - x3) + x3*(-x2 + 9*x3/4 - x4) + \
    x4*(-x3 + 9*x4/4 - x5) + x5*(-x4 + 9*x5/4 - x6) + \
    x6*(-x5 + 9*x6/4 - x7) + x7 * (-x0 - x6 + 9*x7/4)
S = expand(S)
print(S)
f = exp(-S)

for x in [x1, x2, x3, x4, x5, x6, x7]:
    f = simplify(expand(integrate(f, (x, -oo, +oo))))
    print(f)
    print_latex(f)

