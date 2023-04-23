


"""
代码示例 5.1 :
    非纯函数
"""

import random

# 修改外部变量
a = 1
def func_1(x = 2):
    global a
    a = a + x
    return a + x

# 依赖外部变量
def func_2(): 
    return random.uniform(0, 1)

print(func_1())  # 5
print(func_1())  # 7
print(func_2())  # 0.29558890821820216
print(func_2())  # 0.9609259305166594

