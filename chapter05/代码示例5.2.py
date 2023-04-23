


"""
代码示例 5.2 :
    纯函数
"""

import jax
import jax.random as random

@jax.jit
def func_1(x = 2):
    a = 1
    a = a + x
    return a + x
    
@jax.jit
def func_2(seed = 0):
    key = random.PRNGKey(seed)
    return random.uniform(key)

print(func_1())  # 5
print(func_1())  # 5
print(func_2())  # 0.41845703
print(func_2())  # 0.41845703

""" 严格意义上的纯函数 """

def func_1(x = 2):
    a = 1
    b = a + x
    return b + x
