


"""
代码示例 5.4 :
    即时编译
"""

import jax 
import jax.numpy as jnp

""" 不进行即时编译 """
a = 0
def f(x):
    global a
    a += 1
    print("function f is called, ", end="")
    return x + a

x = jnp.array([1,2,3])
print("f(x) = ", f(x)) # function f is called, f(x) =  [2 3 4]
print("f(x) = ", f(x)) # function f is called, f(x) =  [3 4 5]
print("f(x) = ", f(x)) # function f is called, f(x) =  [4 5 6]
print("a =", a) # a = 3


""" 使用 jax.jit 进行即时编译 """
a = 0
@jax.jit
def f(x):
    global a
    a += 1
    print("function f is called, ", end="")
    return x + a

x = jnp.array([1,2,3])
print("f(x) = ", f(x)) # function f is called, f(x) =  [2 3 4]
print("f(x) = ", f(x)) # [2 3 4]
print("f(x) = ", f(x)) # [2 3 4]
print("a =", a)  # a = 1
print("f(x) = ", f(x.reshape(1,1,-1))) # function f is called, f(x) =  2
print("a =", a)  # a = 2

