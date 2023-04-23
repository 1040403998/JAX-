


"""
代码示例 4.1 :
    jax 的 grad 函数
"""

from jax import grad

def f(x,y):
    return (x + y) ** 2

df1  = grad(f, argnums=0)
df2  = grad(f, argnums=1)
df11 = grad(df1, argnums=0)
df12 = grad(df1, argnums=1)
df21 = grad(df2, argnums=0)
df22 = grad(df2, argnums=1)
df111= grad(df11, argnums=0)


x, y = 1.0, 2.0

"""  第零阶  """
print(f(x,y))       # >>  9.0

"""  第一阶  """
print(df1 (x,y))    # >>  6.0
print(df2 (x,y))    # >>  6.0

"""  第二阶  """
print(df11(x,y))    # >>  2.0
print(df12(x,y))    # >>  2.0
print(df21(x,y))    # >>  2.0
print(df22(x,y))    # >>  2.0

"""  第三阶  """
print(df111(x,y))   # >>  0.0

# diction = {1: (2,3), 3: (-1,3), -223:(300,200)}
# print(min(diction, key = diction.__getitem__)) # -223
# print(diction.__getitem__)