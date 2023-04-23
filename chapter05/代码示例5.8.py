"""
代码示例5.1.8:

    使用fori_loop重写循环
"""

# 使用fori_loop重写循环
from jax import lax

def loop(n=10, m=10):
    lower1 = 0
    upper1 = n
    init_val1 = 0
    def body_func1(i, val1):
        lower2 = 0
        upper2 = m
        init_val2 = val1
        def body_func2(j, val2):
            return i * 10 + j + val2
        return lax.fori_loop(lower2, upper2, body_func2, init_val2)
    return lax.fori_loop(lower1, upper1, body_func1, init_val1)

print(loop())
