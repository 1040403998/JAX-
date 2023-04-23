
"""

代码示例5.1.9:

    使用 while_loop 重写循环
    使用 while_loop 重写 fori_loop 函数
    使用 scan 重写 fori_loop 函数

"""

from jax import lax

def loop1(n):
    # 设置循环初始值
    cond_fun = lambda val: val[0] < n
    init_val = (0, 0)        # val = (i, sum)
    def body_func(val):
        i, sum = val
        return (i+1, sum+i)
    # 执行循环
    return lax.while_loop(cond_fun, body_func, init_val)[1]
print(loop1(n=10)) # 45

def loop2(n=10, m=10):
    # 设置外层循环初始值
    cond_fun1 = lambda val: val[0] < n
    init_val1 = (0, 0, 0)    # val = (i, j, sum)
    def body_func1(val1):
        # 设置内层循环初始值
        cond_fun2 = lambda val: val[1] < m
        init_val2 = val1
        def body_func2(val2):
            i, j, sum = val2
            # 在这里书写所需的函数
            return (i, j+1, i * 10 + j + sum)
        # 执行内层循环
        i, j, sum = lax.while_loop(cond_fun2, body_func2, init_val2)
        return (i+1, 0, sum)  # 外层循环变量加1, 内层循环变量归0
    # 执行外层循环
    return lax.while_loop(cond_fun1, body_func1, init_val1)[2]

print(loop2(n=10, m=10)) # 4950


"""    使用 while_loop 重写 fori_loop   """

def fori_loop(lower, upper, body_fun, init_val):
    def while_cond_fun(loop_carry):
        i, upper, _ = loop_carry
        return i < upper
     
    def while_body_fun(loop_carry):
        i, upper, x = loop_carry
        return i+1, upper, body_fun(i, x)

    _, _ , result = lax.while_loop(while_cond_fun, while_body_fun, (lower, upper, init_val))
    return result

# 测试
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
        return fori_loop(lower2, upper2, body_func2, init_val2)
    return fori_loop(lower1, upper1, body_func1, init_val1)

print(loop()) # 4950


"""     使用 scan 重写 fori_loop     """

def fori_loop(lower, upper, body_fun, init_val):
    # 设置初始值
    def scan_body_fun(carry, _):
        i, x = carry
        return (i + 1, body_fun(i, x)), None
    init = (lower, init_val)
    xs = None
    length = upper - lower

    # 执行循环
    (_, result), _ = lax.scan(scan_body_fun, init, xs, length)
    return result

# 测试
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
        return fori_loop(lower2, upper2, body_func2, init_val2)
    return fori_loop(lower1, upper1, body_func1, init_val1)

print(loop()) # 4950