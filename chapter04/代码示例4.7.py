


"""
代码示例 4.7:
    hvp 及 hessian 函数
"""

import jax.numpy as jnp
from jax import grad, jvp

def hvp(f, x, v):
    return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)


def hvp(f, primals, tangents):
    """ 反向模式 + 前向模式 """
    return jvp(grad(f), primals, tangents)[1]

def hvp(f, primals, tangents):
    """ 前向模式 + 反向模式 """
    g = lambda primals: jvp(f, primals, tangents)[1]
    return grad(g)(primals)

def f(position):
    x, y = position
    return jnp.sqrt(x**2 + y**2)

from jax import jacfwd, jacrev

def hessian_1(f): return jacfwd(jacfwd(f))
def hessian_2(f): return jacfwd(jacrev(f))
def hessian_3(f): return jacrev(jacfwd(f))
def hessian_4(f): return jacrev(jacrev(f))

primals = jnp.array([3.0, 4.0])
print(hessian_1(f)(primals))
print(hessian_2(f)(primals))
print(hessian_3(f)(primals))
print(hessian_4(f)(primals))


