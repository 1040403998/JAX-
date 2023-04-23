


"""
代码示例 4.4 :
    利用 grad 函数计算复杂积分
"""

import jax
import jax.numpy as jnp
from typing import Callable, Union
erf = jax.scipy.special.erf

def I0(alpha, r1, r2, r3):
    r = jnp.sqrt(r1**2 + r2**2 + r3**2)
    return (jnp.pi / alpha)**1.5 * erf(jnp.sqrt(alpha) * r) \
        / r * jnp.exp(- 0.25 * alpha * r**2)


def gen_I(m:int, n:int, l:int) -> Callable:

    assert isinstance(m, int) and isinstance(n, int) and isinstance(n, int)

    if m == 0 and n == 0 and l == 0:
        return I0
    elif m < 0 or n < 0 or l < 0:
        return lambda *args: 0.
    
    def I(alpha, r1, r2, r3) -> Union[float, jnp.ndarray]:

        # argnums  =  0      1   2   3
        params     = (alpha, r1, r2, r3)

        if m > 0:
            return (jax.grad(gen_I(m-1, n, l), argnums=1)(alpha, r1, r2, r3) + \
                   (m - 1) * gen_I(m-2, n, l)(alpha, r1, r2, r3))/ (2*alpha)
        elif n > 0:
            return (jax.grad(gen_I(m, n-1, l), argnums=2)(alpha, r1, r2, r3) + \
                   (n - 1) * gen_I(m, n-2, l)(alpha, r1, r2, r3))/ (2*alpha)
        elif l > 0:
            return (jax.grad(gen_I(m, n, l-1), argnums=3)(alpha, r1, r2, r3) + \
                   (l - 1) * gen_I(m, n, l-2)(alpha, r1, r2, r3))/ (2*alpha)
        else:
            raise ValueError("Falal error occurs while doing the iteration in function gen_I")
    return I

I = gen_I(m=2,n=2,l=2)
print(I(alpha=1., r1=1., r2=1., r3=1.))   # 0.23598155


"""test"""
# def f(alpha, m, n, l, x,y,z,r1,r2,r3):
#     d = jnp.sqrt(x**2+y**2+z**2)
#     return (x-r1)**m * (y-r2)**n * (z-r3)**l / d * jnp.e ** (-alpha*((x-r1)**2 + (y-r2)**2 + (z-r3)**2))

# x = jnp.linspace(-2,2,300)
# y = jnp.linspace(-2,2,300)
# z = jnp.linspace(-2,2,300)
# x,y,z = jnp.meshgrid(x,y,z)
# print(x.shape)
# res = jnp.sum(f(alpha=1.,m=2,n=2,l=2,x=x,y=y,z=z,r1=1.,r2=1.,r3=1.,)) * 64 / 300**3
# print(res)
