"""
代码示例6.1:
    使用vmap函数将原有函数转化为可以批计算的函数
"""

import jax.numpy as jnp
from jax import vmap

batch = 8
a, b, c, d, e, f = 2, 3, 4, 5, 6, 7

in_axes = (None, 3, 4, 5, None)
out_axes = (0, 1, 2, None)

A = jnp.ones((a, b, c, d, e, f))
B = jnp.ones((a, b, c, batch, d, e, f))
C = jnp.ones((a, b, c, d, batch, e, f))
D = jnp.ones((a, b, c, d, e, batch, f))
E = jnp.ones((a, b))

def func(A, B, C, D, E):
    return (
        A + B,
        B - C,
        C * D,
        E * 2,
    )
    
ans = vmap(func, in_axes, out_axes)(A, B, C, D, E)
print(ans[0].shape)  # (batch, a, b, c, d, e, f)
print(ans[1].shape)  # (a, batch, b, c, d, e, f)
print(ans[2].shape)  # (a, b, batch, c, d, e, f)
print(ans[3].shape)  # (a, b)