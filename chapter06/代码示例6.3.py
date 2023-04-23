"""
代码示例6.3:
    vmap函数的嵌套使用
"""

import jax.numpy as jnp
from jax import vmap
batch1 = 8
batch2 = 6
a, b, c = 2, 3, 4
As = jnp.ones((batch1, batch2, a, b))
Bs = jnp.ones((batch1, b, c))

def f(x, y):
    a = jnp.dot(x, y)
    b = jnp.tanh(a)
    return b

batched_f = vmap(vmap(f), in_axes=(1, None), out_axes=1)

ans = batched_f(As, Bs)
print(ans.shape)  # (batch1, batch2, a, c)
