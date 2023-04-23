"""
代码示例6.5：
    展示vmap前后函数的jaxpr
"""

import jax.numpy as jnp
from jax import make_jaxpr, vmap, pmap

def f(x, y):
  a = jnp.dot(x, y)
  b = jnp.tanh(a)
  return b

xs = jnp.ones((8, 2, 3))
ys = jnp.ones((8, 3, 4))

print("<f jaxpr>")
print(make_jaxpr(f)(xs[0], ys[0]))

print("<vmap(f) jaxpr<")
print(make_jaxpr(vmap(f))(xs, ys))

# <f jaxpr>
# { lambda ; a:f32[2,3] b:f32[3,4]. let
#     c:f32[2,4] = dot_general[
#       dimension_numbers=(((1,), (0,)), ((), ()))
#       precision=None
#       preferred_element_type=None
#     ] a b
#     d:f32[2,4] = tanh c
#   in (d,) }

# <vmap(f) jaxpr>
# { lambda ; a:f32[8,2,3] b:f32[8,3,4]. let
#     c:f32[8,2,4] = dot_general[
#       dimension_numbers=(((2,), (1,)), ((0,), (0,)))
#       precision=None
#       preferred_element_type=None
#     ] a b
#     d:f32[8,2,4] = tanh c
#   in (d,) }