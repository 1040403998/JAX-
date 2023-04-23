


"""
代码示例 4.6 :
    vjp 及 jacrev 调用方法示意
"""
import jax
import jax.numpy as jnp
from jax import vjp, jacrev

def f(position):
    x, y = position
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan(y/x)
    return jnp.stack([r, theta])

primals    = jnp.array([3.0, 4.0])
cotangents = jnp.array([1.0, 0.0])

value, vjp_fun = vjp(f, primals)
grad = vjp_fun(cotangents)[0]
print(value)  # [ 5.    0.9272952]
print(grad)   # [ 0.6   0.8     ]

print(jacrev(f)(primals))
# [[ 0.6   0.8 ]
#  [-0.16  0.12]]


