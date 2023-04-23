


"""
代码示例 4.5 :
    jvp 及 jacfwd 调用方法示意
"""
import jax
import jax.numpy as jnp
from jax import jvp, jacfwd

def f(position):
    x, y = position
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan(y/x)
    return jnp.stack([r, theta])

primals  = jnp.array([3.0, 4.0])
tangents = jnp.array([1.0, 0.0])
value, grad = jvp(f, (primals,), (tangents,))
print(value)  # [ 5.    0.9272952]
print(grad)   # [ 0.6  -0.16     ]

print(jacfwd(f)(primals))
# [[ 0.6   0.8 ]
#  [-0.16  0.12]]


