


"""
代码示例 4.8 :
    自定义前向模式: custom_jvp
"""


import jax.numpy as jnp
from jax import custom_jvp, grad, custom_vjp

@custom_jvp
# @custom_vjp
def log1pexp(x):
    return jnp.log(1. + jnp.exp(x))

# 前向模式
@log1pexp.defjvp
def log1pexp_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    ans = log1pexp(x)
    ans_dot = (1. - 1./(1.+jnp.exp(x))) * x_dot
    return ans, ans_dot

"""
# 反向模式
def log1pexp_fwd(x):
    return log1pexp(x), x

def log1pexp_bwd(res, g):
    x = res
    return (1. - 1./(1.+jnp.exp(x))) * g

log1pexp.defvjp(log1pexp_fwd, log1pexp_bwd)
"""

print(log1pexp(0.))         # 0.6931472 ~ ln2
print(grad(log1pexp)(100.)) # 1.0

from jax import vjp
primals, cotangents = 0.0, 1.0
value, fun_vjp = vjp(log1pexp, primals)
print(value)                   # >> 0.6931472 ~ ln2
print(fun_vjp(cotangents)[0])  # >> 0.5
