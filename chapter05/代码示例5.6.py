


"""
代码示例5.6:
    条件控制语句
    cond
    switch
"""

import jax
from jax import lax
import jax.numpy as jnp


""" 合法的条件控制语句 """
def f(x):
    assert isinstance(x, jnp.ndarray)
    if x.shape[0] < 4:
      return x * 2
    else:
      return - x * 2

jitted_f = jax.jit(f)
arr1 = jnp.arange(3)
arr2 = jnp.arange(5)
print(jitted_f(arr1))   # [ 0  2  4]
print(jitted_f(arr2))   # [ 0 -2 -4 -6 -8]

print(jax.make_jaxpr(f)(arr1))
# { lambda  ; a.
#   let b = mul a 2
#   in (b,) }
print(jax.make_jaxpr(f)(arr2))
# { lambda  ; a.
#   let b = neg a
#       c = mul b 2
#   in (c,) }

""" 不合法的条件控制语句 """

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

def I0(alpha, r1, r2, r3):
    r = jnp.sqrt(r1**2 + r2**2 + r3**2)
    if r < 1e-6:
        return 2 * jnp.pi / alpha
    else:
        return (jnp.pi / alpha)**1.5 * erf(jnp.sqrt(alpha) * r) / r \
              * jnp.exp(- 0.25 * alpha * r**2)

# jitted_I0 = jax.jit(I0)  # err

""" 基于lax.cond的条件控制语句 """
from jax import lax

def I0(alpha, r1, r2, r3):
    r = jnp.sqrt(r1**2 + r2**2 + r3**2)
    return lax.cond(
      pred      = r < 1e-6,
      true_fun  = lambda void: 2. * jnp.pi / alpha,
      false_fun = lambda void: (jnp.pi / alpha)**1.5 * erf(jnp.sqrt(alpha) * r) / r \
                              * jnp.exp(- 0.25 * alpha * r**2),
      operand   = 0
    )

jitted_I0 = jax.jit(I0)
print(jitted_I0(1.0, 0.0, 0.0, 0.0))

f = lambda operand: lax.cond(True, lambda x: x+1, lambda x: x-1, operand)
print(jax.make_jaxpr(f)(jnp.array(1.0)))