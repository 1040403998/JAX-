"""
代码示例5.1.10:

    静态参量
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.special import erf
from functools import partial

# jax.config.update("jax_enable_x64", True)

@partial(jit, static_argnums=(1,2,3))
def I0(alpha, r1, r2, r3):
    r_square = r1**2 + r2**2 + r3**2
    if r_square < (1e-6)**2:
        return 2 * jnp.pi / alpha
    else:
        r = jnp.sqrt(r_square)
        return (jnp.pi / alpha)**1.5 * erf(jnp.sqrt(alpha) * r) / r \
              * jnp.exp(- 0.25 * alpha * r**2)

print(grad(grad(I0))(1.0, 0.0, 0.0, 0.0,)) # 12.566371
print(jit(grad(grad(I0)), 
          static_argnums=(1,2,3))(1.0, 0.0, 0.0, 0.0,)) # 12.566371


