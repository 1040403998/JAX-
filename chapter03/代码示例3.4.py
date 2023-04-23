


"""
代码示例 3.4
    数组的创建, 常见创建函数
"""

import numpy as np
import jax.numpy as jnp

print(jnp.zeros(3))     # [0. 0. 0.]
print(jnp.ones(3))      # [1. 1. 1.]
print(jnp.full(3, 2.))  # [2. 2. 2.]
print(jnp.empty(3))     # [0. 0. 0.]
print(jnp.eye(2))       # [[1. 0.]
 						#  [0. 1.]] 

x  = jnp.array([[1., 2., 3.]])
x0 = jnp.zeros_like(x)  # [[0. 0. 0.]]
x1 = jnp.ones_like(x)   # [[1. 1. 1.]]
assert x.shape == x0.shape == x1.shape

x = np.logspace(1,2,11)
print(np.log10(x))         # [1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0]
print(np.linspace(1,2,11)) # [1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0]
