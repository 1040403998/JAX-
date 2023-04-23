


"""
代码示例 3.11
    重排多维数组, flip 和 swapaxis 函数
"""

import numpy as np
import jax.numpy as jnp

m = jnp.arange(9).reshape((3, 3))
m0 = jnp.flip(m,0)
m1 = jnp.flip(m,1)
m2 = jnp.swapaxes(m,0,1)
"""
m = [[0 1 2]
     [3 4 5]   
     [6 7 8]]   # original

m0 = [[6 7 8]
      [3 4 5]   
      [0 1 2]]  # flip(m,0)

m1 = [[2 1 0]
      [5 4 3]   
      [8 7 6]]  # flip(m,1)
      
m2 = [[0 3 6]
      [1 4 7]   
      [2 5 8]]  # swapaxes(m,0,1)
"""
print(m)   
print(m0)
print(m1)
print(m2)
