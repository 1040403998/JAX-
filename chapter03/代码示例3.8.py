


"""
代码示例 3.8
    重排多维数组, atleast函数
"""

import numpy as np
import jax.numpy as jnp

arr = jnp.arange(3)  # >> [0 1 2]

arr1 = jnp.atleast_1d(arr)  
arr2 = jnp.atleast_2d(arr)
arr3 = jnp.atleast_3d(arr)

print(arr1.shape) # (3,)
print(arr2.shape) # (1, 3)
print(arr3.shape) # (1, 3, 1)

