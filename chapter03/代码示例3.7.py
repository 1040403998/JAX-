


"""
代码示例 3.7
    重排多为数组, reshape函数
"""

import numpy as np
import jax.numpy as jnp

arr = jnp.arange(9)
print(arr.reshape((3, 3)).shape)   # (3, 3) 
print(arr.reshape((-1, 3)).shape)  # (3, 3) 

