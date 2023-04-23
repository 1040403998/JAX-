


"""
代码示例 3.10
    重排多维数组, squeeze函数
"""

import numpy as np
import jax.numpy as jnp

arr = jnp.zeros(shape=(2,1,3,1,4))

arr0 = jnp.squeeze(arr)          
arr1 = jnp.squeeze(arr, axis=1)  
arr3 = jnp.squeeze(arr, axis=3)  

print(arr0.shape) # (2, 3, 4)
print(arr1.shape) # (2, 3, 1, 4)
print(arr3.shape) # (2, 1, 3, 4)
