


"""
代码示例 3.9
    重排多维数组, expand_dims函数
"""

import numpy as np
import jax.numpy as jnp

arr = jnp.zeros(shape=(2,3,4))

arr0 = jnp.expand_dims(arr, axis=0) # 等价于axis=-4
arr1 = jnp.expand_dims(arr, axis=1) # 等价于axis=-3
arr2 = jnp.expand_dims(arr, axis=2) # 等价于axis=-2
arr3 = jnp.expand_dims(arr, axis=3) # 等价于axis=-1

print(arr0.shape) # (1, 2, 3, 4)
print(arr1.shape) # (2, 1, 3, 4)
print(arr2.shape) # (2, 3, 1, 4)
print(arr3.shape) # (2, 3, 4, 1)
