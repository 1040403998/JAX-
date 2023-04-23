


"""
代码示例 3.15
    扩展多维数组, newaxis参数
"""

import numpy as np
import jax.numpy as jnp

arr = jnp.zeros(shape=(2,3,4))
print(arr[jnp.newaxis,:,:,:].shape)  # (1, 2, 3, 4)
print(arr[:,jnp.newaxis,:,:].shape)  # (2, 1, 3, 4)
print(arr[:,:,jnp.newaxis,:].shape)  # (2, 3, 1, 4)
print(arr[:,:,:,jnp.newaxis].shape)  # (2, 3, 4, 1)
