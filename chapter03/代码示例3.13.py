


"""
代码示例 3.13
    扩展多维数组, concatenate 函数
"""

import numpy as np
import jax.numpy as jnp

arr1 = jnp.array([1,2,3])
arr2 = jnp.array([4,5,6])
print(jnp.concatenate([arr1, arr2]))  # [1, 2, 3, 4, 5, 6]
print(jnp.concatenate([jnp.atleast_2d(arr1), jnp.atleast_2d(arr2)], axis=0)) # [[1, 2, 3], [4, 5, 6]]
print(jnp.concatenate([jnp.atleast_2d(arr1), jnp.atleast_2d(arr2)], axis=1)) # [[1, 2, 3, 4, 5, 6]]
