


"""
代码示例 3.12
    扩展多维数组, repeat函数
"""

import numpy as np
import jax.numpy as jnp

arr1 = jnp.arange(4)
print(jnp.repeat(arr1, 2)) # [0 0 1 1 2 2 3 3]

# 多维数组的repeat
arr2 = jnp.arange(4).reshape((2,2))
print(jnp.repeat(arr2, 2, axis=1))
# [[0 0 1 1]
#  [2 2 3 3]]

# 限制元素个数
arr2 = jnp.arange(4).reshape((2,2))
print(jnp.repeat(arr2, 2, axis=1, total_repeat_length=3))
# [[0 0 1]
#  [2 2 3]]

