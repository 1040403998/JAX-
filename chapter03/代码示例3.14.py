


"""
代码示例 3.14
    扩展多维数组, stack 函数 等
"""

import numpy as np
import jax.numpy as jnp

a = jnp.array([1,2,3,4]).reshape((2,2))
b = jnp.array([5,6,7,8]).reshape((2,2))
print(jnp.concatenate((a,b), axis=0))
# >> [[1, 2], [3, 4], [5, 6], [7, 8]]
print(jnp.concatenate((a,b), axis=1))
# >> [[1, 2, 5, 6],
#     [3, 4, 7, 8]]
print(jnp.stack((a,b), axis=0))
# >> [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
print(jnp.stack((a,b), axis=1))
# >> [[[1, 2], [5, 6]],
#     [[3, 4], [7, 8]]]

# 函数hstack 和 vstack
print(jnp.vstack((a,b)))
# >> [[1, 2], [3, 4], [5, 6], [7, 8]]
print(jnp.hstack((a,b)))
# >> [[1, 2, 5, 6],
#     [3, 4, 7, 8]]

# 函数 column_stack
a = jnp.arange(9).reshape(3,3)
b = jnp.array([10, 11, 12])
print(np.concatenate((a, jnp.expand_dims(b, axis=1)), axis=1))
# [[0, 1, 2, 10],
#  [3, 4, 5, 11],
#  [6, 7, 8, 12]]

print(jnp.column_stack((a,b)))
# [[0, 1, 2, 10],
#  [3, 4, 5, 11],
#  [6, 7, 8, 12]]
