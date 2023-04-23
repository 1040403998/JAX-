


"""
代码示例 3.2
    数组的创建，数组维度的区别
"""

import numpy as np
import jax.numpy as jnp

arr0d = jnp.array(0.)
arr1d = jnp.array([0.,])
arr2d = jnp.array([[0.,]])
arr3d = jnp.array([[[0.,]]])

print("arr0d = ", arr0d.shape)  # arr0d =  ()
print("arr1d = ", arr1d.shape)  # arr1d =  (1,)
print("arr2d = ", arr2d.shape)  # arr2d =  (1, 1)
print("arr3d = ", arr3d.shape)  # arr3d =  (1, 1, 1)

print((arr0d == arr1d).all())  # True
print(arr0d.shape == arr1d.shape)  # False
