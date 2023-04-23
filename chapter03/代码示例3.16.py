


"""
代码示例 3.16
    索引多维数组 及 越界行为处理
"""

import numpy as np
import jax.numpy as jnp

"""              索引多维数组               """

# 3.2.10 索引多维数组

# 单个元素索引
arr1d = jnp.arange(9)
print(arr1d[3], arr1d[-3])  # 3, 6
arr2d = jnp.arange(9).reshape((3,3))
print(arr2d[1])  # [3 4 5]
print(arr2d[1][1])  # 4
print(arr2d[1, 1])  # 4

# 数组切片
arr1d = jnp.arange(9) # [0 1 2 3 4 5 6 7 8]
print(arr1d[ : :-1])  # [8 7 6 5 4 3 2 1 0]
print(arr1d[2:8: 2])  # [2 4 6]

arr3d = jnp.arange(216).reshape((6,6,6))
print(arr3d[2:4, 2:4, 2:4].shape)  # (2, 2, 2)

# 数组作为索引
# 一维数组索引一维数组
labels = jnp.array([0,2,4])
print(jnp.eye(10)[labels])
# >> [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]


# 二维数组索引一维数组
arr1d = jnp.arange(9)
print(arr1d[jnp.array([[1,2,3], [4,5,6]])])
# [[1 2 3]
#  [4 5 6]]

# 一维数组索引二维数组
arr2d = jnp.arange(9).reshape((3,3))
print(arr2d[jnp.array([1,2])])
# [[3 4 5]
#  [6 7 8]]

# 二维数组索引二维数组
arr2d = jnp.arange(9).reshape((3,3))
print(arr2d[jnp.array([[0, 2], [0, 1]])])
# [[[0, 1, 2], [6, 7, 8]]
#  [[0, 1, 2], [3, 4, 5]]]

# 掩码
arr1d = jnp.arange(9)
print(arr1d % 2 == 0)
# [ True  False  True  False  True  False  True  False  True]
print(arr1d[arr1d%2==0])
# [0 2 4 6 8]

# Numpy独有
arr2d = np.arange(16).reshape((4,4))  
arr2d[np.array([1,2,3]), np.array([1,2,3])]  
print(arr2d[np.array([1,2,3]), np.array([1,2,3])])
# [ 5 10 15]

"""            越界行为处理               """

arr1d = jnp.arange(3)
print(arr1d[5])  # 2
print(arr1d[2:5])  # [2]
print(arr1d[-5:-2])  # [0]

# numpy 的原地更新：直接赋值
arr1d = np.arange(9)
print(arr1d)  # [0 1 2 3 4 5 6 7 8]
arr1d[::2] = np.zeros(5)
print(arr1d)  # [0 1 0 3 0 5 0 7 0]

# jax 的异地更新：at方法
arr1d = jnp.arange(9)
print(arr1d)     # [0 1 2 3 4 5 6 7 8]
newarr1d = arr1d.at[::2].set(jnp.zeros(5))
print(newarr1d)  # [0 1 0 3 0 5 0 7 0]

arr1d = jnp.arange(9)
newarr1d = arr1d.at[10].add(3, mode='clip')
# DeviceArray([ 0,  1,  2,  3,  4,  5,  6,  7, 11], dtype=int32)
print(newarr1d)

newarr1d = arr1d.at[10].add(3, mode='drop')
# DeviceArray([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int32)
print(newarr1d)

newarr1d = arr1d.at[7:12].get(mode='fill', fill_value=9)
# DeviceArray([7, 8], dtype=int32)
print(newarr1d)

newarr1d = arr1d.at[12].get(mode='fill', fill_value=9)
# DeviceArray(9, dtype=int32)
print(newarr1d)


