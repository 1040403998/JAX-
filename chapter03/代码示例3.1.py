


"""
代码示例 3.1
    数组的创建
"""

import numpy as np
import jax.numpy as jnp

# 3.1.1 初始化数组
arr0 = np.array(0)
arr1 = np.zeros((3, ))
arr2 = jnp.zeros((3, 3))
arr3 = jnp.zeros((3, 3, 3))

# 3.1.2 检查数组类型
print(type(arr0))    # >>  <class 'numpy.ndarray'>
print(type(arr2))    # >>  <class 'jaxlib.xla_extension.DeviceArray'>
assert isinstance(arr0, np.ndarray )  
assert isinstance(arr2, jnp.ndarray) 

# 3.1.3 数组形状与属性
print(arr0.shape)  # ()
print(arr1.shape)  # (3, )
print(arr2.ndim)   # 2
print(arr3.size)   # 27
