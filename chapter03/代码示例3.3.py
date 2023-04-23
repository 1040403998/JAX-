


"""
代码示例 3.3
    数组的创建, 数组精度的限制及修改
"""

import numpy as np
import jax.numpy as jnp

x = jnp.array([1.,2.,3.], dtype=jnp.float64)
# >> UserWarning: Explicitly requested dtype <class 'jax._src.numpy.lax_numpy.float64'> requested in array is not available, and will be truncated to dtype float32.
print(x.dtype)  
# >> dtype('float32')

# 修改默认精度
# 注意：只有在运行前生效
from jax.config import config
config.update('jax_enable_x64', True)

x = jnp.array([1., 2., 3.], dtype=jnp.float64)
print(x.dtype)  # dtype('float64')
