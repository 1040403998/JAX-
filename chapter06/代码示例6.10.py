"""
代码示例6.10:
    pmap的一般用法
    例子：
        硬件间通讯
"""
import jax
device_count = jax.local_device_count()
try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
    
except KeyError as e:
    if str(e) == "'COLAB_TPU_ADDR'":
        if device_count != 8:
            raise EnvironmentError('我们建议您使用colab环境, 并且使用8个TPU')
    
import jax
device_count = jax.local_device_count()
import jax.numpy as jnp
from jax import random, pmap

# 硬件间通信计算归一化值 
# 写法一
from jax import lax
normalize = lambda x: x / lax.psum(x, axis_name='i')
result = pmap(normalize, axis_name='i')(jnp.arange(4.))
print(result)

# 写法二
from functools import partial
@partial(pmap, axis_name='i')
def normalize(x):
  return x / lax.psum(x, 'i')
print(normalize(jnp.arange(4.)))