"""
代码示例6.11:
    pmap的具名数组形式
"""
import jax
device_count = jax.local_device_count()
try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
    
except KeyError as e:
    if str(e) == "'COLAB_TPU_ADDR'":
        if device_count != 8:
            raise EnvironmentError('我们建议您使用colab环境，并且使用8个TPU')
    
import jax
device_count = jax.local_device_count()
import jax.numpy as jnp
from jax import pmap, lax
from functools import partial

# 嵌套pmap计算
@partial(pmap, axis_name='rows')
@partial(pmap, axis_name='cols')
def f(x):
    row_normed = x / lax.psum(x, 'rows')
    col_normed = x / lax.psum(x, 'cols')
    doubly_normed = x / lax.psum(x, ('rows', 'cols'))
    return row_normed, col_normed, doubly_normed

x = jnp.arange(8.).reshape((4, 2))  
a, b, c = f(x)  
print(a)  
print(a.sum(0))  
