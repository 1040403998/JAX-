"""
代码示例6.8：
    pmap的一般用法
    例子：
        并行计算数组的三角函数值
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
    
device_count = jax.local_device_count()
import jax.numpy as jnp
from jax import random, pmap

rng = random.PRNGKey(0)  
arr = random.normal(rng, shape=(7, 8, 9))  
   
ans = pmap(jnp.sin, in_axes=0)(arr)  # 好！
ans = pmap(jnp.sin, in_axes=1)(arr)  # 好！
ans = pmap(jnp.sin, in_axes=2)(arr)  # 孬！
# ValueError: compiling computation that requires 9 logical devices, but only 8 XLA devices are available (num_replicas=9, num_partitions=1)  

