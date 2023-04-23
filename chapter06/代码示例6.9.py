"""
代码示例6.9 :
    pmap的一般用法
    例子：
        并行生成矩阵并进行计算；
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
from jax import random, pmap


# 并行生成矩阵并进行计算
# 生成8个随机数态    
keys = random.split(random.PRNGKey(0), 8)

# 在8个硬件上并行生成8个大型矩阵
mats = pmap(lambda key: random.normal(key, shape=(5000, 6000)))(keys)
print(mats)
print(mats.shape)  # (8, 5000, 6000)

# 在各个硬件上并行运行8个大型矩阵的矩阵乘法
# 没有主机端与硬件的数据通信（矩阵依然存在硬件上）
result = pmap(lambda x: jnp.dot(x, x.T))(mats)
print(result.shape)  # 仅将结果传回主机

# 在各个硬件上计算平均值
print(pmap(jnp.mean)(result))

