


"""
代码示例 3.6
    数组的创建, 随机数组的生成
"""

import numpy as np
import jax.numpy as jnp

# 3.6.1 随机数组的生成（numpy）
from numpy.random import default_rng
rng = default_rng()
vals = rng.standard_normal((50, 50))

# 3.6.2 随机数组的生成（jax）
from jax import random
key = random.PRNGKey(0)
x = random.uniform(key, (50, 50))
y = random.uniform(key, (50, 10))
z = random.uniform(key, (50, 50))

# key的依赖性
key = random.PRNGKey(0)
print(random.normal(key, shape=(2,)))  # [ 1.81608667 -0.75488484]
print(random.normal(key, shape=(2,)))  # [ 1.81608667 -0.75488484]

# 3.6.3 劈分key
key = random.PRNGKey(0)
key, subkey = random.split(key)
print(random.normal(key   , shape=(2,)))  # [0.13893184  1.37066831]
print(random.normal(subkey, shape=(2,)))  # [1.13787844 -0.14331426]

# 劈分多份key
key = random.PRNGKey(0)
key, *subkeys = random.split(key, num=4)
print(subkeys)
# [ array([1518642379, 4090693311], dtype=uint32), 
#   array([ 433833334, 4221794875], dtype=uint32), 
#   array([ 839183663, 3740430601], dtype=uint32)]


