
"""
代码示例C.1 :
    无向图的数据结构
"""

import jax
import jax.numpy as jnp
import jax.random as random


mean = jnp.array([0.0, 0.0])
cov  = jnp.array([[1.0, 0.8], [0.8, 1.0]])

def p(sample_num, seed=0):
    key = random.PRNGKey(seed)
    return random.multivariate_normal(key, mean=mean,cov=cov, shape=(sample_num,))


print(p(10))
print(p(10).shape)

import matplotlib.pyplot as plt

x, y = p(1000).T
plt.scatter(x, y, s=3)
plt.savefig("distribution")

