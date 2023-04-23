


"""
代码示例 3.21:
    蒙特卡罗计算圆周率 (算法2)
"""

import jax.numpy as jnp
from jax import random

def calcPiViaMC(nsamples, seed=0):
	""" 
		蒙特卡罗计算pi值 
		calculate pi via Monte Carlo
	"""
	key = random.PRNGKey(seed)
	xy_samples = random.uniform(key, shape=(nsamples,2))
	distance = jnp.linalg.norm(xy_samples, axis=1)
	d_samples = (distance < 1) * 4
	pi_mean = d_samples.mean()
	pi_std  = ((d_samples-pi_mean) ** 2).sum() / (nsamples-1)
	return pi_mean, pi_std

nsamples = int(1E7)
pi, std = calcPiViaMC(nsamples)

print(" pi =", pi)  # >>  pi = 3.141635
print("std =", std) # >> std = 2.6966705
print("stderr = {:.6f}".\
	format(jnp.sqrt(std / nsamples)))    # >> stderr = 0.000519
print("err = {:.6f}".format(pi-jnp.pi))  # >> err = 0.000042

