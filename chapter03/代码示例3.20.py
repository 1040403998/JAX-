


"""
代码示例 3.20:
    蒙特卡罗计算圆周率 (算法1)
"""

import jax.numpy as jnp
from jax import random

def calcPiViaMC(nsamples, seed=0):
	""" 
		蒙特卡罗计算pi值 
		calculate pi via Monte Carlo
	"""
	key = random.PRNGKey(seed)
	x_samples = random.uniform(key, shape=(nsamples,))
	y_samples = 4 * jnp.sqrt(1-x_samples**2)
	pi_mean = y_samples.mean()
	pi_std  = ((y_samples-pi_mean) ** 2).sum() / (nsamples-1)
	return pi_mean, pi_std

nsamples = int(1E7)
pi, std = calcPiViaMC(nsamples)

print(" pi =", pi)  # >>  pi = 3.141657
print("std =", std) # >> std = 0.7971628
print("stderr = {:.6f}".\
	format(jnp.sqrt(std / nsamples)))    # >> stderr = 0.000282
print("err = {:.6f}".format(pi-jnp.pi))  # >> err = 0.000064


