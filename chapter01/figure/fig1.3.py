


import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import math

def f(x):
    return x ** 0.5

def err(h, x, fcn: Callable, fcn_prime:Callable):
    return (fcn(x+h) - fcn(x)) / h - fcn_prime(x)

pow = 16
h_list = np.logspace(-pow-1, -1, pow*3+1).tolist()


x = 0.1
fcn = math.sin
fcn_prime = math.cos

err_math = []
h_log_math = []
err_log_math = []
for e in h_list:
    err_math.append(err(e, x, fcn, fcn_prime))
    err_log_math.append(math.log10(abs(err(e,x,fcn,fcn_prime))))
    h_log_math.append(math.log10(e))

fcn0 = np.sin
fcn_prime0 = np.cos

h_32_numpy = np.logspace(-pow-1, -1, pow*3+1, dtype=np.float32)
err_32_numpy = err(h_32_numpy,x,fcn0,fcn_prime0)
h_32_log_numpy = np.log10(h_32_numpy)
err_32_log_numpy = np.log10(np.abs(err_32_numpy))

h_64_numpy = np.logspace(-pow-1, -1, pow*3+1, dtype=np.float64)
err_64_numpy = err(h_64_numpy,x,fcn0,fcn_prime0)
h_64_log_numpy = np.log10(h_64_numpy)
err_64_log_numpy = np.log10(np.abs(err_64_numpy))

import torch
fcn1 = torch.sin
fcn_prime1 = torch.cos

x_32_torch = torch.tensor(x, dtype=torch.float32)
h_32_torch = torch.logspace(-pow-1, -1, pow*3+1, dtype=torch.float32)
err_32_torch = err(h_32_torch, x_32_torch, fcn1, fcn_prime1)
h_32_log_torch = torch.log10(h_32_torch)
err_32_log_torch = torch.log10(torch.abs(err_32_torch))

x_64_torch = torch.tensor(x, dtype=torch.double)
h_64_torch = torch.logspace(-pow-1, -1, pow*3+1, dtype=torch.double)
err_64_torch = err(h_64_torch,x_64_torch,fcn1,fcn_prime1)
h_64_log_torch = torch.log10(h_64_torch)
err_64_log_torch = torch.log10(torch.abs(err_64_torch))

import jax.numpy as jnp
from jax.config import config

fcn2 = jnp.sin
fcn_prime2 = jnp.cos

config.update("jax_enable_x64",False)
h_32_jnp = jnp.logspace(-pow-1, -1, pow*3+1)
err_32_jnp = err(h_32_jnp,x,fcn2,fcn_prime2)
h_32_log_jnp = jnp.log10(h_32_jnp)
err_32_log_jnp = jnp.log10(jnp.abs(err_32_jnp))

config.update("jax_enable_x64",True)
h_64_jnp = jnp.logspace(-pow-1, -1, pow*3+1)
err_64_jnp = err(h_64_jnp,x,fcn2,fcn_prime2)
h_64_log_jnp = jnp.log10(h_64_jnp)
err_64_log_jnp = jnp.log10(jnp.abs(err_64_jnp))


plt.figure(dpi=1500)

plt.plot(h_log_math, err_log_math, label = "math float")
plt.plot(h_32_log_numpy, err_32_log_numpy, label = "NumPy float32", linestyle = "-.")
plt.plot(h_64_log_numpy, err_64_log_numpy, label = "NumPy float64", linestyle = ":")
plt.plot(h_32_log_jnp, err_32_log_jnp, label = "Jax float32", linestyle="--")
plt.scatter(h_64_log_jnp, err_64_log_jnp, s = 4, label = "Jax float64")

plt.legend(loc = "upper right")
plt.xlabel("log(h)")
plt.ylabel("log(err)")
plt.grid()
plt.savefig(r"/mnt/c/Users/lenovo/Desktop/fig1.3.png")