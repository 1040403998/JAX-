
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt



def log1pexp(x):
    return jnp.log(1. + jnp.exp(x))

def dlog1pexp(x):
    return 1. - 1. / (1 + jnp.exp(x))

x = jnp.linspace(0,3,101)
y = log1pexp(x)
dy = dlog1pexp(x)

plt.plot(x, y , label = "f(x)", linewidth = 2.5, linestyle = ":",)
plt.plot(x, dy, label = "f'(x)",  linewidth = 3)

plt.xlabel("x")
plt.ylabel("y")
plt.grid("-")
plt.legend(loc = "lower right")
plt.savefig("log1pexp.png")