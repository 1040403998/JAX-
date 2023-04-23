
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

x = jnp.linspace(-2.5, 2.5, 1001)
y = jax.scipy.special.erf(x)
y0 = jnp.e ** (-x**2)

plt.plot(x, y , label = "erf(x)", linewidth = 3)
plt.plot(x, y0, label = "gaussian", linestyle = ":", linewidth = 3)

plt.xlabel("x")
plt.ylabel("y")
plt.grid("-")
plt.legend(loc = "lower right")
plt.savefig("erf.png")