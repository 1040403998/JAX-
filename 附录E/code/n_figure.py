

import numpy as np
import matplotlib.pyplot as plt

def alpha_n(V) : return 0.01 * (10-V) / (np.exp((10-V)/10) - 1)
def alpha_m(V) : return 0.1  * (25-V) / (np.exp((25-V)/10) - 1)
def alpha_h(V) : return 0.07 * np.exp(-V/20)

def beta_n(V) : return 0.125 * np.exp(-V/80)
def beta_m(V) : return 4 * np.exp(-V/18)
def beta_h(V) : return 1 / (np.exp((30-V)/10) + 1)

V = np.linspace(-100, 20)
V_res = -70
V_arr = V - V_res

plt.figure()
plt.plot(V, alpha_n(V_arr) / (alpha_n(V_arr) + beta_n(V_arr)), c = "blue" , linestyle="-" , label = r"$n_{\infty}(V)$")
plt.plot(V, alpha_m(V_arr) / (alpha_m(V_arr) + beta_m(V_arr)), c = "red"  , linestyle="-.", label = r"$m_{\infty}(V)$")
plt.plot(V, alpha_h(V_arr) / (alpha_h(V_arr) + beta_h(V_arr)), c = "green", linestyle="--", label = r"$h_{\infty}(V)$")

plt.xlabel("V (mV)")
plt.ylabel("probability")
plt.xlim((-100, 20))

plt.grid("-")
plt.legend(loc = "center right")
plt.savefig("figE.6.1.png")



