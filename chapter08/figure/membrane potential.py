
"""
代码示例8.x :
    神经元锋电位的数值模拟
"""

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

# 常数的定义
C = 1.0                                 # (uF/cm^2)
V_res = -60.0                           # (mV)
E_Na, E_K, E_L = 55.0, -72.0, -49.387   # (mV)
g_Na, g_K, g_L = 120.0, 36.0, 0.3       # (m.mho/cm^2)

# 函数的定义
def alpha_n(V) :
    return jax.lax.cond(pred = V==10, 
            true_fun  = lambda void: 0.1, 
            false_fun = lambda void: 0.01 * (10-V) / (jnp.exp((10-V)/10) - 1),
            operand   = None)

def alpha_m(V) : 
    return jax.lax.cond( pred = V==25,
            true_fun  = lambda void: 1.0, 
            false_fun = lambda void: 0.1  * (25-V) / (jnp.exp((25-V)/10) - 1),
            operand   = None)

def alpha_h(V) : return 0.07 * jnp.exp(-V/20)

def beta_n(V) : return 0.125 * jnp.exp(-V/80)
def beta_m(V) : return 4 * jnp.exp(-V/18)
def beta_h(V) : return 1 / (jnp.exp((30-V)/10) + 1)

# 模拟器
def neuron_simulator(time_step):
    def init(V0):
        n0, m0, h0 = 0.0, 0.0, 0.0
        return V0, n0, m0, h0
    
    @jax.jit
    def update(neuron_state, Iext):
        V, n, m, h = neuron_state
        dV = - g_L  * (V-E_L)              \
             - g_K  * (V-E_K)  * n**4      \
             - g_Na * (V-E_Na) * m**3 * h  \
             + Iext
        dn = alpha_n(V-V_res) * (1-n) - beta_n(V-V_res) * n
        dm = alpha_m(V-V_res) * (1-m) - beta_m(V-V_res) * m
        dh = alpha_h(V-V_res) * (1-h) - beta_h(V-V_res) * h

        V += dV / C * time_step
        n += dn * time_step
        m += dm * time_step
        h += dh * time_step
        return V, n, m, h
    
    def get_params(neuron_state):
        V, _, _, _ = neuron_state
        return V
    
    return init, update, get_params

# 定义超参数
V_init    = -60.0  # (mV)
time_step = 0.01   # (ms)
time_stop = 950   # (ms)
t_trace = jnp.arange(0, time_stop, time_step)

# 开始迭代
fun_init, fun_update, fun_get_params = neuron_simulator(time_step)

key = random.PRNGKey(0)
for Iext in [6.5, ]:
    V_trace = []
    neuron_state = fun_init(V0 = V_init)
    for idx in range(int(time_stop / time_step)):
        _, key = random.split(key)
        neuron_state = fun_update(neuron_state, Iext=Iext+8.0*random.normal(key))
        V_trace.append(fun_get_params(neuron_state))

    excitation_times = 0
    duration = 0
    excite = False
    excitation_time_trace = []

    idx = 0
    for V in V_trace:
        duration -= 1
        if V > 0.0 and not excite and duration < 0:
            excite = True
            excitation_times += 1
            excitation_time_trace.append(idx * time_step)
            duration = 100

        elif V < 0.0 and excite:
            excite = False
        idx += 1
    
    print("excitation_times = ", excitation_times,  ", Iext = ", Iext)
    print(excitation_time_trace)

    plt.figure(figsize = (18.0, 3.6))
    plt.plot(t_trace, V_trace, c = "blue", linestyle="-", 
            label = r"$I_{ext} = $" + str(Iext) + r" $nA$")
    # plt.scatter(excitation_time_trace, jnp.ones(len(excitation_time_trace)) * 40, c = "r")
    
    plt.xlim((-20, 1015))
    plt.ylim((-81, 50))

    plt.xlabel(r"t ($ms$)", fontsize = 16, labelpad=-10.0)
    plt.ylabel(r"$\rho(t)$ ($mV$)" , fontsize = 16)
    plt.legend(loc = "lower right", fontsize = 14)
    plt.grid("-")
    plt.savefig("fig8.6-Iext-{}nA.png".format(Iext))
    plt.close()
