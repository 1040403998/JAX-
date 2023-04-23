


import numpy as np
import matplotlib.pyplot as plt

def K(t, tau = 150.0):
    if t < 0.0 : return 0.0
    if t >= 0.0 : return 1/ tau * np.exp(-t / tau)

time_step = 0.01   # (ms)
time_stop = 950   # (ms)
firing_list = [2.95, 144.75, 193.67000000000002, 208.97, 226.25, 241.95000000000002, 258.52, 274.09000000000003, 290.71, 445.03000000000003, 526.65, 543.8, 560.16, 576.27, 593.38, 612.74, 628.84, 646.41, 725.72, 814.36, 843.3100000000001, 859.3000000000001]

Ps_trace = []
for idx in range(int(time_stop / time_step)):
    time = idx * time_step
    Ps = 0.0
    for ti in firing_list:
        Ps += K(time-ti)
    Ps_trace.append(Ps) 
        
plt.figure(figsize = (18.0, 3.3))
plt.plot(np.arange(0.0, time_stop, time_step), Ps_trace, label = r"$P_{s}(t)$", c="blue")

plt.xlim((-20, 1015))
plt.ylim((-0.005, 0.045))

plt.xlabel(r"t ($ms$)", fontsize = 16, labelpad=-10.0)
plt.ylabel(r"$P_{s}(t)$" , fontsize = 16)
plt.legend(loc = "lower right", fontsize = 14)
plt.grid("-")
plt.savefig("fig8.10.png")
plt.close()
