


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4,4,1001)
y_logistic = 1.0 / (1.0 + np.exp(-x))
y_tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


x0, y0 = 0.0, 0.0
fig, ax = plt.subplots()
ax.spines["left"].set_position(("data", x0))
ax.spines["bottom"].set_position(("data", y0))
ax.spines["top"].set_visible(False)   # 隐藏顶部线条
ax.spines["right"].set_visible(False) # 隐藏右侧线条
ax.plot(1, y0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(x0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

plt.xlim((-3.9, 3.9))
plt.ylim((-1.1, 1.1))
plt.plot(x, y_logistic, c = "blue", linestyle = "-" , label = "logistic")
plt.plot(x, y_tanh    , c = "red" , linestyle = "-.", label = "tanh")
plt.grid()
plt.legend(loc = "lower right")
plt.savefig("fig8.4.png")
plt.show()