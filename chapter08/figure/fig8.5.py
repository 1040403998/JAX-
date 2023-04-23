


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3,3,1001)
y_ReLU     = np.max(np.stack([x, np.zeros_like(x)]), axis=0)
print(np.stack([x,x]).shape)
y_softplus = np.log(1+np.exp(x))


x0, y0 = 0.0, 0.0
fig, ax = plt.subplots()
ax.spines["left"].set_position(("data", x0))
ax.spines["bottom"].set_position(("data", y0))
ax.spines["top"].set_visible(False)   # 隐藏顶部线条
ax.spines["right"].set_visible(False) # 隐藏右侧线条
ax.plot(1, y0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot(x0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

plt.xlim((-2.9, 2.9))
plt.ylim((-0.1, 2.9))
plt.plot(x, y_ReLU    , c = "blue", linestyle = "-" , label = "ReLU")
plt.plot(x, y_softplus, c = "red" , linestyle = "-.", label = "softplus")
plt.grid()
plt.legend(loc = "lower right")
plt.savefig("fig8.5.png")
plt.show()