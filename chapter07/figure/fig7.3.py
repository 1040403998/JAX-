

import jax.numpy as jnp
import matplotlib.pyplot as plt


def f(x):
    return 2.0 * (x - 2.0) ** 2 - 1.0



"""f(x).png"""
def draw_f():
    x = jnp.arange(-1.0, 5.5, 0.1)
    y = f(x)

    x_points = []
    y_points = []

    for k in range(1, 20):
        # 求解 f(x_k) = 1/k
        y_pos = 20 / k
        y_points.append(y_pos) 
        if k % 2 == 0: x_pos = (-jnp.sqrt((y_pos + 1.0) / 2.0)) + 2.0
        else:          x_pos = ( jnp.sqrt((y_pos + 1.0) / 2.0)) + 2.0
        x_points.append(x_pos)
    
    plt.scatter(x_points, y_points, c = "r", s=30, label=r"$\theta_k$")
    plt.plot(x_points, y_points, c = "orange", linestyle="--")
    plt.plot(x,y, label=r"$g_1(\theta)$")

    plt.annotate(text=r"$g_1(\theta)=2(\theta-2)^2 - 1$", xy=(-0.65, 13.0), xytext=(10, 25), textcoords = "offset points")
    plt.annotate(text=r"",  arrowprops=dict(arrowstyle="-|>"), xy=(-0.56, 13.0), xytext=(10, 20), textcoords = "offset points")

    
    plt.annotate(text=r"$\theta_0=2.0$", xy=(2.0, -1.0), xytext=(-20, -15), textcoords = "offset points")
    plt.scatter(*(2,-1), s=80, c="b", marker="p", label="minimum")

    plt.annotate(text=r"$\theta_1$", xy=(x_points[0], y_points[0]), xytext=(  5, -10), textcoords = "offset points")
    plt.annotate(text=r"$\theta_2$", xy=(x_points[1], y_points[1]), xytext=(-15, -10), textcoords = "offset points")
    plt.annotate(text=r"$\theta_3$", xy=(x_points[2], y_points[2]), xytext=(  5, -10), textcoords = "offset points")
    plt.annotate(text=r"$\theta_4$", xy=(x_points[3], y_points[3]), xytext=(-15, -10), textcoords = "offset points")
    plt.annotate(text=r"$\theta_5$", xy=(x_points[4], y_points[4]), xytext=(  5, -10), textcoords = "offset points")
    plt.annotate(text=r"$\theta_6$", xy=(x_points[5], y_points[5]), xytext=(-15, -10), textcoords = "offset points")
    plt.annotate(text=r"$\theta_7$", xy=(x_points[6], y_points[6]), xytext=(  5, -10), textcoords = "offset points")
    plt.annotate(text=r"$\theta_8$", xy=(x_points[7], y_points[7]), xytext=(-10, -12), textcoords = "offset points")
    plt.annotate(text=r"$\theta_9$", xy=(x_points[8], y_points[8]), xytext=(  5, -10), textcoords = "offset points")
    plt.annotate(text=r"...", xy=(x_points[13], y_points[13]), xytext=(-15, -10), textcoords = "offset points")
    plt.annotate(text=r"...", xy=(x_points[14], y_points[14]), xytext=(  5, -10), textcoords = "offset points")

    plt.grid("-")
    plt.legend(loc="lower right")
    plt.xlabel(r"parameter $\theta$")
    plt.ylabel(r"$g_1(\theta)$")
    plt.ylim((-3, 24))
    plt.savefig("fig7.3.png")

if __name__ == "__main__":
    draw_f()
