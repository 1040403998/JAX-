

import jax.numpy as jnp
import matplotlib.pyplot as plt

def Himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y** 2 - 7) ** 2


"""Himmelblau_maxmin.png"""
def draw_Himmelblau__with_line():
    x = jnp.arange(-6.5,6.5,0.1)
    y = jnp.arange(-6.5,6.5,0.1)
    x, y = jnp.meshgrid(x,y)
    z = Himmelblau(x,y)

    maximum = (-0.270845, -0.923039)
    minima1 = (      3.0,       2.0)
    minima2 = (-2.905118,  3.131312)
    minima3 = (-3.779310, -3.283186)
    minima4 = ( 3.584428, -1.848126)

    plt.scatter(*maximum, s=100, c="r", marker="*", label="maxinum")
    plt.scatter(*minima1, s=80, c="b", marker="p", label="minima")
    plt.scatter(*minima2, s=80, c="b", marker="p", )
    plt.scatter(*minima3, s=80, c="b", marker="p", )
    plt.scatter(*minima4, s=80, c="b", marker="p", )

    # Draw a line here.
    plt.plot((-4.3, 4.5), (2.0, 2.0), linestyle="--")
    plt.annotate(text=r"",  arrowprops=dict(arrowstyle="Fancy"),
                 xy=(-3, 1.9), xytext=(10, -20), textcoords = "offset points")
    plt.annotate(text=r"take line search here to construct $\phi(\alpha) = f(\vec{\theta}_k + \alpha \vec{p}_k)$", 
                 xy=(-2, 1.9), xytext=(-65, -30), textcoords = "offset points")
    plt.annotate(text=r"with $\vec{\theta}_k$ = (-4.3, 2.0), $\vec{p}_k = (1.0, 0.0)$", 
                 xy=(-2, 2.0), xytext=(-20, -50), textcoords = "offset points")

    log_levels = jnp.array([2.5, 3.2, 3.8, 4.4, 4.7, 5.0, 5.1849, 5.45, 5.8, 6.2, 6.5])
    CP = plt.contour(x,y,z, levels=jnp.exp(log_levels),cmap="Blues")
    plt.legend(loc="lower right")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.savefig("Himmelblau_maxmin_with_line.png")

import math
const = (-1+math.sqrt(5)) / 2

def minimizer(fun, left, mid, right, err = 1E-10):
    # left <- a -> mid <- c -> right
    a, c = mid - left , right - mid
    vleft, vmid, vright = fun(left), fun(mid), fun(right)
    assert a > 0 and c > 0
    # assert vmid < vleft or vmid < vright
    
    delta = (a + c) * const

    if a+c < err:
        return mid
    if a < c:
        mid_new = left + delta 
        assert mid < mid_new
        if fun(mid_new) < vmid:
            return minimizer(fun, mid, mid_new, right, err)
        else:
            return minimizer(fun, left, mid, mid_new, err)
    else:
        mid_new = right - delta
        assert mid_new < mid
        if fun(mid_new) < vmid:
            return minimizer(fun, left, mid_new, mid, err)
        else:
            return minimizer(fun, mid_new, mid, right, err)    

def draw_line_search():
    theta = (-4.3, 2.0)
    p = (1.0, 0.0)

    def phi(alpha):
        x = theta[0] + alpha * p[0]
        y = theta[1] + alpha * p[1]
        return Himmelblau(x, y)
    alpha = jnp.linspace(0, 8.9, 1001)
    phi_alpha = phi(alpha)

    fig = plt.figure()
    plt.plot(alpha, phi_alpha, label = r"$\phi(\alpha)$")

    # annotate
    xmin_1 = minimizer(phi, 1.0, 1.5, 3.0)
    xmin_2 = minimizer(phi, 7.0, 7.5, 9.0)
    y1, y2 = phi(xmin_1), phi(xmin_2)

    plt.plot((xmin_1, xmin_1), (-20.0, y1), linestyle="--")
    plt.plot((xmin_2, xmin_2), (-20.0, y2), linestyle="--")

    plt.annotate(text= "first local \n minimizer",  arrowprops=dict(arrowstyle="-|>"),
                 xy=(xmin_1, y1), xytext=(-20, 35), textcoords = "offset points")
    plt.annotate(text= "  global\n minimizer",  arrowprops=dict(arrowstyle="-|>"),
                 xy=(xmin_2, y2), xytext=(-40, 82), textcoords = "offset points")
    plt.annotate(text=r"$\phi(\alpha)=f(\vec{\theta}_k+\alpha_k \vec{p}_k)$" , 
                 xy=(1.0, 110.0), xytext=(0, 0), textcoords = "offset points")

    plt.grid("-")
    plt.legend(loc="lower right")
    plt.xlim((-0.7, 10.2))
    plt.xlabel(r"$\alpha$")
    plt.ylim((-20, 175))
    plt.ylabel(r"$\phi(\alpha)$")

    plt.savefig("line_search.png")


if __name__ == "__main__":
    draw_Himmelblau__with_line()
    draw_line_search()
