
import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def Himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y** 2 - 7) ** 2

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

def intersect(fun1, fun2, left, right, err=1E-10):
    mid = (right + left) / 2
    vleft1, vright1 = fun1(left) , fun1(right)
    vleft2, vright2 = fun2(left) , fun2(right)

    assert right - left > 0
    assert (vleft1 - vleft2) * (vright1 - vright2) <=0
    
    if right - left < err:
        return mid
    
    vmid1, vmid2 = fun1(mid) , fun2(mid)
    if (vleft1 - vleft2) * (vmid1 - vmid2) <=0:
        return intersect(fun1, fun2, left, mid, err)
    else:
        return intersect(fun1, fun2, mid, right, err)
    

def draw_tangent():
    theta = (-4.3, 2.0)
    p = (1.0, 0.0)

    def phi(alpha):
        x = theta[0] + alpha * p[0]
        y = theta[1] + alpha * p[1]
        return Himmelblau(x, y)

    dphi = jax.grad(lambda x: jnp.sum(phi(x)))
    
    alpha = jnp.linspace(0, 8.9, 1001)
    phi_alpha = phi(alpha)
    dphi_alpha = dphi(alpha)

    fig = plt.figure()
    plt.plot(alpha, phi_alpha, label = r"$\phi(\alpha)$")
    plt.plot(alpha, dphi_alpha, label = r"$d\phi (\alpha)$", c = "r", linestyle="--")
    plt.savefig("fig7.6.png")

def draw_line_search():
    theta = (-4.3, 2.0)
    p = (1.0, 0.0)

    def phi(alpha):
        x = theta[0] + alpha * p[0]
        y = theta[1] + alpha * p[1]
        return Himmelblau(x, y)
    dphi = jax.grad(lambda x: jnp.sum(phi(x)))

    dphi0 = dphi(0.0)
    c2 = 0.15

    x0 = 0.0
    x1 = intersect(dphi, lambda x: dphi0*c2, 0.0, 2.0)
    x2 = intersect(dphi, lambda x: dphi0*c2, 4.0, 6.0)
    x3 = intersect(dphi, lambda x: dphi0*c2, 6.0, 8.0)
    y0, y1, y2, y3 = phi(x0), phi(x1), phi(x2), phi(x3)
    delta = 1.0
    x0_line = jnp.linspace(x0, x0 + delta * 1.5, 101)
    x1_line = jnp.linspace(x1-delta, x1+delta, 101)
    x2_line = jnp.linspace(x2-delta, x2+delta, 101)
    x3_line = jnp.linspace(x3-delta, x3+delta, 101)

    y0_tangent = dphi0 * (x0_line - x0) + y0
    y0_line = (c2 * dphi0) * (x0_line - x0) + y0
    y1_line = (c2 * dphi0) * (x1_line - x1) + y1
    y2_line = (c2 * dphi0) * (x2_line - x2) + y2
    y3_line = (c2 * dphi0) * (x3_line - x3) + y3

    alpha = jnp.linspace(0, 8.9, 1001)
    phi_alpha = phi(alpha)
    dphi_alpha = dphi(alpha)
    plt.plot(alpha, phi_alpha, label = r"$\phi(\alpha)$")
    # plt.plot(alpha, dphi_alpha, label = r"$d\phi (\alpha)$", c = "r", linestyle="--")
    # plt.plot((0.0,8.0), (c2 * dphi0, c2 * dphi0))

    plt.plot(x0_line[:30], y0_tangent[:30], c = "purple", linestyle = "--")
    plt.annotate(text="desired\n  slope", arrowprops=dict(arrowstyle="<|-"), 
                 xy=(1.0, 120.0), xytext=(10, 20), textcoords = "offset points")

    plt.plot(x0_line, y0_line, c = "red", linestyle = "--")
    plt.annotate(text="tangent", arrowprops=dict(arrowstyle="<|-"), 
                 xy=(x0_line[25], y0_tangent[25]), xytext=(30, -20), textcoords = "offset points")
    plt.plot(x1_line, y1_line, c = "red", linestyle = "--")
    plt.plot(x2_line, y2_line, c = "red", linestyle = "--")
    plt.plot(x3_line, y3_line, c = "red", linestyle = "--")

    plt.plot((x0, x0), (-20.0, y0), linestyle=":", c = "green")
    plt.plot((x1, x1), (-20.0, y1), linestyle=":", c = "green")
    plt.plot((x2, x2), (-20.0, y2), linestyle=":", c = "green")
    plt.plot((x3, x3), (-20.0, y3), linestyle=":", c = "green")

    plt.annotate(text=r"$\phi(\alpha)=f(\vec{\theta}_k+\alpha_k \vec{p}_k)$", 
                 xy=(5.8, 130.0), xytext=(0, 0), textcoords = "offset points")

    plt.annotate(text= "", arrowprops=dict(arrowstyle="<|-|>"), xy=(x1, -15), xytext=(x2, -15))
    plt.annotate(text= "acceptable", xy=((x1+x2)/2, -15), xytext=(-25, 5), textcoords = "offset points")

    plt.annotate(text= "", arrowprops=dict(arrowstyle="-|>"), xy=(x3, -15), xytext=(10, -15))
    plt.annotate(text= "acceptable", xy=((x3+10)/2, -15), xytext=(-10, 5), textcoords = "offset points")


    # plt.grid("-")
    plt.legend(loc="lower left")
    plt.xlim((-1.1, 10))
    plt.xlabel(r"$\alpha$")
    plt.ylim((-20, 175))
    plt.ylabel(r"$\phi(\alpha)$")
    plt.savefig("fig7.6.png")

    print(x1, x2, x3)
    print(c2 * dphi0)

    # alpha = jnp.linspace(0, 8.9, 1001)
    # phi_alpha = phi(alpha)
    # L_alpha = L(alpha)

    # fig = plt.figure()
    # plt.plot(alpha, phi_alpha, label = r"$\phi(\alpha)$")
    # plt.plot(alpha, L_alpha, label = r"$l (\alpha)$", c = "r", linestyle="--")

    # x0 = 0.0
    # x1 = intersect(phi, L, 2.0, 4.0)
    # x2 = intersect(phi, L, 4.0, 6.0)
    # x3 = intersect(phi, L, 6.0, 8.0)
    # print(x1, x2, x3)
    # y0, y1, y2, y3 = phi(x0), phi(x1), phi(x2), phi(x3)

    # # annotate

    # plt.plot((x0, x0), (-20.0, y0), linestyle=":", c = "green")
    # plt.plot((x1, x1), (-20.0, y1), linestyle=":", c = "green")
    # plt.plot((x2, x2), (-20.0, y2), linestyle=":", c = "green")
    # plt.plot((x3, x3), (-20.0, y3), linestyle=":", c = "green")
    # plt.annotate(text=r"$\phi(\alpha)=f(\vec{\theta}_k+\alpha_k \vec{p}_k)$" , 
    #              xy=(0.5, 26.0), xytext=(0, 0), textcoords = "offset points")
    # plt.annotate(text=r"$l(\alpha)=f(\vec{\theta}_k) + c_1 \alpha \vec{p}_k^T \cdot \nabla f(\vec{\theta}_k)$" , 
    #              xy=(1.0, 130.0), xytext=(0, 0), textcoords = "offset points")
    # plt.annotate(text= "", arrowprops=dict(arrowstyle="<|-|>"), xy=(x0, -15), xytext=(x1, -15))
    # plt.annotate(text= "acceptable", xy=((x0+x1)/2, -15), xytext=(-25, 5), textcoords = "offset points")

    # plt.annotate(text= "", arrowprops=dict(arrowstyle="<|-|>"), xy=(x2, -15), xytext=(x3, -15))
    # plt.annotate(text= "acceptable", xy=((x2+x3)/2, -15), xytext=(-28, 5), textcoords = "offset points")

    # plt.annotate(text= "  global\n minimizer",  arrowprops=dict(arrowstyle="-|>"),
    #              xy=(xmin_2, y2), xytext=(-40, 82), textcoords = "offset points")
    # plt.annotate(text=r"with $\theta_k$ = (-4.8, 2.0), $p_k = (1.0, 0.0)$", 
    #              xy=(-2, 2.0), xytext=(-20, -45), textcoords = "offset points")

    # plt.grid("-")
    # plt.legend(loc="lower right")
    # plt.xlim((-0.7, 11))
    # plt.xlabel(r"$\alpha$")
    # plt.ylim((-20, 175))
    # plt.ylabel(r"$\phi(\alpha)$")

    # plt.savefig("fig7.6.png")


if __name__ == "__main__":
    # draw_tangent()
    draw_line_search()
