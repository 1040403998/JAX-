


"""
代码示例 4.9 :
    自定义反向模式: custom_vjp

    例子: 不动点法
"""

from typing import Callable

def BisectionSolver(f: Callable, left, right, params, eps=1E-6):
    """ 二分法求解函数f的不动点 """
    assert isinstance(f, Callable) and left < right
    lvalue = f(left , *params)
    rvalue = f(right, *params)
    assert rvalue * lvalue < 0.

    mid = (right + left) / 2
    mvalue = f(mid, *params)
    
    step = 0
    while right - left > eps:
        if rvalue * mvalue < 0:     # 零点在右侧
            left, lvalue = mid, mvalue
        elif lvalue * mvalue <= 0:  # 零点在左侧
            right, rvalue = mid, mvalue 
        mid = (right + left) / 2
        mvalue = f(mid, *params)
        step += 1
        if step >= 100:
            print("max while loop number exceeded, {}".format(params))
            break
    return mid


"""  不动点问题  """

import jax.numpy as jnp
from jax import grad, custom_vjp

l, r  = 0.0, 5.0  # 二分法求解区间

def f(M0, T, H0, C = 1.0, Tc = 0.0):
    return M0 - jnp.tanh((C * H0 + Tc * M0) / T)

@custom_vjp
def M(T, H, C = 1.0, Tc = 0.0):
    params = (T, H, C, Tc)
    M_fixed = BisectionSolver(f, l, r, params)
    return M_fixed

# 在前向传播时储存反向传播时的需要信息
def M_fwd(T, H, C = 1.0, Tc = 0.0):
    params = (T, H, C, Tc)
    M_fix = BisectionSolver(f, l, r, params)
    res = (M_fix, *params)
    return M_fix, res

# 在反向传播时指定构造节点的更新方式
def M_bwd(res, g):
    M_fix, T, H, C, Tc = res
    dfdM = grad(f, argnums=0)(M_fix, T, H, C, Tc)
    dfdT = grad(f, argnums=1)(M_fix, T, H, C, Tc)
    dfdH = grad(f, argnums=2)(M_fix, T, H, C, Tc)
    dfdC = grad(f, argnums=3)(M_fix, T, H, C, Tc)
    dfdTc = grad(f, argnums=4)(M_fix, T, H, C, Tc)

    dMdT = - dfdT / dfdM
    dMdH = - dfdH / dfdM
    dMdC = - dfdC / dfdM
    dMdTc = - dfdTc / dfdM
    return (dMdT * g, dMdH * g, dMdC * g, dMdTc * g)

M.defvjp(M_fwd, M_bwd)



"""   图像绘制  """
def M_limit(T, H, C = 1.0, Tc = 0.0):
    return C / (T-Tc) * H

dM = grad(M_limit, argnums=1)
print("hi~")
print(dM(1.,1.))

H0 = 2.
T0 = 1.

M_T_list = []
M0_T_list = []
T_arr = jnp.linspace(1,10,51)
H_arr = jnp.linspace(0.1,2,51)
for t in T_arr:
    M_T_list.append(M(t, H0))
    M0_T_list.append(M_limit(t,H0))

M_H_list = []
M0_H_list = []
for h in H_arr:
    M_H_list.append(M(T0, h))
    M0_H_list.append(M_limit(T0, h))

dMdH_list = []
dM0dH_list = []
dMdH = grad(M, argnums=1)
dM0dH = grad(M_limit, argnums=1)
for h in H_arr:
    dMdH_list.append(dMdH(T0, h))
    dM0dH_list.append(dM0dH(T0, h))

import matplotlib.pyplot as plt


""" M-T.png """
# plt.plot(T_arr, M_T_list, label = "M-T", linewidth = 3)
# plt.plot(T_arr, M0_T_list, label = "Curie-Weiss law", linestyle = ":", linewidth = 3)
# plt.legend(loc= "upper right")
# plt.xlabel("T")
# plt.ylabel("M")
# plt.grid("-")
# plt.savefig("M-T.png")
# plt.show()

""" M-H.png """
# plt.plot(H_arr, M_H_list, label = "M-H", linewidth = 3)
# plt.plot(H_arr, M0_H_list, label = "Curie-Weiss law", linestyle = ":", linewidth = 3)
# plt.legend(loc= "lower right")
# plt.xlabel("H")
# plt.ylabel("M")
# plt.grid("-")
# plt.savefig("M-H.png")
# plt.show()

""" dM-dH.png """
plt.plot(H_arr, dMdH_list, label = "dM/dH", linewidth = 3)
plt.plot(H_arr, dM0dH_list, label = "Curie-Weiss law", linestyle = ":", linewidth = 3)
plt.legend(loc= "lower left")
plt.xlabel("H")
plt.ylabel("M")
plt.grid("-")
plt.savefig("dM-dH.png")
plt.show()

""" examples """

# @custom_vjp
# def log1pexp(x):
#     return jnp.log(1. + jnp.exp(x))

# # 反向模式
# def log1pexp_fwd(x):
#     return log1pexp(x), x

# def log1pexp_bwd(res, g):
#     x = res
#     return ((1. - 1./(1.+jnp.exp(x))) * g, )

# log1pexp.defvjp(log1pexp_fwd, log1pexp_bwd)

# print(log1pexp(0.))         # 0.6931472 ~ ln2
# print(grad(log1pexp)(100.)) # 1.0

