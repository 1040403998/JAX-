# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

def stream(lines, lr,lz,rbins,zbins,u,R,threshold, density):
    """
      lines: 流场线的数量
      lr/lz：起止坐标
      rbins/zbins：数据分辨率
      u：流体速度
      R：小球大小
      threshold：过滤局部速度过快的流场线
      density：绘图密度
    """
    r = np.linspace(-lr, lr, rbins)
    z = np.linspace(-lz, lz, zbins)
    rv, zv = np.meshgrid(r, z)  
    frac_3_4 = 3/4
    rzu = rv*zv*u
    sqrt_r_z = np.sqrt(rv**2+zv**2)
    ur_1 = frac_3_4*R**3*rzu/sqrt_r_z**5
    ur_2 = frac_3_4*R*rzu/sqrt_r_z**3
    ur = ur_1 - ur_2 
    uz1 = R**3/4 * (3*u*zv**2/sqrt_r_z**5 - u/sqrt_r_z**3)
    uz2 = 3/4 * R * (u/sqrt_r_z + u*zv**2/sqrt_r_z**3)
    uz = uz1 + u - uz2 
    ur[np.abs(ur)>threshold] = np.nan
    uz[np.abs(uz)>threshold] = np.nan
    plt.figure(figsize=(10, 10))  # 生成图的大小
    # plt.streamplot(rv, zv, ur, uz, density, arrowsize=2, )
    s = np.zeros((lines,2))
    half_lines = int(lines/2)
    s[:half_lines,0]=np.linspace(-3,-1.2,half_lines)
    s[half_lines:,0]=np.linspace(1.2,3, half_lines)
    plt.streamplot(rv, zv, ur, uz, density, arrowsize=2, start_points=s, color='red')

    # 绘制中心球形
    circle = mpatches.Circle([0,0], radius=0.8*R, color='blue')
    ax = plt.gca()
    collection = PatchCollection([circle])
    ax.add_collection(collection)

    # plt.annotate(text=r"$\phi(\alpha)=f(\vec{\theta}_k+\alpha_k \vec{p}_k)$", 
    #              xy=(), xytext=(0, 0), textcoords = "offset points")

    # plt.annotate(text= "", arrowprops=dict(arrowstyle="<|-"), xy=(0, 0), xytext=(0, 1.5))
    # plt.annotate(text= "", arrowprops=dict(headlength=0.5, headwidth=0.5, tailwidth=0.2), xy=(0, 1.5), xytext=(0, 0))
    ax.arrow(0, 0.05, 0, 1.5, head_width=0.1, head_length=0.3, shape="full",fc='black',ec='black',alpha=0.9, overhang=0.5)
    ax.arrow(0,-0.05, 0,-1.5, head_width=0.1, head_length=0.3, shape="full",fc='black',ec='black',alpha=0.9, overhang=0.5)

u = 10
stream(16, 5, 3, 200, 200, u=u, R=1, threshold=10*u, density=5)
plt.savefig("ball in stream.png")