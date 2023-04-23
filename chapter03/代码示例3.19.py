


"""
代码示例 3.19
    爱因斯坦求和(等价形式)
"""

import numpy as np  # 因为这里需要使用原地更新，所以引用的是numpy库

a, b, c, d, e = 3, 4, 5, 6, 7
A = np.ones(shape = (a,b,c))
B = np.ones(shape = (1,b,d))
C = np.ones(shape = (a,e))

D = np.zeros(shape = (d,c,e))

# 在等号异侧的指标不作求和
for id in range(d):
	for ic in range(c):
		for ie in range(e):
			
			# 在等号同侧的指标代表求和
			for ia in range(a):
				for ib in range(b):
					D[id,ic,ie] += A[ia,ib,ic] * B[:,ib,id] * C[ia,ie]
