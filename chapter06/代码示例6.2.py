"""
代码示例6.2:
    vmap函数的等价for循环形式
"""

import numpy as np

batch = 8
a, b, c, d, e, f = 2, 3, 4, 5, 6, 7

in_axes = (None, 3, 4, 5, None)
out_axes = (0, 1, 2, None)

A = np.ones((a, b, c, d, e, f))
B = np.ones((a, b, c, batch, d, e, f))
C = np.ones((a, b, c, d, batch, e, f))
D = np.ones((a, b, c, d, e, batch, f))
E = np.ones((a, b))

def func(A, B, C, D, E):
    return (
        A + B,
        B - C,
        C * D,
        E * 2,
    )
    
ans0 = np.zeros((batch, a, b, c, d, e, f))
ans1 = np.zeros((a, batch, b, c, d, e, f))
ans2 = np.zeros((a, b, batch, c, d, e, f))
ans3 = np.zeros((a, b))
ans4 = np.zeros((batch, a, b))

for i in range(batch):
    a1, a2, a3, a4 = func(A, B[:,:,:,i], C[:,:,:,:,i], D[:,:,:,:,:,i], E)
    ans0[i] = a1
    ans1[:,i] = a2
    ans2[:,:,i] = a3
ans3[:] = a4

ans = (ans0, ans1, ans2, ans3)
print(ans[0].shape)  # (batch, a, b, c, d, e, f)
print(ans[1].shape)  # (a, batch, b, c, d, e, f)
print(ans[2].shape)  # (a, b, batch, c, d, e, f)
print(ans[3].shape)  # (a, b)
