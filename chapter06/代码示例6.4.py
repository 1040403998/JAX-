"""
代码示例6.4:
    嵌套vmap的for等价形式
"""

import numpy as np
batch1 = 8
batch2 = 6
a, b, c = 2, 3, 4
As = np.ones((batch1, batch2, a ,b))
Bs = np.ones((batch1, b, c))

def f(x, y):
    a = np.dot(x, y)
    b = np.tanh(a)
    return b

def batched_f(As, Bs):
    ans_outer = np.zeros((batch1, batch2, a, c))

    for i_outer in range(batch2):
        # in_axes = (1, None), out_axis = 1
        As_inner = As[:, i_outer]
        Bs_inner = Bs
        ans_inner = np.zeros((batch1, a, c))

        for i_inner in range(batch1):
            ans_inner[i_inner] = f(As_inner[i_inner], Bs_inner[i_inner])

        ans_outer[:, i_outer] = ans_inner
    
    return ans_outer

ans = batched_f(As, Bs)
print(ans.shape)  # (batch1, batch2, a, c)
