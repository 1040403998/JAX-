


"""
代码示例 4.2 :
    torch 的 grad 函数
    
    note: 需要使用 pip install torch 安装torch库
"""

import torch
from torch import autograd
grad = autograd.grad

def f(x,y):
    return (x + y) ** 2

x, y = 1.0, 2.0

x = torch.tensor(x, requires_grad=True)
y = torch.tensor(y, requires_grad=True)
z = f(x, y)

# create_graph 参数用于指定在求导时 对导函数构建计算图
df1  = grad(z, x, create_graph = True)[0]
df2  = grad(z, y, create_graph = True)[0]

# retain_graph 参数用于指定在反向传播时进行计算图的保存
df11 = grad(df1, x, retain_graph=True, create_graph=True)[0]
df12 = grad(df1, y, retain_graph=True)[0]
df21 = grad(df2, x, retain_graph=True)[0]
df22 = grad(df2, y, retain_graph=True)[0]

df111 = grad(df11, x)[0]

"""  第零阶  """
print(z.item())      # >>  9.0

"""  第一阶  """
print(df1.item())    # >>  6.0
print(df2.item())    # >>  6.0

"""  第二阶  """
print(df11.item())    # >>  2.0
print(df12.item())    # >>  2.0
print(df21.item())    # >>  2.0
print(df22.item())    # >>  2.0

"""  第三阶  """
print(df111.item())   # >>  0.0
