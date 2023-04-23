


"""
代码示例 3.18
	爱因斯坦求和 及 其他

"""


import jax.numpy as jnp


"""   线性代数    """

vec = jnp.array([3.,4.])
print(jnp.linalg.norm(vec, ord=-50))  #  min(3, 4)
print(jnp.linalg.norm(vec, ord=1))    #  3 + 4
print(jnp.linalg.norm(vec, ord=2))    #  sqrt(3^2 + 4^2)
print(jnp.linalg.norm(vec, ord=50))   #  max(3, 4)

# 以下是一些测试的样例：
mat = jnp.array([[1., 2., 3.], 
				 [0., 3., 4.],
				 [0., 0., 5.]])
				 
mat_inv = jnp.linalg.inv(mat)       # 矩阵的逆
mat_det = jnp.linalg.det(mat)       # 矩阵的行列式
mat_eigval, mat_eigvec = jnp.linalg.eig(mat)  # 矩阵的特征值和特征向量(复数)
rank = jnp.linalg.matrix_rank(mat)  # 矩阵的秩
u, s, vh = jnp.linalg.svd(mat)      # 矩阵的奇异值分解 mat = u @ np.diag(s) @ vh 
print(mat_eigval.real)



"""  代码示例3.18  爱因斯坦求和 """

# 取模：jnp.linalg.norm
A = jnp.ones(shape = (3,4,5))
B = jnp.ones(shape = (1,4,6))
C = jnp.ones(shape = (3,7))

D = jnp.einsum("abc,abd,ae->dce", A, B, C)
print(D.shape)
# >> (6,5,7)

