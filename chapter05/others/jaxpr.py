"""
代码示例5.2.1 :
   jit: 概述
   
   例子: make_jaxpr编译函数 
"""
# make_jaxpr编译函数 
from jax import make_jaxpr  
import jax.numpy as jnp 

def func1(first, second):  
    temp = first + jnp.sin(second) * 3.  
    return jnp.sum(temp)  

arr1 = jnp.zeros(8)
arr2 = jnp.ones(8)
expr = make_jaxpr(func1)(arr1, arr2)
print(expr)

# { lambda ; a:f32[8] b:f32[8]. let  
#     c:f32[8] = sin b  
#     d:f32[8] = mul c 3.0  
#     e:f32[8] = add a d  
#     f:f32[] = reduce_sum[axes=(0,)] e  
#   in (f,) }  
