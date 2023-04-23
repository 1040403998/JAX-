


"""
代码示例5.5 :
    使用make_jaxpr 打印表达式 
"""
# make_jaxpr编译函数 
import jax.numpy as jnp 
from jax import make_jaxpr  

def func(first, second):  
    temp = first + jnp.sin(second) * 3.
    return jnp.sum(temp)

arr1 = jnp.zeros(8)
arr2 = jnp.ones(8)
expr = make_jaxpr(func)(arr1, arr2)
print(expr)


"""
{ lambda  ; a b.
  let c = sin b
      d = mul c 3.0
      e = add a d
      f = reduce_sum[ axes=(0,) ] e
  in (f,) }
"""
print(expr.in_avals)  # [ShapedArray(float32[8]), ShapedArray(float32[8])]
print(expr.out_avals) # [ShapedArray(float32[])]
print(expr.consts)    # []
print(expr.eqns)      # [c = sin b, d = mul c 3.0, e = add a d, f = reduce_sum[ axes=(0,) ] e]
