"""
代码示例5.7:
    for 循环的展开
"""
import jax.numpy as jnp
from jax import make_jaxpr, jit

# for循环的展开
def f(x):
    for i in range(10):
        x += i
    return x
jitted_f = jit(f)

# 测试
arr = jnp.array([0.0, 1.0, 2.0])
print(jitted_f(arr))      # [45. 46. 47.]
print(make_jaxpr(f)(arr))

"""
{ lambda  ; a.
  let b = add a 0.0
      c = add b 1.0
      d = add c 2.0
      e = add d 3.0
      f = add e 4.0
      g = add f 5.0
      h = add g 6.0
      i = add h 7.0
      j = add i 8.0
      k = add j 9.0
  in (k,) }
"""


