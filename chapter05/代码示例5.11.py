


"""
代码示例 5.11 :
    柯里化及其他
"""

from functools import partial

def curry(f):
  return partial(partial, f)

def uncurry(fun):
  return fun()

f = lambda x, y, z, w: x * y + z * w

print(f(1,2,3,4))           # >> 14
print(curry(f)()(1,2,3,4))  # >> 14
print(curry(f)(1,)(2,3,4))  # >> 14
print(curry(f)(1,2,)(3,4))  # >> 14
print(curry(f)(1,2,3,)(4))  # >> 14
print(curry(f)(1,2,3,4)())  # >> 14



"""  其他代码示例  """


z = (lambda x: (x, x))("y")
print(z)  # ('y', 'y')

F = lambda x: (lambda y: (x, y))
print(F("a")("b"))  # ('a', 'b')

F = curry(lambda x,y : (x, y))
print(F("a")("b"))  # ('a', 'b')


G = lambda f: (lambda n: 0 if n==0 else n + f(f)(n-1))
Y = lambda g: g(g)
G = G(G)
print(G(100))

G = lambda f, n: 0 if n==0 else n + f(n-1)
Y = lambda f: (lambda m : (lambda x: (lambda n: f(x(x), n)))
                          (lambda x: (lambda n: f(x(x), n)))(m))
print(Y(G)(100))


""" 死循环 """
# G = lambda f: (lambda n: 0 if n==0 else n + f(n-1))
# Y = lambda f: (lambda x: f(x(x)))(lambda x: f(x(x)))
# print(Y(G)(100))


def fori_loop(lower, upper, body_fun, init_val):  
  val = init_val  
  for i in range(lower, upper):  
    val = body_fun(i, val)  
  return val  

lower = 0
upper = 100+1
body_fun = lambda i, val: val + i
init_val = 0
print(fori_loop(lower, upper, body_fun, init_val))


Y = lambda f: (lambda *m : (lambda x: (lambda *n: f(x(x), *n)))
                           (lambda x: (lambda *n: f(x(x), *n)))(*m))
fori_loop = Y(lambda g,a,b,f,v : v if b <= a else g(a+1, b, f, f(a, v)))

# 一行代码解决fori_loop
fori_loop = (lambda f: (lambda *m : (lambda x: (lambda *n: f(x(x), *n)))
                                    (lambda x: (lambda *n: f(x(x), *n)))(*m))) \
                                    (lambda g,a,b,f,v : v if b <= a else g(a+1, b, f, f(a, v)))

# 测试 
lower = 0
upper = 101
body_fun = lambda i, val: val + i
init_val = 0
print(fori_loop(lower, upper, body_fun, init_val)) # 5050


# 一行代码解决 while_loop
while_loop = (lambda f: (lambda *m : (lambda x: (lambda *n: f(x(x), *n)))
                                     (lambda x: (lambda *n: f(x(x), *n)))(*m))) \
                                     (lambda h,g,f,v : h(g,f,f(*v)) if g(*v) else v)

# 测试 
cond_fun = lambda v, s: v < 101
body_fun = lambda v, s: (v + 1, s + v)
init_val = (0, 0)
print(while_loop(cond_fun, body_fun, init_val)) # (101, 5050)
