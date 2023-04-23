

import jax

# 函数式

def f(x): return x + 1
print(f(1))  # 2


# 非函数式（无法编译）
a = 0
def g(x):
  global a
  a += x
  return x + a

print(g(1)) # 2
print(g(1)) # 3
print(g(1)) # 4

def gg(a,x):
  a = a + x
  return a + x

def uncurry(fun):
  return fun()

@jax.jit
@uncurry
def h(b = 0):
  b += 1
  def g(x):
    return x + b
  return g

print(h(1))  # 3
print(h(1))  # 3


s = 0
for i in range(10):
  s = s + i 

i = 1
s = 0
while i < 11:
  i = i + 1
  s = s + i

def f(x):
  if x < 0: return 0
  if x >=0: return 1


result = 0
for n in [1,2,3]:
  result += n * (n+1) / 2
print(result)  # 10.0

