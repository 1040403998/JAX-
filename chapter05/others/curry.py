

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



"""Curries arguments of f, returning a function on any remaining arguments.

# For example:
# >>> f = lambda x, y, z, w: x * y + z * w
# >>> f(2,3,4,5)
# 26
# >>> curry(f)(2)(3, 4, 5)
# 26
# >>> curry(f)(2, 3)(4, 5)
# 26
# >>> curry(f)(2, 3, 4, 5)()
# 26
# """