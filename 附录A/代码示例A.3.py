

"""
代码示例 A.3 :
    Python在构造整数时用到了单例
"""


number = 0
a, b = 0, 0

while id(a) == id(b) and number < 1000:
    exec("a = {}".format(number))
    exec("b = {}".format(number))
    if id(a) != id(b):
        print(number)  # >> 257
    number += 1

number = 0
a, b = 0, 0
while id(a) == id(b) and number > -1000:
    exec("a = {}".format(number))
    exec("b = {}".format(number))
    if id(a) != id(b):
        print(number)  # >> -6
    number -= 1

