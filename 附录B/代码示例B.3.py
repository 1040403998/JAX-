


"""
代码示例 B.3 :
    Python类的继承顺序

   object
     |
    base
    /  \
   /    A
  |    / \
  B   C  D
   \ /  /
    E  /
    \ /
     F
"""


class Base(object):
    def __init__(self):
        print("enter Base")
        print("leave Base")


class A(Base):
    def __init__(self):
        print("enter A")
        super(A, self).__init__()
        print("leave A")


class B(Base):
    def __init__(self):
        print("enter B")
        super(B, self).__init__()
        print("leave B")


class C(A):
    def __init__(self):
        print("enter C")
        super(C, self).__init__()
        print("leave C")


class D(A):
    def __init__(self):
        print("enter D")
        super(D, self).__init__()
        print("leave D")


class E(B, C):
    def __init__(self):
        print("enter E")
        super(E, self).__init__()
        print("leave E")

class F(E, D):
    def __init__(self):
        print("enter F")
        super(F, self).__init__()
        print("leave F")


print(F.mro())


f = F()
print(super(B, f).__init__)
# >> <bound method C.__init__ of <__main__.F object at 0x7f150a7efc10>>