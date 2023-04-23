

"""
代码示例 A.2 :
    用构造函数 __new__ 实现 Python 的单例
"""


class Zero(object):

    _instance = None  # 用于存储已经创建的实例

    def __new__(cls, *args):
        if cls._instance == None:
            # 如果实例未被创建，则用object.__new__函数创建一个新的实例
            obj = object.__new__(cls)
            obj.value = 0
            obj.name = "zero 0"
            cls._instance = obj
            return obj
        # 如果实例已经被创建，则直接将之前创建的实例返回
        return cls._instance

    def __init__(self):
        pass

zero1 = Zero()
zero2 = Zero()

# 1. 所有实例的id恒等
print(zero1 == zero2)   # 返回：True
# 2. 改变其中一个实例的参数，将同时改变所有实例的参数
print(zero2.name)       # 返回：zero 0
zero1.name = "zero 1"
print(zero2.name)       # 返回：zero 1



