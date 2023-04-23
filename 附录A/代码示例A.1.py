

"""
代码示例 A.1 :
    构造函数 __new__ 的测试
"""


class Number(object):
    # 构造函数
    def __new__(cls, *args):
        print("function __new__  is called:")

        obj = object.__new__(cls)
        print("\t args in function __new__  :", args)  # 打印参数
        print("\t id(obj)  = ", id(obj))               # 打印 id(obj)
        print("\t id(cls)  = ", id(cls))               # 打印 id(cls)
        return obj

    # 初始化函数
    def __init__(self, value, name = None):
        print("function __init__ is called:")
        print("\t args in function __init__ :", (value, name))     # 打印参数
        print("\t id(self)    = ", id(self))                       # 打印 id(self)
        self.value = value
        self.name = name

if __name__ == "__main__":
    zero = Number(0, "Zero")
    print("\t id(Number)  = ", id(Number))                # 打印 id(Number)



