

"""
代码示例 A.4 :
    Python 类的继承
"""


class Number(object):
        
    _is_number = True

    def __init__(self, value, name = None):
        self.value = value
        self.name = name

    def get_value(self):
        return self.value

    def get_name(self):
        name = self.name if self.name is not None else str(self.value)
        return name

    def __add__(self, other):
        return Add(self, other)


class Add(Number):

    _operand_name = " plus "
    
    def __init__(self, v1, v2):
        """
        由两个已知的实例 v1 和 v2, 初始化一个 Add 类的实例.
        -- 输入的参数 v1 和 v2 , 要么属于 Number 类，要么属于继承自 Number 类的子类.
        """
        self.value = v1.value + v2.value
        name_1 = v1.get_name()
        name_2 = v2.get_name()
        name_op = self._operand_name
        self.name = "{}{}{}".format(name_1, name_op, name_2)

if __name__ == "__main__":
    v1 = Number(1, name = "1")
    v2 = Number(2, name = "2")
    v3 = Add(v1, v2)

    print(v3.get_name())   # >> 1 plus 2
    print(v3.get_value())  # >> 3

    v4 = v1 + v3
    v5 = v4 + v2

    print(v4.get_value())  # >> 4
    print(v4.get_name())   # >> 1 plus 1 plus 2
    print(v4.get_value())  # >> 6
    print(v5.get_name())   # >> 1 plus 1 plus 2 plus 2

    print(type(v5).mro())

