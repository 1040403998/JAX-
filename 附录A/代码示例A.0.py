

"""
代码示例 A.0 :
    Python类的基本内容
"""


class Number(object):
    
    """class Number's documents"""
    
    # 成员变量
    _is_number = True   

    # 初始化方法
    def __init__(self, value, name = None):
        self.value = value
        self.name = name
        
    # 成员函数
    def add(self, x):
        """function add's documents"""
        print(f"Adding {x} to {self.value}")
        self.value += x

    def get_value(self):
        return self.value

if __name__ == "__main__":
    zero = Number(value=0, name="Zero")
    zero.add(1)                #   Adding 1 to 0
    print(zero.get_value())    #   1

    print(Number.__class__)    #   <class 'type'>
    print(Number.__bases__)    #   (<class 'object'>,)
    print(Number.__name__)     #   Number
    print(Number.__module__)   #   __main__
    print(Number.__dir__)      #   <method '__dir__' of 'object' objects>
    print(Number.__doc__)      #   class Number's documents
    print(Number.add.__doc__)  #   function add's documents
    print(dir(Number))         #   ['_is_number', 'add', 'get_value', ...]
    print(dir(zero))           #   [ 'name', 'value', '_is_number', 'add', 'get_value', ...]



