

"""
代码示例 A.6 :
    与佩奇先生的博弈
"""


""" ==================    生成方  ==================  """

warning = "input value should be {}, get {} instead"  

class MyFloat:  
    
    """ 初始化部分 """
    def __init__(self, value):  
        try:  
            value = float(value)  
        except TypeError:  
            print(warning.format("float or int", type(value).__name__))  
        finally:  
            self.value = value  
    
    """ str, repr, print 部分"""
    def __str__(self):
        return str(self.value)
    
    __repr__ = __str__

    """ 算数运算符 部分"""
    # 加法 +
    def __add__(self, other) : return self.value + other
    def __radd__(self, other): return self.__add__(other)
    
    # 减法 -
    def __sub__(self, other) : return self.value - other
    def __rsub__(self, other): return other - self.value

    # 乘法 *
    def __mul__(self, other) : return self.value * other
    def __rmul__(self, other): return self.value * other

    # 除法 /
    def __truediv__(self, other):  return self.value / other
    def __rtruediv__(self, other): return other / self.value

    # 乘方 **
    def __pow__(self , pow) : return self.value ** pow
    def __rpow__(self, base): return base ** self.value

    # 取模 %
    def __mod__(self, other): return self.value % other
    def __rmod__(self, other): return other % self.value

    # 取整除 //
    def __floordiv__(self, other) : return self.value // other
    def __rfloordiv__(self, other): return other // self.value


    """ 赋值运算符 部分"""
    # 加法赋值  += 
    def __iadd__(self, other): return self.value + other
    
    # 减法赋值  -=
    def __isub__(self, other): return self.value - other

    # 乘法赋值  *=
    def __imul__(self, other): return self.value * other

    # 除法赋值  /=
    def __idiv__(self, other): return self.value / other

    # 乘方赋值  **=
    def __ipow__(self, other): return self.value ** other

    # 取模赋值  %=
    def __imod__(self, other): return self.value % other

    # 整除赋值  //=
    def __idiv__(self, other): return self.value // other

    """ 比较运算符 部分"""
    # 小于 <
    def __lt__(self, other): return self.value < other
    
    # 小于等于 <=
    def __le__(self, other): return self.value <= other

    # 等于 ==
    def __eq__(self, other): return self.value == other

    # 不等于 !=
    def __neq__(self, other): return self.value != other

    # 大于等于  >=
    def __ge__(self, other): return self.value >= other

    # 大于 >
    def __gt__(self, other): return self.value > other


    """ 调用行为"""
    def __call__(self, *args, **kwargs):
        raise TypeError("'float' object is not callable")


a = MyFloat(1.0)
b = 1.0

""" ==================    测试方  ==================  """

import unittest

class FloatTest(unittest.TestCase):
    
    def setUp(self):
        """ 在测试开始时执行 """
        self.info = "可以通过{}操作区分 a 和 b"
        
    def tearDown(self):
        """ 在测试结束时执行 """
        pass
    
    def testPrint(self,):
        """ print, str 及 repr 的测试"""
        self.assertEqual(str(a) , str(b) , msg = self.info.format("str 函数"))
        self.assertEqual(repr(a), repr(b), msg = self.info.format("repr函数"))
    
    def testArithmetic(self,):
        """ 运算算数符测试"""
        # 加法
        self.assertEqual(a + 1.0, b + 1.0, self.info.format("加法"))
        self.assertEqual(1.0 + a, 1.0 + b, self.info.format("右加法"))

        # 减法
        self.assertEqual(a - 1.0, b - 1.0, self.info.format("减法"))
        self.assertEqual(1.0 - a, 1.0 - b, self.info.format("右减法"))

        # 乘法
        self.assertEqual(a * 2.0, b * 2.0, self.info.format("乘法"))
        self.assertEqual(2.0 * a, 2.0 * b, self.info.format("右乘法"))
        
        # 除法
        self.assertEqual(a / 2.0, b / 2.0, self.info.format("除法"))
        self.assertEqual(2.0 / a, 2.0 / b, self.info.format("右除法"))

        # 乘方
        self.assertEqual(a ** 2.0, b ** 2.0, self.info.format("乘方"))
        self.assertEqual(2.0 ** a, 2.0 ** b, self.info.format("右乘方"))

        # 取模
        self.assertEqual(a % 2, b % 2, self.info.format("取模"))
        self.assertEqual(2 % a, 2 % a, self.info.format("'右'取模"))

        # 取整除
        self.assertEqual(a // 2, b // 2, self.info.format("取整除"))
        self.assertEqual(2 // a, 2 // a, self.info.format("'右'取整除"))

    def testAssignment(self,):
        """ 赋值运算符测试 """
        pass

    def testComparison(self,):
        """ 比较运算符的测试 """
        # 小于 <
        self.assertTrue(a < 2., msg = "a 是内鬼")
        self.assertTrue(b < 2., msg = "b 是内鬼")
        # 大于 >
        self.assertTrue(a > 0., msg = "a 是内鬼")
        self.assertTrue(b > 0., msg = "b 是内鬼")
        # 等于 ==
        self.assertTrue(a == 1., msg = "a 是内鬼")
        self.assertTrue(b == 1., msg = "b 是内鬼")
        # 不等于 !=
        self.assertTrue(a != -1., msg = "a 是内鬼")
        self.assertTrue(b != -1., msg = "b 是内鬼")
        # 大于等于 >=
        self.assertTrue(a >= 0.5, msg = "a 是内鬼")
        self.assertTrue(b >= 0.5, msg = "b 是内鬼")
        # 小于等于 >=   
        self.assertTrue(a <= 2., msg = "a 是内鬼")
        self.assertTrue(b <= 2., msg = "b 是内鬼")



if __name__ == "__main__":
    unittest.main()
