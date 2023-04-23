


"""
代码示例 B.4 :
   super 函数的实质

    转载自 https://www.jianshu.com/p/de7d38c84443

"""

def super(cls, inst):
    mro = inst.__class__.mro()
    return mro[mro.index(cls) + 1]
    