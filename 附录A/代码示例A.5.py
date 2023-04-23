

"""
代码示例 A.5 :
    Python 子类的初始化
"""

class Point(object):
    def __init__(self, x: float, 
                       y: float, 
                       z: float, 
                       name: str = None):
        # 用 name 和 坐标参数 初始化一个Point类的实例
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def __eq__(self, other):
        # 如果 P1 和 P2 都是 Point类，且名称（也就是name参数）相同，
        # 我们就认为两个点是相同的，即 P1 == P2 将会返回 True
        if type(self) != type(other): return False
        return self.name == other.name
    
    def __str__(self):
        # 指定调用print函数时，程序终端将会输出的string
        return "class Point:\n\t{}:({},{},{})"\
            .format(self.name, self.x, self.y, self.z)

from typing import List

class Vertex(Point):
    def __init__(self, name: str, 
                          x: float = None, 
                          y: float = None, 
                          z: float = None):
        super().__init__(x, y, z, name = name)  # 初始化顶点的名称及位置参数
        self.adjacent: List[Vertex] = []

    def __str__(self):
        # 指定调用print函数时，程序终端将会输出的string
        return "class Vertex:\n\tname: {}\n\tposition: {}\n\tadjacent vertex: {}".\
        format(self.name, (self.x, self.y, self.z), [_.name for _ in self.adjacent])
    
    @property
    def n_adjacent(self):
        return len(self.adjacent)