
"""
代码示例B.2:
    有向无环图的拓扑排序
"""

from typing import List

class VertexInfo(object): 
    pass

class Vertex(object):
        
    def __init__(self,  info: VertexInfo = None,
                adjacentList: List = None):
        """ 节点的初始化 """
        # info: 用于储存节点的信息
        self.info: VertexInfo = info 
        # adjacent_list: 用于储存与该节点相邻的邻接表
        self.adjacent_list: List[Vertex] = adjacentList  

    @property
    def n_adjacent(self,):
        return len(self.adjacent_list)


class Edge(object):
    def __init__(self, v1: Vertex, v2: Vertex):
        self.v1 = v1
        self.v2 = v2
        
class Graph(object):
    def __init__(self, vertex_list: List[Vertex]=[], 
                       edge_list  : List[Edge  ]=[]):
        self.vertex_list: List[Vertex] = vertex_list
        self.edge_list  : List[Edge  ] = edge_list

    @property
    def n_edge(self):
        return len(self.edge_list)

    @property
    def n_vertex(self):
        return len(self.vertex_list)


""" 拓扑排序的代码实现 """
FASTnet = Graph(vertex_list = [], edge_list = [])

# 标记节点
for v in FASTnet.vertex_list:
    v._processed = False

# 定义计算节点入度的函数
def indegree(v: Vertex):
    n = 0
    for vbar in v.adjacent:
        if v.z < vbar.z and not vbar._processed:
            n += 1
    return n

# 初始化队列 L
## -- 它将包含所有已经排列元素，初始化为空队列
from collections import deque
L = deque() 

# 初始化列表 S
## -- 它存放程序运行过程中所有入度为零的节点（同时被按照此规则被初始化）
S: List[Vertex] = []
for v in FASTnet.vertex_list:
    if indegree(v) == 0:
        S.append(v)
print("初始化列表S的长度", len(S))

# 拓扑排序
while len(S) != 0:
    v = S.pop()
    v._processed = True  # 相当于删除节点
    L.append(v)
    for vbar in v.adjacent:
        # 无需再删除边
        if indegree(vbar) == 0 and not vbar._processed: # 考察节点vbar是否还有其他入边
            S.append(vbar)

if len(L) != FASTnet.n_vertex:
    print("图中至少有一个环")
else:
    print("拓扑排序完成")

# 简单测试： 检查L中是否有相同的节点
for idx, v in enumerate(L):
    for i in range(idx+1, len(L)):
        if L[i].name == v.name:
            print(v)
            for vbar in v.adjacent:
                print(vbar)



