
"""
代码示例B.1 :
    无向图的数据结构
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

