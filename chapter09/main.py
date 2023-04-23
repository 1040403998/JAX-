


""" 数据的读取及打印 """

import enum
import os
import pandas as pd  
  
# 设置文件名称
dir_path = os.path.dirname(__file__)    # 获取当前文件所在文件夹的路径
print("dir_path = " , dir_path)

# 数据读取
vertex_frame        = pd.read_csv(os.path.join(dir_path, 'data/vertex.csv')       ,header=0,sep=',')  
ground_vertex_frame = pd.read_csv(os.path.join(dir_path, 'data/vertex_ground.csv'),header=0,sep=',')  
face_vertex_frame   = pd.read_csv(os.path.join(dir_path, 'data/face.csv')         ,header=0,sep=',')  
  
# 数据打印  
print(vertex_frame.head())  
print(ground_vertex_frame.head())  
print(face_vertex_frame.head())


""" 数据的预处理 """
import jax
import numpy as np
import jax.numpy as jnp
import logging

from jax.config import config
config.update("jax_enable_x64",True)

from typing import Union, List, Optional, Set, Callable
ScalorType = Union[float, np.ndarray, jnp.ndarray]
LengthType = Union[float, np.ndarray, jnp.ndarray]

class Point(object):
    def __init__(self, x: Optional[ScalorType], 
                       y: Optional[ScalorType] = 0., 
                       z: Optional[ScalorType] = 0., 
                       name: str = None):
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def __eq__(self, other):
        """
        如果 P1 和 P2 都是 Point类, 且名称(name参数)相同, 我们就认为两个点是相同的， 
            此时 P1 == P2 将会返回 True
        """
        if type(self) != type(other): return False
        return self.name == other.name
    
    def __str__(self):
        # 指定调用print函数时，程序终端将会输出的string
        return "class Point:\n\t name = {} \n\t position = ({}, {}, {})"\
                .format(self.name, self.x, self.y, self.z)
    
    @property
    def position(self):
        return (self.x, self.y, self.z)

Origin = Point(0., 0., 0., name="Origin")
print(Origin)

class VertexInfo(object):
    def __init__(self, index:int = -1,
                       status: int = -1,
                       vertex_init_pos  :np.ndarray = None,
                       vertex_ground_pos:np.ndarray = None):
        self.idx: int = index      # 储存该顶点在输入 3xN 顶点列表之中的位置
        self._status: int = status # 描述该节点目前的状态, 
            # 0表示位于工作区，1表示位于工作区外（但不固定），2表示该节点固定
        self.vertex_init_pos  : np.ndarray = vertex_init_pos   # 顶点的初始位置
        self.vertex_ground_pos: np.ndarray = vertex_ground_pos # 地面上促动器的坐标 
    
    @property
    def status(self):
        if self._status == 0: return "working"
        if self._status == 1: return "idling"
        if self._status == 2: return "fixing"
        return "unsure"

class Vertex(Point):
    def __init__(self, name: str, 
                          x: ScalorType = None, 
                          y: ScalorType = None, 
                          z: ScalorType = None,
                          info:VertexInfo = None):
        super().__init__(x, y, z, name = name)  # 初始化顶点的名称及位置参数
        self.adjacent: List[Vertex] = []
        self.info : VertexInfo = info

    def __eq__(self, other):
        # 如果Vertex的名称(name参数)相同，我们就认为两个Vertex是一样的
        if type(self) != type(other): return False
        if self.name == other.name:
            # 如果两个顶点的名称相同，我们将会检查这两个节点的坐标和相邻节点是否相同
            # 如若不同，将会在终端产生warning, 但程序并不报错
            if self.x != other.x or self.y != other.y or self.z != other.z:
                logging.warning("Inconsistent position between Vertexes {}:\n"
                "  [self]: {}\n [other]: {}".format(self.name, 
                (self.x, self.y, self.z), (other.x, other.y, other.z)))
            if self.adjacent != other.adjacent:
                logging.warning("Inconsistent adjacent list between Vertex {}:\n"
                " [self ]: {}\n [other]: {}".format(self.name, 
                self.adjacent, other.adjacent))
            return True
        return False
    
    def __str__(self):
        # 指定调用print函数时，程序终端将会输出的string
        return "class Vertex:\n\tname: {}\n\tposition: {}\n\tadjacent vertex: {}".\
            format(self.name, (self.x, self.y, self.z), [_.name for _ in self.adjacent])
    
    def add_adjacent(self, vertex):
        # 向顶点的邻近表中添加节点
        assert type(self) == type(vertex)
        if vertex not in self.adjacent:
            self.adjacent.append(vertex)
            return True
        return False

    @property
    def n_adjacent(self):
        return len(self.adjacent)


class Edge(object):
    def __init__(self, v1: Vertex, v2: Vertex):
        self.v1 = v1
        self.v2 = v2
        self.name: Set = {v1.name, v2.name}
        self.length_init = self.length

    def __eq__(self, other):
        if type(self) != type(other): return False
        return self.name == other.name

    def __str__(self):
        return "class Edge: \n\t name = {} \n\t v1.position = {} \n\t \
            v2.position = {} \n\t length = {}".\
        format(self.name, self.v1.position, self.v2.position, self.length)

    @property
    def length(self) -> LengthType:
        # the length of the edge
        return ((self.v1.x - self.v2.x) ** 2 \
              + (self.v1.y - self.v2.y) ** 2 \
              + (self.v1.z - self.v2.z) ** 2) ** 0.5

class Graph(object):
    def __init__(self, vertex_list: Optional[List[Vertex]]=[], 
                       edge_list  : Optional[List[Edge  ]]=[]):
        self.vertex_list: List[Vertex] = vertex_list
        self.edge_list  : List[Edge  ] = [] if edge_list == None else edge_list

    def update_edge(self):
        logging.info("Updating Edges...")
        from tqdm import tqdm
        for v in tqdm(self.vertex_list):
            for v_adjacent in v.adjacent:
                assert v in v_adjacent.adjacent
                edge = Edge(v, v_adjacent)
                if edge not in self.edge_list:
                    self.edge_list.append(edge)
    
    @property
    def n_edge(self):
        return len(self.edge_list)

    @property
    def n_vertex(self):
        return len(self.vertex_list)

from tqdm import tqdm
vertex_list: List[Vertex] = []
print("Getting Information...")
for idx in tqdm(range(len(vertex_frame))):
    # 获取主索节点的位置信息
    vx = vertex_frame["X"][idx]
    vy = vertex_frame["Y"][idx]
    vz = vertex_frame["Z"][idx]
    init_v_pos = np.array([vx, vy, vz], dtype=np.float64)
    # 获取主索节点促动器固定处位置的信息
    assert vertex_frame["Index"][idx] == ground_vertex_frame["Index"][idx]
    gvx = ground_vertex_frame["Xg"][idx]
    gvy = ground_vertex_frame["Yg"][idx]
    gvz = ground_vertex_frame["Zg"][idx]
    gv_pos = np.array([gvx, gvy, gvz], dtype=np.float64)
    vertex_info = VertexInfo(index = idx, status=-1, 
                             vertex_init_pos   =init_v_pos,
                             vertex_ground_pos =gv_pos)
    vertex_list.append(Vertex(name=vertex_frame["Index"][idx],
                              x=vx, y=vy, z=vz,
                              info=vertex_info))

print("len(vertex_list) = ", len(vertex_list)) # 2226

print("updating adjacent relationships...")
for face_idx in tqdm(range(len(face_vertex_frame))):
    # 读取三角反射面三个顶点的 name 参数
    v1_name = face_vertex_frame["Index1"][face_idx]
    v2_name = face_vertex_frame["Index2"][face_idx]
    v3_name = face_vertex_frame["Index3"][face_idx]
    # 在 vertex_list 之中根据 name参数 找到相应的顶点，
    v1, v2, v3 = None, None, None
    for v in vertex_list:
        if v.name == v1_name: v1 = v
        if v.name == v2_name: v2 = v
        if v.name == v3_name: v3 = v
        if v1 and v2 and v3: break
    
    # 更新近邻节点的信息
    assert v1 and v2 and v3
    v1.add_adjacent(v2)
    v1.add_adjacent(v3)
    v2.add_adjacent(v3)
    v2.add_adjacent(v1)
    v3.add_adjacent(v1)
    v3.add_adjacent(v2)

FASTnet = Graph(vertex_list = vertex_list)
FASTnet.update_edge()
print("n_edge   = ", FASTnet.n_edge)
print("n_vertex = ", FASTnet.n_vertex)


""" 节点工作区域的确定 """

def isworking(x:ScalorType,y:ScalorType,z:ScalorType):
    return (x ** 2 + y ** 2) < 150**2

vertex_list_working : List[Vertex] = []
vertex_list_fixing : List[Vertex]  = []
vertex_list_idling : List[Vertex]  = []
for i, vertex in enumerate(FASTnet.vertex_list):
    if isworking(*vertex.info.vertex_init_pos):
        vertex.info._status=0
        vertex_list_working.append(vertex)
    elif vertex.n_adjacent < 5:
        vertex.info._status=2
        vertex_list_fixing.append(vertex)
    else:
        vertex.info._status=1
        vertex_list_idling.append(vertex)

print(len(vertex_list_working)) # 中心工作区节点数 706
print(len(vertex_list_fixing))  # 边界固定区节点数 130
print(len(vertex_list_idling))  # 外围闲置区节点数 1390
# 测试
print(vertex_list_working[0].info.status)  # working
print(vertex_list_fixing[0].info.status)   # fixing
print(vertex_list_idling[0].info.status)   # idling


""" 能量函数的获得 """
from jax import lax

# 参数
A = 2.25 * 3.1415 * 1E-4  # 横截面积（米^2）
E = 1.8E4                 # 杨氏模量（Pa）
c1 = E * A

# 超参数
kappa = 45
offset_ratio = 6.5*10**-4

def edge_energy(length:LengthType, length_init:LengthType):
    delta_L = jnp.abs(length - length_init)
    delta_L0 = offset_ratio * length_init
    k = c1 / length_init
    true_fun  = lambda void: 0.5 * k * delta_L ** 2
    false_fun = lambda void: 0.5 * k * delta_L0 ** 2 + \
                k * delta_L0 * (lax.abs(delta_L) - delta_L0) * kappa
    return lax.cond(lax.abs(delta_L) < delta_L0, true_fun, false_fun, None)
# 通过vmap扩展
edges_energy = jax.vmap(edge_energy, in_axes = (0,0), out_axes=0, )

# 注意，因为不需要微分，这里所有的长度都是 numpy.array 格式
init_pos_array = np.array(vertex_frame.iloc[:,1:], dtype=np.float64)
init_v1_idx = [e.v1.info.idx for e in FASTnet.edge_list]
init_v2_idx = [e.v2.info.idx for e in FASTnet.edge_list]
init_v1_pos_array = init_pos_array[init_v1_idx,:]  # (nedge, 3)
init_v2_pos_array = init_pos_array[init_v2_idx,:]  # (nedge, 3)
init_length_array = np.sum((init_v1_pos_array - init_v2_pos_array)**2, axis=-1) ** 0.5  # (nedge,)

# 输出测试
print("len = ", len(init_length_array))  #  6525
print("max = ", max(init_length_array))  # 12.41
print("min = ", min(init_length_array))  # 10.39
print(init_length_array[:5])

# 初始条件下主索节点的位置，用jnp.ndarray格式储存
pos_array = jnp.array(vertex_frame.iloc[:,1:])
print(pos_array.shape) # (2226, 3)

# 无约束条件下能量函数的生成
def gen_net_energy(net: Graph, summing=True):
    adjacent_idx1 = [e.v1.info.idx for e in net.edge_list]
    adjacent_idx2 = [e.v2.info.idx for e in net.edge_list]
    
    def net_energy(pos_array: jnp.ndarray, init_length_array = init_length_array):
        # pos_array 的 shape 是 (nvertex, 3)
        v1_array = pos_array[adjacent_idx1,:]  # (nedge, 3)
        v2_array = pos_array[adjacent_idx2,:]  # (nedge, 3)
        length_array = jnp.sum((v1_array - v2_array)**2, axis=-1) ** 0.5  # (nedge,)：实际连接索们的长度
        ene = edges_energy(length_array, init_length_array)

        if summing:
            return jnp.sum(ene)
        return ene
    return net_energy

net_energy = gen_net_energy(FASTnet, summing=False)  # 无约束条件下能量的计算函数
ene_unrestricted = net_energy(pos_array)
print(ene_unrestricted)  # [0,0,0, ...]

# 之前已经得到的list:
#   工作区： vertex_list_working
#   过渡区： vertex_list_idling
#   固定区： vertex_list_fixing

# 获取不同区域主索节点们的index
working_vertex_index_list = [v.info.idx for v in vertex_list_working]
fixing_vertex_index_list  = [v.info.idx for v in vertex_list_fixing ]
idling_vertex_index_list  = [v.info.idx for v in vertex_list_idling ]

# 将这些index组合在一起
vertex_index_list: List[int] = []
vertex_index_list.extend(working_vertex_index_list)  # 先是 working
vertex_index_list.extend(idling_vertex_index_list)   # 再放 idling
vertex_index_list.extend(fixing_vertex_index_list)   # 最后 fixing
print(len(vertex_index_list)) # 2226
print(vertex_index_list[:5])

# 上述的 vertex_index_list 指明了所有的 vertex 在以 [working, idling, fixing]的顺序组
# 合后，相比于最初，以一种什么样的方式被打乱，相当于定义了一个置换操作p,如果我们想要让已
# 经被打乱的 vertex 位置数组重新恢复原本的顺序，就需要求出该置换操作的逆 p_inverse:

def inverse_index(p:List[int]):
    p = np.array(p)
    p_original = np.arange(len(p))
    operation = np.vstack([p_original, p]).T.tolist()
    operation_inv = sorted(operation, key=lambda x:x[1], reverse=False)
    p_inverse = [op[0] for op in operation_inv]
    return p_inverse

vertex_index_list_inverse = inverse_index(vertex_index_list)

# 生成约束条件下的函数
def calc_z(x, y, h=300.89097588, R=300.4):
    return (x**2+y**2) / (4*(h-R+0.466*R))-h

fixing_pos_array = jnp.asarray([[v.x, v.y, v.z] for v in vertex_list_fixing])

def gen_net_energy_restricted(energy_fcn: Callable, calc_z_working:Callable):
    def net_energy_restricted(params):
        working_xy_array, idling_pos_array = params

        # 我们先按照约束把工作区主索节点的坐标补全
        working_z_array = calc_z_working(working_xy_array[:,0], working_xy_array[:,1])
        working_z_array = working_z_array[:,jnp.newaxis]
        working_pos_array = jnp.hstack([working_xy_array, working_z_array])
        # 将 pos_array 分别 stack 在一起
        pos_array = jnp.vstack([working_pos_array, idling_pos_array, fixing_pos_array])
        # 将 pos_array 按照原本的顺序排列
        pos_array_ordered = pos_array[vertex_index_list_inverse, :]
        # 计算得到能量函数
        ene = energy_fcn(pos_array_ordered)
        return ene
    return net_energy_restricted

import optax
from jax import grad, jit

net_energy_fun_unrestricted = gen_net_energy(FASTnet, summing=True)
net_energy_fun = gen_net_energy_restricted(net_energy_fun_unrestricted, calc_z) # 有约束条件下能量的计算函数
net_energy_fun = jit(net_energy_fun)
dE = jit(grad(net_energy_fun))

working_xy_array = jnp.array([[v.x, v.y] for v in vertex_list_working])
idling_pos_array = jnp.array([[v.x, v.y, v.z] for v in vertex_list_idling])
params = (working_xy_array, idling_pos_array)

@jax.jit
def step(params, opt_state):
    value, grads = jax.value_and_grad(net_energy_fun)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value


step_num = 20000
ene_list = []

# lr=0.0001
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

print("ene_init = ", net_energy_fun(params))
for idx in range(step_num):
    params, opt_state, value = step(params, opt_state)        
    if (idx + 1) % 1000 == 0:
        print("[{}]: ene = {:.6f}".format(idx+1, float(value)))


# 更新节点的坐标
working_xy_array, idling_pos_array = params

print(len(working_xy_array))
for v_idx in range(len(working_xy_array)):    
    v = vertex_list_working[v_idx]
    v.x, v.y = working_xy_array[v_idx][0], working_xy_array[v_idx][1]
    v.z = calc_z(v.x, v.y)


print(len(idling_pos_array))
for v_idx in range(len(idling_pos_array)):
    v = vertex_list_idling[v_idx]
    v.x, v.y, v.z = idling_pos_array[v_idx]


# 能量测试
pos_list = []
for idx, v in enumerate(FASTnet.vertex_list):
    assert v.info.idx == idx
    pos_list.append([v.x, v.y, v.z])

pos_array = jnp.array(pos_list)
print(net_energy_fun_unrestricted(pos_array))

# 杆子长测试
c = 0
for idx, e in enumerate(FASTnet.edge_list):
    epsilon0 = 7E-4
    epsilon  = abs(e.length_init - e.length) / e.length_init
    if epsilon >=epsilon0:
        c+=1
print("-" * 100)
print("number = ", c)
print("-" * 100)

# 水平方向偏移输出
X, Y, Z = [], [], []
dX, dY, dZ = [], [], []
Color = []
R_list = []
for idx, v in enumerate(FASTnet.vertex_list):
    x, y, z = v.info.vertex_init_pos
    dx, dy, dz = v.x - x, v.y - y , v.z - z
    X.append(x)
    Y.append(y)
    Z.append(z)
    norm = np.sqrt(dx**2+ dy**2)
    if norm > 0.1:
        dx = dx / norm * 0.1
        dy = dy / norm * 0.1

    dX.append(dx)
    dY.append(dy)
    dZ.append(dz)
    R_list.append(np.sqrt(x**2+y**2))
    Color.append(dz)
print(max(dZ))
print(max(dX))
print(max(dY))

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 10))

r1 = 150
r2 = max(R_list)
theta = np.linspace(0, 2*np.pi, 1001)
r1x = r1 * np.cos(theta)
r1y = r1 * np.sin(theta)
r2x = r2 * np.cos(theta)
r2y = r2 * np.sin(theta)
plt.plot(r1x,r1y, linestyle = "-", label = "working regine")
plt.plot(r2x,r2y, linestyle = "-.", label = "total regine")
plt.legend(loc = "lower right")
plt.quiver(X, Y, dX, dY, scale=4)
plt.savefig("horizontal.png")
plt.close()


fig = plt.figure()  # 定义三维坐标轴
ax = plt.axes(projection = '3d')
ax.view_init(20, -120)
ax.scatter3D(X, Y, dZ, c=R_list/r2,s=2)
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
plt.xlim((-200, 200))
plt.ylim((-200, 200))
my_x_ticks = np.linspace(-200, 200, 5)
my_y_ticks = np.linspace(-200, 200, 5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.savefig("vertical.png")


