"""
代码示例6.4.2：
    pmap实例
    例子：
        康威生命游戏
"""

import jax
import jax.tools.colab_tpu
import numpy as np
import jax.numpy as jnp
from jax import pmap, lax
from functools import partial
from time import time

# 配置colab绘图环境
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib import rc
rc('animation', html='html5')
# %matplotlib inline

# 初始化硬件
jax.tools.colab_tpu.setup_tpu()
ndevices = jax.local_device_count()
print(f'{ndevices} devices available')

def init_board_with_pattern(scale, pattern='glider'):
    """ 初始化棋盘，并绘制初始图样""" 
    board = np.zeros((ndevices*scale, ndevices*scale)).astype(int)

    if pattern == 'glider':
        board[3, 3] = 1
        board[3, 5] = 1
        board[4, 4:6] = 1
        board[5, 4] = 1 
    elif pattern == 'blinker':
        board[3:6, 3] = 1
    elif pattern == 'block':
        board[3:5, 3:5] = 1

    return board.reshape(ndevices, scale, -1)

def send_positive(x, axis_name):
    """ 向下个硬件发送数据，并将上个硬件发送的数据返回"""
    left_perm = [(i, (i + 1) % ndevices) for i in range(ndevices)]
    return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

def send_negtive(x, axis_name):
    """ 向上个硬件发送数据，并将下个硬件发送的数据返回"""
    right_perm = [((i + 1) % ndevices, i) for i in range(ndevices)]
    return lax.ppermute(x, perm=right_perm, axis_name=axis_name)

def update_pattern(enlarged_board, board):
    """ 更新棋盘"""
    return lax.reduce_window(enlarged_board, 0, lax.add, (3,3), (1,1), 'VALID') - board


@partial(pmap, axis_name='row')  # 对二维棋盘进行水平分割，分给每个硬件
def update(board_slice):
    """ 一次完整的更新，包括数据通讯，更新棋盘"""
    left, right = board_slice[:, :1], board_slice[:, -1:]  # 选中边界格点
    horizon_enlarged_board_slice = jnp.concatenate([right, board_slice, left], axis=1)  # 水平方向扩展棋盘

    up, down = horizon_enlarged_board_slice[:1], horizon_enlarged_board_slice[-1:]  # 选中垂直方向的边界格点
    down, up = send_negtive(up, 'row'), send_positive(down, 'row')  # 消息通讯
    padding_board_slice = jnp.concatenate([up, horizon_enlarged_board_slice, down])

    count = update_pattern(padding_board_slice, board_slice)

    # 当前细胞存活，周围存活细胞少于两个，该细胞下一时刻死亡
    board_slice = jnp.where(count<2, 0, board_slice)
    # 当前细胞存活，周围有两个或三个细胞，该细胞下一时刻依然存活
    board_slice  # no need to implement
    # 当前细胞存活，周围有超过三个细胞存活，该细胞下一时刻死亡
    board_slice = jnp.where(count>3, 0, board_slice)
    # 当前细胞死亡，周围有三个存活细胞，该细胞下一时刻变成存活
    board_slice = jnp.where(jnp.logical_and(board_slice==0, count==3), 1, board_slice)

    return board_slice

def evolution(epochs, scale=8):
    
    board = init_board_with_pattern(scale, pattern='glider')
    print(f'board.shape:{board.shape}')

    snapshot = []
    fig, ax = plt.subplots()
    plt.close()

    for i in range(epochs):

        board = update(board)

        im = ax.imshow(board.reshape((ndevices*scale, ndevices*scale)))
        snapshot.append([im])

    ani = ArtistAnimation(fig, snapshot, interval=100, blit=True)

    return ani

def test_evolution():
    costs = []
    for j in range(4, 15):
        scale = 2**j
        board = init_board_with_pattern(scale, pattern='glider')
        print(f'board.shape:{board.shape}')
        start = time()
        for i in range(10):

            board = update(board)
        end = time()
        costs.append(end-start)
    return costs

evolution(10)
