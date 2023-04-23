"""
代码示例6.12：
    pmap实例
    例子：
        规则30 细胞自动机
"""
import jax
device_count = jax.local_device_count()
print(device_count)
try:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()   
except KeyError as e:
    if str(e) == "'COLAB_TPU_ADDR'":
        if device_count != 8:
            raise EnvironmentError('我们建议您使用colab环境，并且使用8个TPU')


import jax.numpy as jnp
from jax import pmap, lax
from functools import partial


def send_right(x, axis_name):
    """ 将x发送到下个硬件，并返回上个硬件发送的数据"""
    left_perm = [(i, (i + 1) % device_count) for i in range(device_count)]
    return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

def send_left(x, axis_name):
    """ 将x发送到上个硬件，并返回下个硬件发送的数据"""
    left_perm = [((i + 1) % device_count, i) for i in range(device_count)]
    return lax.ppermute(x, perm=left_perm, axis_name=axis_name)

def update_board(board):
    """ 按照规则30更新棋盘"""
    left = board[:-2]
    right = board[2:]
    center = board[1:-1]
    return lax.bitwise_xor(left, lax.bitwise_or(center, right))  # rule30

@partial(pmap, axis_name='i')
def step(board_slice):
    """ 一次完整更新，包括数据的通讯，棋盘的扩展，格子的更新"""
    left, right = board_slice[:1], board_slice[-1:]  # 选中边界格点
    right, left = send_left(left, 'i'), send_right(right, 'i')  # 向其他硬件发送消息
    enlarged_board_slice = jnp.concatenate([left, board_slice, right])  # 合并其他硬件发送的消息
    return update_board(enlarged_board_slice)  # 根据其他硬件发送的消息更新

def print_board(nstep, board):
    """ 将棋盘以字符串的形式打印"""
    print(f'{nstep}'.rjust(3, ' '), ' | ', ''.join('*' if x else ' ' for x in board.ravel()))
    
def random_init_pattern(p=0.5, seed=42):
    """ 按照伯努利分布随机生成初始结构"""
    rng = jax.random.PRNGKey(seed)
    board = jax.random.bernoulli(rng, p, shape=(40, ))
    board = jnp.asarray(board, dtype=bool)
    return board.reshape((device_count, -1))

def simple_init_pattern():
    """ 生成简单初始结构"""
    board = jnp.zeros(40, dtype=bool)
    board = board.at[20].set(True).reshape(device_count, -1)
    return board

def evolution(nsteps, pattern='simple', *args, **kwargs):
    """ 代码入口：nsteps指定更新次数，pattern指定初始结构"""
    init_rule = {
        'simple': simple_init_pattern,
        'random_init_pattern': random_init_pattern,
    }
    board = init_rule[pattern](*args, **kwargs)

    print_board(0, board)
    for nstep in range(1, nsteps+1):
        board = step(board)
        print_board(nstep, board)
        
evolution(15)
