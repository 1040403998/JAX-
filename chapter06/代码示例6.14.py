"""
代码示例6.14：
    pmap实例
    例子：
        单机版康威生命游戏
"""

import numpy as np
import jax.numpy as jnp
from jax import lax
from time import time

# init board
def init_board(scale):
  board = np.zeros((scale, scale))
  return board.astype(int)

def print_board(board):
    flatten_repr = ['*' if x else ' ' for i,x in enumerate(board.flatten())]
    for i in range(board.shape[0]):
        flatten_repr.insert(i*(board.shape[1]+1), '\n')
    print(''.join(flatten_repr))

def update(board_slice):
  
    left, right = board_slice[:, :1], board_slice[:, -1:]
    # right, left = send_negtive(left, 'col'), send_positive(right, 'col')
    horizon_enlarged_board_slice = jnp.concatenate([right, board_slice, left], axis=1)

    up, down = horizon_enlarged_board_slice[:1], horizon_enlarged_board_slice[-1:]
    padding_board_slice = jnp.concatenate([up, horizon_enlarged_board_slice, down])

    count = update_pattern(padding_board_slice, board_slice)

    # Any live cell with fewer than two live neighbours dies, as if by underpopulation.
    board_slice = jnp.where(count<2, 0, board_slice)
    # Any live cell with two or three live neighbours lives on to the next generation.
    board_slice  # no need to implement
    # Any live cell with more than three live neighbours dies, as if by overpopulation.
    board_slice = jnp.where(count>3, 0, board_slice)
    # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    board_slice = jnp.where(jnp.logical_and(board_slice==0, count==3), 1, board_slice)

    return board_slice

def update_pattern(enlarged_board, board):
    return lax.reduce_window(enlarged_board, 0, lax.add, (3,3), (1,1), 'VALID') - board

def init_board_with_pattern(scale, pattern='glider'):

    board = init_board(scale)

    if pattern == 'glider':
        board[3, 3] = 1
        board[3, 5] = 1
        board[4, 4:6] = 1
        board[5, 4] = 1 
    elif pattern == 'blinker':
        board[3:6, 3] = 1
    elif pattern == 'block':
        board[3:5, 3:5] = 1

    return board.reshape(scale, -1)

def evolution(epochs, scale=8):
    
    board = init_board_with_pattern(scale, pattern='glider')
    print(f'board.shape:{board.shape}')
    for i in range(epochs):
        board = update(board)



def test_evolution(order):
    costs = []
    for j in range(4,15):
        scale = 2**order
        start = time()
        evolution(10, scale)
        end = time()
    costs.append(end-start)
    return costs
