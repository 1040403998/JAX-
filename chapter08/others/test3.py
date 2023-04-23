

import functools
import math
from typing import Tuple, TypeVar
import warnings

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd

T = TypeVar('T')
Pair = Tuple[T, T]

warnings.filterwarnings('ignore')

def sine_seq(
    phase: float,
    seq_len: int,
    samples_per_cycle: int,
) -> Pair[np.ndarray]:
  """Returns x, y in [T, B] tensor."""
  t = np.arange(seq_len + 1) * (2 * math.pi / samples_per_cycle)
  t = t.reshape([-1, 1]) + phase
  sine_t = np.sin(t)
  return sine_t[:-1, :], sine_t[1:, :]


def generate_data(
    seq_len: int,
    train_size: int,
    valid_size: int,
) -> Pair[Pair[np.ndarray]]:
  phases = np.random.uniform(0., 2 * math.pi, [train_size + valid_size])
  all_x, all_y = sine_seq(phases, seq_len, 3 * seq_len / 4)

  all_x = np.expand_dims(all_x, -1)
  all_y = np.expand_dims(all_y, -1)
  train_x = all_x[:, :train_size]
  train_y = all_y[:, :train_size]
  
  valid_x = all_x[:, train_size:]
  valid_y = all_y[:, train_size:]

  return (train_x, train_y), (valid_x, valid_y)


class Dataset:
  """An iterator over a numpy array, revealing batch_size elements at a time."""

  def __init__(self, xy: Pair[np.ndarray], batch_size: int):
    self._x, self._y = xy
    self._batch_size = batch_size
    self._length = self._x.shape[1]
    self._idx = 0
    if self._length % batch_size != 0:
      msg = 'dataset size {} must be divisible by batch_size {}.'
      raise ValueError(msg.format(self._length, batch_size))

  def __next__(self) -> Pair[np.ndarray]:
    start = self._idx
    end = start + self._batch_size
    x, y = self._x[:, start:end], self._y[:, start:end]
    if end >= self._length:
      end = end % self._length
      assert end == 0  # Guaranteed by ctor assertion.
    self._idx = end
    return x, y


TRAIN_SIZE = 2 ** 14
VALID_SIZE = 128
BATCH_SIZE = 8
SEQ_LEN = 64

train, valid = generate_data(SEQ_LEN, TRAIN_SIZE, VALID_SIZE)
print("train_x.shape = ", train[0].shape)  # (64, 16384, 1)
print("train_y.shape = ", train[1].shape)  # (64, 16384, 1)
print("valid_x.shape = ", valid[0].shape)  # (64, 128, 1)
print("valid_y.shape = ", valid[1].shape)  # (64, 128, 1)

# Plot an observation/target pair.
df = pd.DataFrame({'x': train[0][:, 0, 0], 'y': train[1][:, 0, 0]}).reset_index()
df = pd.melt(df, id_vars=['index'], value_vars=['x', 'y'])

train_ds = Dataset(train, BATCH_SIZE)
valid_ds = Dataset(valid, BATCH_SIZE)

x, y = next(valid_ds)
print("x.shape = ", x.shape)  # (64, 8, 1)
print("y.shape = ", y.shape)  # (64, 8, 1)
del train, valid  # Don't leak temporaries.


def unroll_net(seqs: jnp.ndarray):
  """Unrolls an LSTM over seqs, mapping each output to a scalar."""
  # seqs is [T, B, F].
  core = hk.LSTM(32)
  batch_size = seqs.shape[1]
  outs, state = hk.dynamic_unroll(core, seqs, core.initial_state(batch_size))
  # We could include this Linear as part of the recurrent core!
  # However, it's more efficient on modern accelerators to run the linear once
  # over the entire sequence than once per sequence element.
  return hk.BatchApply(hk.Linear(1))(outs), state

model = hk.transform(unroll_net)


def train_model(train_ds: Dataset, valid_ds: Dataset) -> hk.Params:
  """Initializes and trains a model on train_ds, returning the final params."""
  rng = jax.random.PRNGKey(428)
  opt = optax.adam(1e-3)

  @jax.jit
  def loss(params, x, y):
    pred, _ = model.apply(params, None, x)
    return jnp.mean(jnp.square(pred - y))

  @jax.jit
  def update(step, params, opt_state, x, y):
    l, grads = jax.value_and_grad(loss)(params, x, y)
    grads, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return l, params, opt_state

  # Initialize state.
  sample_x, _ = next(train_ds)
  params = model.init(rng, sample_x)
  print("sample_x.shape", sample_x.shape)
  
  opt_state = opt.init(params)

  for step in range(2001):
    if step % 100 == 0:
      x, y = next(valid_ds)
      print("Step {}: valid loss {}".format(step, loss(params, x, y)))

    x, y = next(train_ds)
    train_loss, params, opt_state = update(step, params, opt_state, x, y)
    if step % 100 == 0:
      print("Step {}: train loss {}".format(step, train_loss))

  return params

trained_params = train_model(train_ds, valid_ds)
