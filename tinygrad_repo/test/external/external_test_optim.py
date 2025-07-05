#!/usr/bin/env python
import unittest, math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import math_ops
from extra.lr_scheduler import LRSchedulerGroup

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import LAMB, LARS, SGD, OptimizerGroup, AdamW

from test.external.mlperf_resnet.lars_optimizer import LARSOptimizer

from examples.mlperf.lr_schedulers import PolynomialDecayWithWarmup, CosineAnnealingLRWithWarmup
from test.external.mlperf_resnet.lars_util import PolynomialDecayWithWarmup as PolynomialDecayWithWarmup_tf

np.random.seed(1337)
x_init = np.random.randn(1,4).astype(np.float32)
W_init = np.random.randn(4,4).astype(np.float32)
m_init = np.random.randn(1,4).astype(np.float32)

class TinyNet:
  def __init__(self):
    self.x = Tensor(x_init.copy(), requires_grad=True)
    self.W = Tensor(W_init.copy(), requires_grad=True)
    self.m = Tensor(m_init.copy())

  def forward(self):
    out = self.x.matmul(self.W).relu()
    out = out.log_softmax(1)
    out = out.mul(self.m).add(self.m).sum()
    return out

class TinyNetTF:
  def __init__(self):
    self.x = tf.Variable(x_init.copy(), trainable=True, name="x")
    self.W = tf.Variable(W_init.copy(), trainable=True, name="W")
    self.m = tf.constant(m_init.copy())

  def forward(self):
    out = tf.matmul(self.x, self.W)
    out = tf.nn.relu(out)
    out = tf.nn.log_softmax(out, axis=1)
    out = tf.multiply(out, self.m) + self.m
    out = tf.reduce_sum(out)
    return out

def step(optim, steps=1, kwargs={}, scheduler=None, schedopts=None, do_optim=True):
  net = TinyNet()
  optim = optim([net.x, net.W], **kwargs)
  if scheduler is not None: scheduler = scheduler(optim, **schedopts)
  lrs = []
  for _ in range(steps):
    if do_optim:
      out = net.forward()
      optim.zero_grad()
      out.backward()
    lrs.append(optim.lr.item() if not isinstance(optim, OptimizerGroup) else optim.optimizers[0].lr.item())
    if do_optim: optim.step()
    if scheduler is not None: scheduler.step()
  return lrs, net.x.detach().numpy(), net.W.detach().numpy()

def step_tf(optim, steps=1, kwargs={}, scheduler=None, schedopts=None, do_optim=True):
  net = TinyNetTF()
  if scheduler is not None: kwargs['lr'] = scheduler(**schedopts)
  optim = optim(**kwargs)
  lrs = []
  for _ in range(steps):
    if do_optim:
      with tf.GradientTape() as tape:
        out = net.forward()

    lr_t = optim.learning_rate
    # refer to test/external/mlperf_resnet/lars_optimizer.py:_prepare_local
    if callable(lr_t): lr_t = lr_t(math_ops.cast(optim.iterations, tf.float32))
    lrs.append(lr_t)

    if do_optim:
      grads = tape.gradient(out, [net.x, net.W])
      optim.apply_gradients(zip(grads, [net.x, net.W]))
      # optim calls scheduler in tf
    else:
      optim._iterations.assign_add(1)
  return lrs, net.x.numpy(), net.W.numpy()

# skip list is skipping W
def create_tiny_lars(params, lr, skip_list=False):
  if skip_list: return OptimizerGroup(LARS([params[0]], lr), SGD([params[1]], lr, classic=True, weight_decay=0., momentum=.9))
  return LARS(params, lr)
def create_tf_lars(lr, skip_list=False): return LARSOptimizer(lr, skip_list=["W"] if skip_list else None)

def create_tiny_polylr(optim, initial_lr, end_lr, train_steps, warmup, power=2, skip_list=False):
  assert power == 2
  if skip_list: return LRSchedulerGroup(
    PolynomialDecayWithWarmup(optim[0], initial_lr, end_lr, train_steps, warmup, power),
    PolynomialDecayWithWarmup(optim[1], initial_lr, end_lr, train_steps, warmup, power))
  return PolynomialDecayWithWarmup(optim, initial_lr, end_lr, train_steps, warmup, power)
def create_tf_polylr(initial_lr, end_lr, train_steps, warmup, power=2, skip_list=False):
  assert power == 2
  return PolynomialDecayWithWarmup_tf(1, 1, train_steps,
                                      initial_learning_rate=initial_lr, end_learning_rate=end_lr, warmup_epochs=warmup)

class ExternalTestOptim(unittest.TestCase):
  def setUp(self):
    self.old_training = Tensor.training
    Tensor.training = True
  def tearDown(self):
    Tensor.training = self.old_training

  def _test_optim(self, tinygrad_optim, tensorflow_optim, steps, opts, atol, rtol, tiny_sched=None, tf_sched=None, schedopts=None, do_optim=True):
    for x,y in zip(step(tinygrad_optim, steps=steps, kwargs=opts, scheduler=tiny_sched, schedopts=schedopts, do_optim=do_optim),
                   step_tf(tensorflow_optim, steps=steps, kwargs=opts, scheduler=tf_sched, schedopts=schedopts, do_optim=do_optim)):
      np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

  def _test_lamb(self, steps, opts, atol, rtol): self._test_optim(LAMB, tfa.optimizers.LAMB, steps, opts, atol, rtol)
  def _test_lars(self, steps, opts, atol, rtol): self._test_optim(create_tiny_lars, create_tf_lars, steps, opts, atol, rtol)
  def _test_lars_polylr(self, steps, opts, schedopts, atol, rtol, do_optim=True):
    self._test_optim(create_tiny_lars, create_tf_lars, steps, opts, atol, rtol,
                     tiny_sched=create_tiny_polylr, tf_sched=create_tf_polylr, schedopts=schedopts, do_optim=do_optim)

  def test_lamb(self): self._test_lamb(1, {'lr': 0.001}, 1e-5, 0)
  def test_lamb_high_lr(self): self._test_lamb(1, {'lr': 10}, 1e-5, 1e-5)

  def test_multistep_lamb(self): self._test_lamb(10, {'lr': 0.001}, 1e-5, 0)
  def test_multistep_lamb_high_lr(self): self._test_lamb(10, {'lr': 10}, 1e-5, 3e-4)

  def test_lars(self): self._test_lars(1, {'lr': 0.01}, 1e-5, 0)
  def test_lars_high_lr(self): self._test_lars(1, {'lr': 10}, 1e-5, 1e-5)
  def test_multistep_lars(self): self._test_lars(10, {'lr': 0.001}, 1e-5, 0)
  def test_multistep_lars_high_lr(self): self._test_lars(10, {'lr': 10}, 1e-5, 3e-4)
  def test_lars_skip(self): self._test_lars(10, {'lr': 10, 'skip_list': True}, 1e-5, 3e-4)
  def test_lars_skip_high_lr(self): self._test_lars(1, {'lr': 10, 'skip_list': True}, 1e-5, 1e-5)
  def test_lars_skip_multistep(self): self._test_lars(10, {'lr': 0.001, 'skip_list': True}, 1e-5, 0)
  def test_lars_skip_multistep_high_lr(self): self._test_lars(10, {'lr': 10, 'skip_list': True}, 1e-5, 3e-4)

  def test_lars_polylr(self):
    self._test_lars_polylr(10, {'lr': 1.0}, {
      'initial_lr': 1.0,
      'end_lr': 1e-4,
      'train_steps': 10,
      'warmup': 3
    }, 1e-5, 1e-5)
  def test_lars_polylr_large(self):
    self._test_lars_polylr(100, {'lr': 10.0}, {
      'initial_lr': 10.0,
      'end_lr': 1e-5,
      'train_steps': 100,
      'warmup': 43
    }, 1e-5, 1e-5, do_optim=False)
  def test_lars_polylr_skip(self):
    self._test_lars_polylr(10, {'lr': 1.0, 'skip_list': True}, {
      'initial_lr': 1.0,
      'end_lr': 1e-4,
      'train_steps': 10,
      'warmup': 3,
      'skip_list': True
    }, 1e-5, 1e-5)

  @unittest.skip("slow, but you can run this locally to check")
  def test_lars_polylr_resnet(self):
    train_files = 1_281_167
    BS = 624
    steps_per_epoch = train_files // BS
    epochs = 45
    warmup_epochs = 5
    self._test_lars_polylr(steps_per_epoch * epochs, {'lr': 10.4}, {
      'initial_lr': 10.4,
      'end_lr': 1e-4,
      # step counts for BS=624 EPOCHS=45 resnet
      'train_steps': steps_per_epoch * epochs,
      'warmup': steps_per_epoch * warmup_epochs,
    }, 1e-5, 1e-5, do_optim=False)


class TestCosineAnnealingLRWithWarmup(unittest.TestCase):
  # only tests the lr
  def _test_lr(self, base_lr, end_lr, warmup_steps, decay_steps):
    net = TinyNet()
    optim = AdamW([net.W], lr=0.0)
    tiny_lr = CosineAnnealingLRWithWarmup(optim, base_lr, end_lr, warmup_steps, decay_steps)
    lr = []
    for _ in range(warmup_steps+decay_steps):
      lr.append(optim.lr.item())
      tiny_lr.step()
    # reimplemented in python
    expected = []
    for i in range(warmup_steps): expected.append((i+1)/warmup_steps*base_lr)
    for i in range(decay_steps): expected.append(end_lr+(base_lr-end_lr)*(1+math.cos((i+1)/decay_steps*math.pi))/2)
    np.testing.assert_allclose(lr, expected, rtol=1e-5)

  def test_lr_0(self): self._test_lr(3e-4, 8e-5, 3, 5)
  def test_lr_1(self): self._test_lr(3e-4, 8e-5, 10, 20)
  def test_lr_llama3(self): self._test_lr(8e-5, 8e-7, 20, 100)

if __name__ == '__main__':
  unittest.main()
