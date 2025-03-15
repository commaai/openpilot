# https://github.com/mlcommons/training/blob/e237206991d10449d9675d95606459a3cb6c21ad/image_classification/tensorflow2/lars_util.py
# changes: commented out logging
# changes: convert_to_tensor_v2 -> convert_to_tensor
# changes: extend from tf.python.keras.optimizer_v2.learning_rate_schedule.LearningRateScheduler

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Enable Layer-wise Adaptive Rate Scaling optimizer in ResNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

#from tf2_common.utils.mlp_log import mlp_log
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule

FLAGS = flags.FLAGS


def define_lars_flags():
  """Defines flags needed by LARS optimizer."""

  flags.DEFINE_float(
      'end_learning_rate', default=None,
      help=('Polynomial decay end learning rate.'))

  flags.DEFINE_float(
      'lars_epsilon', default=0.0,
      help=('Override autoselected LARS epsilon.'))

  flags.DEFINE_float(
      'warmup_epochs', default=None,
      help=('Override autoselected polynomial decay warmup epochs.'))

  flags.DEFINE_float(
      'momentum',
      default=0.9,
      help=('Momentum parameter used in the MomentumOptimizer.'))


class PolynomialDecayWithWarmup(learning_rate_schedule.LearningRateSchedule):
  """A LearningRateSchedule that uses a polynomial decay with warmup."""

  def __init__(
      self,
      batch_size,
      steps_per_epoch,
      train_steps,
      initial_learning_rate=None,
      end_learning_rate=None,
      warmup_epochs=None,
      compute_lr_on_cpu=False,
      name=None):
    """Applies a polynomial decay to the learning rate with warmup."""
    super(PolynomialDecayWithWarmup, self).__init__()

    self.batch_size = batch_size
    self.steps_per_epoch = steps_per_epoch
    self.train_steps = train_steps
    self.name = name
    self.learning_rate_ops_cache = {}
    self.compute_lr_on_cpu = compute_lr_on_cpu

    if batch_size < 16384:
      self.initial_learning_rate = 10.0
      warmup_epochs_ = 5
    elif batch_size < 32768:
      self.initial_learning_rate = 25.0
      warmup_epochs_ = 5
    else:
      self.initial_learning_rate = 31.2
      warmup_epochs_ = 25

    # Override default poly learning rate and warmup epochs
    if initial_learning_rate:
      self.initial_learning_rate = initial_learning_rate

    if end_learning_rate:
      self.end_learning_rate = end_learning_rate
    else:
      self.end_learning_rate = 0.0001

    if warmup_epochs is not None:
      warmup_epochs_ = warmup_epochs
    self.warmup_epochs = warmup_epochs_

    """
    opt_name = FLAGS.optimizer.lower()
    mlp_log.mlperf_print('opt_name', opt_name)
    if opt_name == 'lars':
      mlp_log.mlperf_print('{}_epsilon'.format(opt_name), FLAGS.lars_epsilon)
    mlp_log.mlperf_print('{}_opt_weight_decay'.format(opt_name),
                         FLAGS.weight_decay)
    mlp_log.mlperf_print('{}_opt_base_learning_rate'.format(opt_name),
                         self.initial_learning_rate)
    mlp_log.mlperf_print('{}_opt_learning_rate_warmup_epochs'.format(opt_name),
                         warmup_epochs_)
    mlp_log.mlperf_print('{}_opt_end_learning_rate'.format(opt_name),
                         self.end_learning_rate)
    """
    warmup_steps = warmup_epochs_ * steps_per_epoch
    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    self.decay_steps = train_steps - warmup_steps + 1
    """
    mlp_log.mlperf_print('{}_opt_learning_rate_decay_steps'.format(opt_name),
                         int(self.decay_steps))
    mlp_log.mlperf_print(
        '{}_opt_learning_rate_decay_poly_power'.format(opt_name), 2.0)
    mlp_log.mlperf_print('{}_opt_momentum'.format(opt_name), FLAGS.momentum)
    """

    self.poly_rate_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=self.initial_learning_rate,
        decay_steps=self.decay_steps,
        end_learning_rate=self.end_learning_rate,
        power=2.0)

  def __call__(self, step):
    if tf.executing_eagerly():
      return self._get_learning_rate(step)

    # In an eager function or graph, the current implementation of optimizer
    # repeatedly call and thus create ops for the learning rate schedule. To
    # avoid this, we cache the ops if not executing eagerly.
    graph = tf.compat.v1.get_default_graph()
    if graph not in self.learning_rate_ops_cache:
      if self.compute_lr_on_cpu:
        with tf.device('/device:CPU:0'):
          self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
      else:
        self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
    return self.learning_rate_ops_cache[graph]

  def _get_learning_rate(self, step):
    with ops.name_scope_v2(self.name or 'PolynomialDecayWithWarmup') as name:

      initial_learning_rate = ops.convert_to_tensor(
          self.initial_learning_rate, name='initial_learning_rate')
      warmup_steps = ops.convert_to_tensor(
          self.warmup_steps, name='warmup_steps')

      warmup_rate = (
          initial_learning_rate * step / warmup_steps)

      poly_steps = math_ops.subtract(step, warmup_steps)
      poly_rate = self.poly_rate_scheduler(poly_steps)

      decay_rate = tf.where(step <= warmup_steps,
                            warmup_rate, poly_rate, name=name)
      return decay_rate

  def get_config(self):
    return {
        'batch_size': self.batch_size,
        'steps_per_epoch': self.steps_per_epoch,
        'train_steps': self.train_steps,
        'initial_learning_rate': self.initial_learning_rate,
        'end_learning_rate': self.end_learning_rate,
        'warmup_epochs': self.warmup_epochs,
        'name': self.name,
    }
