# https://github.com/mlcommons/training/blob/e3769c8dcf88cd21e1001dd2f894b40a1513ec5d/image_classification/tensorflow2/lars_optimizer.py
# changes: don't call lr_t if it's not a schedule

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
"""Layer-wise Adaptive Rate Scaling optimizer for large-batch training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from tf2_common.training import optimizer_v2modified
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training_ops
from tensorflow.python.ops import state_ops


# class LARSOptimizer(optimizer_v2modified.OptimizerV2Modified):
class LARSOptimizer(optimizer_v2.OptimizerV2):
  """Layer-wise Adaptive Rate Scaling for large batch training.

  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)

  Implements the LARS learning rate scheme presented in the paper above. This
  optimizer is useful when scaling the batch size to up to 32K without
  significant performance degradation. It is recommended to use the optimizer
  in conjunction with:
      - Gradual learning rate warm-up
      - Linear learning rate scaling
      - Poly rule learning rate decay

  Note, LARS scaling is currently only enabled for dense tensors. Sparse tensors
  use the default momentum optimizer.
  """

  def __init__(
      self,
      learning_rate,
      momentum=0.9,
      weight_decay=0.0001,
      # The LARS coefficient is a hyperparameter
      eeta=0.001,
      epsilon=0.0,
      name="LARSOptimizer",
      # Enable skipping variables from LARS scaling.
      # TODO(sameerkm): Enable a direct mechanism to pass a
      # subset of variables to the optimizer.
      skip_list=None,
      use_nesterov=False,
      **kwargs):
    """Construct a new LARS Optimizer.

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate.
      momentum: A floating point value. Momentum hyperparameter.
      weight_decay: A floating point value. Weight decay hyperparameter.
      eeta: LARS coefficient as used in the paper. Dfault set to LARS
        coefficient from the paper. (eeta / weight_decay) determines the highest
        scaling factor in LARS.
      epsilon: Optional epsilon parameter to be set in models that have very
        small gradients. Default set to 0.0.
      name: Optional name prefix for variables and ops created by LARSOptimizer.
      skip_list: List of strings to enable skipping variables from LARS scaling.
        If any of the strings in skip_list is a subset of var.name, variable
        'var' is skipped from LARS scaling. For a typical classification model
        with batch normalization, the skip_list is ['batch_normalization',
        'bias']
      use_nesterov: when set to True, nesterov momentum will be enabled
      **kwargs: keyword arguments.

    Raises:
      ValueError: If a hyperparameter is set to a non-sensical value.
    """
    if momentum < 0.0:
      raise ValueError("momentum should be positive: %s" % momentum)
    if weight_decay < 0.0:
      raise ValueError("weight_decay should be positive: %s" % weight_decay)
    super(LARSOptimizer, self).__init__(name=name, **kwargs)

    self._set_hyper("learning_rate", learning_rate)

    # When directly using class members, instead of
    # _set_hyper and _get_hyper (such as learning_rate above),
    # the values are fixed after __init(), and not being
    # updated during the training process.
    # This provides better performance but less flexibility.
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.eeta = eeta
    self.epsilon = epsilon or backend_config.epsilon()
    self._skip_list = skip_list
    self.use_nesterov = use_nesterov

  def _prepare_local(self, var_device, var_dtype, apply_state):
    lr_t = self._get_hyper("learning_rate", var_dtype)
    local_step = math_ops.cast(self.iterations, var_dtype)
    if callable(lr_t): lr_t = math_ops.cast(lr_t(local_step), var_dtype)
    learning_rate_t = array_ops.identity(lr_t)

    apply_state[(var_device, var_dtype)].update(
        dict(
            learning_rate=learning_rate_t,
            ))

  def _create_slots(self, var_list):
    for v in var_list:
      self.add_slot(v, "momentum")

  def compute_lr(self, grad, var, coefficients):
    scaled_lr = coefficients["learning_rate"]
    if self._skip_list is None or not any(v in var.name
                                          for v in self._skip_list):
      w_norm = linalg_ops.norm(var, ord=2)
      g_norm = linalg_ops.norm(grad, ord=2)
      trust_ratio = array_ops.where(
          math_ops.greater(w_norm, 0),
          array_ops.where(
              math_ops.greater(g_norm, 0),
              (self.eeta * w_norm /
               (g_norm + self.weight_decay * w_norm + self.epsilon)), 1.0), 1.0)

      scaled_lr = coefficients["learning_rate"] * trust_ratio
      # Add the weight regularization gradient
      grad = grad + self.weight_decay * var
    return scaled_lr, grad

  def _apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    scaled_lr, grad = self.compute_lr(grad, var, coefficients)
    mom = self.get_slot(var, "momentum")
    return training_ops.apply_momentum(
        var,
        mom,
        math_ops.cast(1.0, var.dtype.base_dtype),
        grad * scaled_lr,
        self.momentum,
        use_locking=False,
        use_nesterov=self.use_nesterov)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    scaled_lr, grad = self.compute_lr(grad, var, coefficients)
    mom = self.get_slot(var, "momentum")
    # Use ApplyKerasMomentum instead of ApplyMomentum
    # training_ops.resource_apply_keras_momentum(
    #     var.handle,
    #     mom.handle,
    #     scaled_lr,
    #     grad,
    #     coefficients["momentum"],
    #     use_locking=False,
    #     use_nesterov=self.use_nesterov)

    mom_t = mom * self.momentum - grad * scaled_lr
    mom_t = state_ops.assign(mom, mom_t, use_locking=False)
    if self.use_nesterov:
      var_t = var + mom_t * self.momentum - grad * scaled_lr
    else:
      var_t = var + mom_t
    return state_ops.assign(var, var_t, use_locking=False).op

  # Fallback to momentum optimizer for sparse tensors
  def _apply_sparse(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    mom = self.get_slot(var, "momentum")
    return training_ops.sparse_apply_momentum(
        var,
        mom,
        coefficients["learning_rate"],
        grad.values,
        grad.indices,
        self.momentum,
        use_locking=False,
        use_nesterov=self.use_nesterov)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    mom = self.get_slot(var, "momentum")
    return training_ops.resource_sparse_apply_keras_momentum(
        var.handle,
        mom.handle,
        coefficients["learning_rate"],
        grad,
        indices,
        self.momentum,
        use_locking=False,
        use_nesterov=self.use_nesterov)

  def get_config(self):
    config = super(LARSOptimizer, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "momentum": self.momentum,
        "weight_decay": self.weight_decay,
        "eeta": self.eeta,
        "epsilon": self.epsilon,
        "use_nesterov": self.use_nesterov,
    })
    return config
