import numpy as np
from tinygrad.tensor import Tensor, _to_np_dtype

def mask_like(like, mask_inx, mask_value = 1.0):
  mask = np.zeros(like.shape, dtype=_to_np_dtype(like.dtype)).reshape(-1)
  mask[mask_inx] = mask_value
  return mask.reshape(like.shape)

def jacobian(func, input):
  output = func(input)

  ji = input.numpy().reshape(-1).shape[-1]
  jo = output.numpy().reshape(-1).shape[-1]
  J = np.zeros((jo,ji), dtype=np.float32)

  for o in range(jo):
    input.grad = None
    output = func(input)

    # tinygrad doesn't support slicing, tiny-hack to select
    # the needed scalar an backpropagate only through it
    o_scalar = Tensor(mask_like(output, o, 1.)).mul(output).sum()
    o_scalar = Tensor(mask_like(output, o, 1.)).mul(output).sum()
    o_scalar.backward()

    for i, grad in enumerate(input.grad.numpy().reshape(-1)):
      J[o,i] = grad
  return J

def numerical_jacobian(func, input, eps = 1e-3):
  output = func(input)

  ji = input.numpy().reshape(-1).shape[-1]
  jo = output.numpy().reshape(-1).shape[-1]
  NJ = np.zeros((jo, ji), dtype=np.float32)

  for i in range(ji):
    eps_perturb = mask_like(input, i, mask_value = eps)

    output_perturb_add = func(Tensor(input.numpy() + eps_perturb)).numpy().reshape(-1)
    output_perturb_sub = func(Tensor(input.numpy() - eps_perturb)).numpy().reshape(-1)

    grad_approx = ((output_perturb_add) - (output_perturb_sub)) / (2*eps)

    NJ[:,i] = grad_approx
  return NJ

def gradcheck(func, input, eps = 1e-3, atol = 1e-3, rtol = 1e-3):
  NJ = numerical_jacobian(func, input, eps)
  J = jacobian(func, input)
  return np.allclose(J, NJ, atol = atol, rtol = rtol)
