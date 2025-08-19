from tinygrad import Tensor
from tinygrad.nn import Conv2d, BatchNorm2d, Linear
from tinygrad.nn.state import load_state_dict, torch_load
from tinygrad.helpers import fetch

from typing import Optional, Dict
import numpy as np
from scipy import linalg

# Base Inception Model

class BasicConv2d:
  def __init__(self, in_ch:int, out_ch:int, **kwargs):
    self.conv = Conv2d(in_ch, out_ch, bias=False, **kwargs)
    self.bn   = BatchNorm2d(out_ch, eps=0.001)

  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential([self.conv, self.bn, Tensor.relu])

class InceptionA:
  def __init__(self, in_ch:int, pool_feat:int):
    self.branch1x1 = BasicConv2d(in_ch, 64, kernel_size=1)

    self.branch5x5_1 = BasicConv2d(in_ch, 48, kernel_size=1)
    self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

    self.branch3x3dbl_1 = BasicConv2d(in_ch, 64, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=(3,3), padding=1)
    self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=(3,3), padding=1)

    self.branch_pool = BasicConv2d(in_ch, pool_feat, kernel_size=1)

  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      self.branch1x1(x),
      x.sequential([self.branch5x5_1, self.branch5x5_2]),
      x.sequential([self.branch3x3dbl_1, self.branch3x3dbl_2, self.branch3x3dbl_3]),
      self.branch_pool(x.avg_pool2d(kernel_size=(3,3), stride=1, padding=1)),
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionB:
  def __init__(self, in_ch:int):
    self.branch3x3 = BasicConv2d(in_ch, 384, kernel_size=(3,3), stride=2)

    self.branch3x3dbl_1 = BasicConv2d(in_ch, 64, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=(3,3), padding=1)
    self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=(3,3), stride=2)

  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      self.branch3x3(x),
      x.sequential([self.branch3x3dbl_1, self.branch3x3dbl_2, self.branch3x3dbl_3]),
      x.max_pool2d(kernel_size=(3,3), stride=2, dilation=1),
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionC:
  def __init__(self, in_ch, ch_7x7):
    self.branch1x1 = BasicConv2d(in_ch, 192, kernel_size=1)

    self.branch7x7_1 = BasicConv2d(in_ch, ch_7x7, kernel_size=1)
    self.branch7x7_2 = BasicConv2d(ch_7x7, ch_7x7, kernel_size=(1, 7), padding=(0, 3))
    self.branch7x7_3 = BasicConv2d(ch_7x7, 192, kernel_size=(7, 1), padding=(3, 0))

    self.branch7x7dbl_1 = BasicConv2d(in_ch, ch_7x7, kernel_size=1)
    self.branch7x7dbl_2 = BasicConv2d(ch_7x7, ch_7x7, kernel_size=(7, 1), padding=(3, 0))
    self.branch7x7dbl_3 = BasicConv2d(ch_7x7, ch_7x7, kernel_size=(1, 7), padding=(0, 3))
    self.branch7x7dbl_4 = BasicConv2d(ch_7x7, ch_7x7, kernel_size=(7, 1), padding=(3, 0))
    self.branch7x7dbl_5 = BasicConv2d(ch_7x7, 192, kernel_size=(1, 7), padding=(0, 3))

    self.branch_pool = BasicConv2d(in_ch, 192, kernel_size=1)

  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      self.branch1x1(x),
      x.sequential([self.branch7x7_1, self.branch7x7_2, self.branch7x7_3]),
      x.sequential([self.branch7x7dbl_1, self.branch7x7dbl_2, self.branch7x7dbl_3, self.branch7x7dbl_4, self.branch7x7dbl_5]),
      self.branch_pool(x.avg_pool2d(kernel_size=(3,3), stride=1, padding=1)),
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionD:
  def __init__(self, in_ch:int):
    self.branch3x3_1 = BasicConv2d(in_ch, 192, kernel_size=1)
    self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=(3,3), stride=2)

    self.branch7x7x3_1 = BasicConv2d(in_ch, 192, kernel_size=1)
    self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
    self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
    self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=(3,3), stride=2)

  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      x.sequential([self.branch3x3_1, self.branch3x3_2]),
      x.sequential([self.branch7x7x3_1, self.branch7x7x3_2, self.branch7x7x3_3, self.branch7x7x3_4]),
      x.max_pool2d(kernel_size=(3,3), stride=2, dilation=1),
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionE:
  def __init__(self, in_ch:int):
    self.branch1x1 = BasicConv2d(in_ch, 320, kernel_size=1)

    self.branch3x3_1  = BasicConv2d(in_ch, 384, kernel_size=1)
    self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
    self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

    self.branch3x3dbl_1 = BasicConv2d(in_ch, 448, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=(3,3), padding=1)
    self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
    self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

    self.branch_pool = BasicConv2d(in_ch, 192, kernel_size=1)

  def __call__(self, x:Tensor) -> Tensor:
    branch3x3 = self.branch3x3_1(x)
    branch3x3dbl = x.sequential([self.branch3x3dbl_1, self.branch3x3dbl_2])
    outputs = [
      self.branch1x1(x),
      Tensor.cat(self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3), dim=1),
      Tensor.cat(self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl), dim=1),
      self.branch_pool(x.avg_pool2d(kernel_size=(3,3), stride=1, padding=1)),
    ]
    return Tensor.cat(*outputs, dim=1)

class InceptionAux:
  def __init__(self, in_ch:int, num_classes:int):
    self.conv0 = BasicConv2d(in_ch, 128, kernel_size=1)
    self.conv1 = BasicConv2d(128, 768, kernel_size=5)
    self.fc = Linear(768, num_classes)

  def __call__(self, x:Tensor) -> Tensor:
    x = x.avg_pool2d(kernel_size=5, stride=3, padding=1).sequential([self.conv0, self.conv1])
    x = x.avg_pool2d(kernel_size=1, padding=1).reshape(x.shape[0],-1)
    return self.fc(x)

class Inception3:
  def __init__(self, num_classes:int=1008, cls_map:Optional[Dict]=None):
    def get_cls(key1:str, key2:str, default):
      return default if cls_map is None else cls_map.get(key1, cls_map.get(key2, default))

    self.transform_input = False
    self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=(3,3), stride=2)
    self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=(3,3))
    self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=(3,3), padding=1)
    self.maxpool1 = lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=2, padding=1)
    self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
    self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=(3,3))
    self.maxpool2 = lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=2, padding=1)
    self.Mixed_5b = get_cls("A1","A",InceptionA)(192, pool_feat=32)
    self.Mixed_5c = get_cls("A2","A",InceptionA)(256, pool_feat=64)
    self.Mixed_5d = get_cls("A3","A",InceptionA)(288, pool_feat=64)
    self.Mixed_6a = get_cls("B1","B",InceptionB)(288)
    self.Mixed_6b = get_cls("C1","C",InceptionC)(768, ch_7x7=128)
    self.Mixed_6c = get_cls("C2","C",InceptionC)(768, ch_7x7=160)
    self.Mixed_6d = get_cls("C3","C",InceptionC)(768, ch_7x7=160)
    self.Mixed_6e = get_cls("C4","C",InceptionC)(768, ch_7x7=192)
    self.Mixed_7a = get_cls("D1","D",InceptionD)(768)
    self.Mixed_7b = get_cls("E1","E",InceptionE)(1280)
    self.Mixed_7c = get_cls("E2","E",InceptionE)(2048)
    self.avgpool = lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8), padding=1)
    self.fc = Linear(2048, num_classes)

  def __call__(self, x:Tensor) -> Tensor:
    return x.sequential([
      self.Conv2d_1a_3x3,
      self.Conv2d_2a_3x3,
      self.Conv2d_2b_3x3,
      self.maxpool1,

      self.Conv2d_3b_1x1,
      self.Conv2d_4a_3x3,
      self.maxpool2,

      self.Mixed_5b,
      self.Mixed_5c,
      self.Mixed_5d,
      self.Mixed_6a,
      self.Mixed_6b,
      self.Mixed_6c,
      self.Mixed_6d,
      self.Mixed_6e,

      self.Mixed_7a,
      self.Mixed_7b,
      self.Mixed_7c,
      self.avgpool,

      lambda y: y.reshape(x.shape[0],-1),
      self.fc,
    ])


# FID Inception Variation

class FidInceptionA(InceptionA):
  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      self.branch1x1(x),
      x.sequential([self.branch5x5_1, self.branch5x5_2]),
      x.sequential([self.branch3x3dbl_1, self.branch3x3dbl_2, self.branch3x3dbl_3]),
      self.branch_pool(x.avg_pool2d(kernel_size=(3,3), stride=1, padding=1, count_include_pad=False))
    ]
    return Tensor.cat(*outputs, dim=1)

class FidInceptionC(InceptionC):
  def __call__(self, x:Tensor) -> Tensor:
    outputs = [
      self.branch1x1(x),
      x.sequential([self.branch7x7_1, self.branch7x7_2, self.branch7x7_3]),
      x.sequential([self.branch7x7dbl_1, self.branch7x7dbl_2, self.branch7x7dbl_3, self.branch7x7dbl_4, self.branch7x7dbl_5]),
      self.branch_pool(x.avg_pool2d(kernel_size=(3,3), stride=1, padding=1, count_include_pad=False))
    ]
    return Tensor.cat(*outputs, dim=1)

class FidInceptionE1(InceptionE):
  def __call__(self, x:Tensor) -> Tensor:
    branch3x3 = self.branch3x3_1(x)
    branch3x3dbl = x.sequential([self.branch3x3dbl_1, self.branch3x3dbl_2])
    outputs = [
      self.branch1x1(x),
      Tensor.cat(self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3), dim=1),
      Tensor.cat(self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl), dim=1),
      self.branch_pool(x.avg_pool2d(kernel_size=(3,3), stride=1, padding=1, count_include_pad=False)),
    ]
    return Tensor.cat(*outputs, dim=1)

class FidInceptionE2(InceptionE):
  def __call__(self, x:Tensor) -> Tensor:
    branch3x3 = self.branch3x3_1(x)
    branch3x3dbl = x.sequential([self.branch3x3dbl_1, self.branch3x3dbl_2])
    outputs = [
      self.branch1x1(x),
      Tensor.cat(self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3), dim=1),
      Tensor.cat(self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl), dim=1),
      self.branch_pool(x.max_pool2d(kernel_size=(3,3), stride=1, padding=1)),
    ]
    return Tensor.cat(*outputs, dim=1)

class FidInceptionV3:
  m1: Optional[np.ndarray] = None
  s1: Optional[np.ndarray] = None

  def __init__(self):
    inception = Inception3(cls_map={
      "A":  FidInceptionA,
      "C":  FidInceptionC,
      "E1": FidInceptionE1,
      "E2": FidInceptionE2,
    })

    self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
    self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
    self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3

    self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
    self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3

    self.Mixed_5b = inception.Mixed_5b
    self.Mixed_5c = inception.Mixed_5c
    self.Mixed_5d = inception.Mixed_5d
    self.Mixed_6a = inception.Mixed_6a
    self.Mixed_6b = inception.Mixed_6b
    self.Mixed_6c = inception.Mixed_6c
    self.Mixed_6d = inception.Mixed_6d
    self.Mixed_6e = inception.Mixed_6e

    self.Mixed_7a = inception.Mixed_7a
    self.Mixed_7b = inception.Mixed_7b
    self.Mixed_7c = inception.Mixed_7c

  def load_from_pretrained(self):
    state_dict = torch_load(str(fetch("https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth", "pt_inception-2015-12-05-6726825d.pth")))
    for k,v in state_dict.items():
      if k.endswith(".num_batches_tracked"):
        state_dict[k] = v.reshape(1)
    load_state_dict(self, state_dict)
    return self

  def __call__(self, x:Tensor) -> Tensor:
    x = x.interpolate((299,299), mode="linear")
    x = (x * 2) - 1
    x = x.sequential([
      self.Conv2d_1a_3x3,
      self.Conv2d_2a_3x3,
      self.Conv2d_2b_3x3,
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=2, dilation=1),

      self.Conv2d_3b_1x1,
      self.Conv2d_4a_3x3,
      lambda x: Tensor.max_pool2d(x, kernel_size=(3,3), stride=2, dilation=1),

      self.Mixed_5b,
      self.Mixed_5c,
      self.Mixed_5d,
      self.Mixed_6a,
      self.Mixed_6b,
      self.Mixed_6c,
      self.Mixed_6d,
      self.Mixed_6e,

      self.Mixed_7a,
      self.Mixed_7b,
      self.Mixed_7c,
      lambda x: Tensor.avg_pool2d(x, kernel_size=(8,8)),
    ])
    return x

  def compute_score(self, inception_activations:Tensor, val_stats_path:str) -> float:
    if self.m1 is None and self.s1 is None:
      with np.load(val_stats_path) as f:
        self.m1, self.s1 = f['mu'][:], f['sigma'][:]
    assert self.m1 is not None and self.s1 is not None

    m2 = inception_activations.mean(axis=0).numpy()
    s2 = np.cov(inception_activations.numpy(), rowvar=False)

    return calculate_frechet_distance(self.m1, self.s1, m2, s2)

def calculate_frechet_distance(mu1:np.ndarray, sigma1:np.ndarray, mu2:np.ndarray, sigma2:np.ndarray, eps:float=1e-6) -> float:
  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)
  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)
  assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape

  diff = mu1 - mu2
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  if np.iscomplexobj(covmean):
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError(f"Imaginary component {m}")
    covmean = covmean.real

  tr_covmean = np.trace(covmean)

  return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*tr_covmean
