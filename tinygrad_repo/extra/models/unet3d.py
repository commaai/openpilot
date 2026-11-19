from pathlib import Path
import torch
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch, get_child

class DownsampleBlock:
  def __init__(self, c0, c1, stride=2):
    self.conv1 = [nn.Conv2d(c0, c1, kernel_size=(3,3,3), stride=stride, padding=(1,1,1,1,1,1), bias=False), nn.InstanceNorm(c1), Tensor.relu]
    self.conv2 = [nn.Conv2d(c1, c1, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False), nn.InstanceNorm(c1), Tensor.relu]

  def __call__(self, x):
    return x.sequential(self.conv1).sequential(self.conv2)

class UpsampleBlock:
  def __init__(self, c0, c1):
    self.upsample_conv = [nn.ConvTranspose2d(c0, c1, kernel_size=(2,2,2), stride=2)]
    self.conv1 = [nn.Conv2d(2 * c1, c1, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False), nn.InstanceNorm(c1), Tensor.relu]
    self.conv2 = [nn.Conv2d(c1, c1, kernel_size=(3,3,3), padding=(1,1,1,1,1,1), bias=False), nn.InstanceNorm(c1), Tensor.relu]

  def __call__(self, x, skip):
    x = x.sequential(self.upsample_conv)
    x = Tensor.cat(x, skip, dim=1)
    return x.sequential(self.conv1).sequential(self.conv2)

class UNet3D:
  def __init__(self, in_channels=1, n_class=3):
    filters = [32, 64, 128, 256, 320]
    inp, out = filters[:-1], filters[1:]
    self.input_block = DownsampleBlock(in_channels, filters[0], stride=1)
    self.downsample = [DownsampleBlock(i, o) for i, o in zip(inp, out)]
    self.bottleneck = DownsampleBlock(filters[-1], filters[-1])
    self.upsample = [UpsampleBlock(filters[-1], filters[-1])] + [UpsampleBlock(i, o) for i, o in zip(out[::-1], inp[::-1])]
    self.output = {"conv": nn.Conv2d(filters[0], n_class, kernel_size=(1, 1, 1))}

  def __call__(self, x):
    x = self.input_block(x)
    outputs = [x]
    for downsample in self.downsample:
      x = downsample(x)
      outputs.append(x)
    x = self.bottleneck(x)
    for upsample, skip in zip(self.upsample, outputs[::-1]):
      x = upsample(x, skip)
    x = self.output["conv"](x)
    return x

  def load_from_pretrained(self):
    fn = Path(__file__).parents[1] / "weights" / "unet-3d.ckpt"
    fetch("https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch.ptc?download=1", fn)
    state_dict = torch.jit.load(fn, map_location=torch.device("cpu")).state_dict()
    for k, v in state_dict.items():
      obj = get_child(self, k)
      assert obj.shape == v.shape, (k, obj.shape, v.shape)
      obj.assign(v.numpy())

if __name__ == "__main__":
  mdl = UNet3D()
  mdl.load_from_pretrained()
