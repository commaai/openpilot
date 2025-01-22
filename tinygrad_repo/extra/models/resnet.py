import tinygrad.nn as nn
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch, get_child

# allow monkeypatching in layer implementations
BatchNorm = nn.BatchNorm2d
Conv2d = nn.Conv2d
Linear = nn.Linear


class BasicBlock:
  expansion = 1

  def __init__(self, in_planes, planes, stride=1, groups=1, base_width=64):
    assert groups == 1 and base_width == 64, "BasicBlock only supports groups=1 and base_width=64"
    self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = BatchNorm(planes)
    self.conv2 = Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = BatchNorm(planes)
    self.downsample = []
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = [
        Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        BatchNorm(self.expansion*planes)
      ]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out


class Bottleneck:
  # NOTE: stride_in_1x1=False, this is the v1.5 variant
  expansion = 4

  def __init__(self, in_planes, planes, stride=1, stride_in_1x1=False, groups=1, base_width=64):
    width = int(planes * (base_width / 64.0)) * groups
    # NOTE: the original implementation places stride at the first convolution (self.conv1), control with stride_in_1x1
    self.conv1 = Conv2d(in_planes, width, kernel_size=1, stride=stride if stride_in_1x1 else 1, bias=False)
    self.bn1 = BatchNorm(width)
    self.conv2 = Conv2d(width, width, kernel_size=3, padding=1, stride=1 if stride_in_1x1 else stride, groups=groups, bias=False)
    self.bn2 = BatchNorm(width)
    self.conv3 = Conv2d(width, self.expansion*planes, kernel_size=1, bias=False)
    self.bn3 = BatchNorm(self.expansion*planes)
    self.downsample = []
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = [
        Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        BatchNorm(self.expansion*planes)
      ]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out)).relu()
    out = self.bn3(self.conv3(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out

class ResNet:
  def __init__(self, num, num_classes=None, groups=1, width_per_group=64, stride_in_1x1=False):
    self.num = num
    self.block = {
      18: BasicBlock,
      34: BasicBlock,
      50: Bottleneck,
      101: Bottleneck,
      152: Bottleneck
    }[num]

    self.num_blocks = {
      18: [2,2,2,2],
      34: [3,4,6,3],
      50: [3,4,6,3],
      101: [3,4,23,3],
      152: [3,8,36,3]
    }[num]

    self.in_planes = 64

    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
    self.bn1 = BatchNorm(64)
    self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1, stride_in_1x1=stride_in_1x1)
    self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2, stride_in_1x1=stride_in_1x1)
    self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2, stride_in_1x1=stride_in_1x1)
    self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2, stride_in_1x1=stride_in_1x1)
    self.fc = Linear(512 * self.block.expansion, num_classes) if num_classes is not None else None

  def _make_layer(self, block, planes, num_blocks, stride, stride_in_1x1):
    strides = [stride] + [1] * (num_blocks-1)
    layers = []
    for stride in strides:
      if block == Bottleneck:
        layers.append(block(self.in_planes, planes, stride, stride_in_1x1, self.groups, self.base_width))
      else:
        layers.append(block(self.in_planes, planes, stride, self.groups, self.base_width))
      self.in_planes = planes * block.expansion
    return layers

  def forward(self, x):
    is_feature_only = self.fc is None
    if is_feature_only: features = []
    out = self.bn1(self.conv1(x)).relu()
    out = out.pad([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.layer1)
    if is_feature_only: features.append(out)
    out = out.sequential(self.layer2)
    if is_feature_only: features.append(out)
    out = out.sequential(self.layer3)
    if is_feature_only: features.append(out)
    out = out.sequential(self.layer4)
    if is_feature_only: features.append(out)
    if not is_feature_only:
      out = out.mean([2,3])
      out = self.fc(out.cast(dtypes.float32))
      return out
    return features

  def __call__(self, x:Tensor) -> Tensor:
    return self.forward(x)

  def load_from_pretrained(self):
    model_urls = {
      (18, 1, 64): 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
      (34, 1, 64): 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
      (50, 1, 64): 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
      (50, 32, 4): 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
      (101, 1, 64): 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
      (152, 1, 64): 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    self.url = model_urls[(self.num, self.groups, self.base_width)]
    for k, dat in torch_load(fetch(self.url)).items():
      try:
        obj: Tensor = get_child(self, k)
      except AttributeError as e:
        if 'fc.' in k and self.fc is None:
          continue

        raise e

      if 'fc.' in k and obj.shape != dat.shape:
        print("skipping fully connected layer")
        continue # Skip FC if transfer learning

      if 'bn' not in k and 'downsample' not in k: assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat.to(obj.device).reshape(obj.shape))

ResNet18 = lambda num_classes=1000: ResNet(18, num_classes=num_classes)
ResNet34 = lambda num_classes=1000: ResNet(34, num_classes=num_classes)
ResNet50 = lambda num_classes=1000: ResNet(50, num_classes=num_classes)
ResNet101 = lambda num_classes=1000: ResNet(101, num_classes=num_classes)
ResNet152 = lambda num_classes=1000: ResNet(152, num_classes=num_classes)
ResNeXt50_32X4D = lambda num_classes=1000: ResNet(50, num_classes=num_classes, groups=32, width_per_group=4)

if __name__ == "__main__":
  model = ResNet18()
  model.load_from_pretrained()
  from tinygrad import Context, GlobalCounters, TinyJit
  jmodel = TinyJit(model)
  jmodel(Tensor.rand(1, 3, 224, 224)).realize()
  GlobalCounters.reset()
  jmodel(Tensor.rand(1, 3, 224, 224)).realize()
  for i in range(10): jmodel(Tensor.rand(1, 3, 224, 224))
