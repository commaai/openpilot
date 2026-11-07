import math
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2d
from tinygrad.helpers import get_child, fetch
from tinygrad.nn.state import torch_load

class MBConvBlock:
  def __init__(self, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio, has_se, track_running_stats=True):
    oup = expand_ratio * input_filters
    if expand_ratio != 1:
      self._expand_conv = Tensor.glorot_uniform(oup, input_filters, 1, 1)
      self._bn0 = BatchNorm2d(oup, track_running_stats=track_running_stats)
    else:
      self._expand_conv = None

    self.strides = strides
    if strides == (2,2):
      self.pad = [(kernel_size-1)//2-1, (kernel_size-1)//2]*2
    else:
      self.pad = [(kernel_size-1)//2]*4

    self._depthwise_conv = Tensor.glorot_uniform(oup, 1, kernel_size, kernel_size)
    self._bn1 = BatchNorm2d(oup, track_running_stats=track_running_stats)

    self.has_se = has_se
    if self.has_se:
      num_squeezed_channels = max(1, int(input_filters * se_ratio))
      self._se_reduce = Tensor.glorot_uniform(num_squeezed_channels, oup, 1, 1)
      self._se_reduce_bias = Tensor.zeros(num_squeezed_channels)
      self._se_expand = Tensor.glorot_uniform(oup, num_squeezed_channels, 1, 1)
      self._se_expand_bias = Tensor.zeros(oup)

    self._project_conv = Tensor.glorot_uniform(output_filters, oup, 1, 1)
    self._bn2 = BatchNorm2d(output_filters, track_running_stats=track_running_stats)

  def __call__(self, inputs):
    x = inputs
    if self._expand_conv is not None:
      x = self._bn0(x.conv2d(self._expand_conv)).swish()
    x = x.conv2d(self._depthwise_conv, padding=self.pad, stride=self.strides, groups=self._depthwise_conv.shape[0])
    x = self._bn1(x).swish()

    if self.has_se:
      x_squeezed = x.avg_pool2d(kernel_size=x.shape[2:4])
      x_squeezed = x_squeezed.conv2d(self._se_reduce, self._se_reduce_bias).swish()
      x_squeezed = x_squeezed.conv2d(self._se_expand, self._se_expand_bias)
      x = x.mul(x_squeezed.sigmoid())

    x = self._bn2(x.conv2d(self._project_conv))
    if x.shape == inputs.shape:
      x = x.add(inputs)
    return x

class EfficientNet:
  def __init__(self, number=0, classes=1000, has_se=True, track_running_stats=True, input_channels=3, has_fc_output=True):
    self.number = number
    global_params = [
      # width, depth
      (1.0, 1.0), # b0
      (1.0, 1.1), # b1
      (1.1, 1.2), # b2
      (1.2, 1.4), # b3
      (1.4, 1.8), # b4
      (1.6, 2.2), # b5
      (1.8, 2.6), # b6
      (2.0, 3.1), # b7
      (2.2, 3.6), # b8
      (4.3, 5.3), # l2
    ][max(number,0)]

    def round_filters(filters):
      multiplier = global_params[0]
      divisor = 8
      filters *= multiplier
      new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
      if new_filters < 0.9 * filters: # prevent rounding by more than 10%
        new_filters += divisor
      return int(new_filters)

    def round_repeats(repeats):
      return int(math.ceil(global_params[1] * repeats))

    out_channels = round_filters(32)
    self._conv_stem = Tensor.glorot_uniform(out_channels, input_channels, 3, 3)
    self._bn0 = BatchNorm2d(out_channels, track_running_stats=track_running_stats)
    blocks_args = [
      [1, 3, (1,1), 1, 32, 16, 0.25],
      [2, 3, (2,2), 6, 16, 24, 0.25],
      [2, 5, (2,2), 6, 24, 40, 0.25],
      [3, 3, (2,2), 6, 40, 80, 0.25],
      [3, 5, (1,1), 6, 80, 112, 0.25],
      [4, 5, (2,2), 6, 112, 192, 0.25],
      [1, 3, (1,1), 6, 192, 320, 0.25],
    ]

    if self.number == -1:
      blocks_args = [
        [1, 3, (2,2), 1, 32, 40, 0.25],
        [1, 3, (2,2), 1, 40, 80, 0.25],
        [1, 3, (2,2), 1, 80, 192, 0.25],
        [1, 3, (2,2), 1, 192, 320, 0.25],
      ]
    elif self.number == -2:
      blocks_args = [
        [1, 9, (8,8), 1, 32, 320, 0.25],
      ]

    self._blocks = []
    for num_repeats, kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio in blocks_args:
      input_filters, output_filters = round_filters(input_filters), round_filters(output_filters)
      for n in range(round_repeats(num_repeats)):
        self._blocks.append(MBConvBlock(kernel_size, strides, expand_ratio, input_filters, output_filters, se_ratio, has_se=has_se, track_running_stats=track_running_stats))
        input_filters = output_filters
        strides = (1,1)

    in_channels = round_filters(320)
    out_channels = round_filters(1280)
    self._conv_head = Tensor.glorot_uniform(out_channels, in_channels, 1, 1)
    self._bn1 = BatchNorm2d(out_channels, track_running_stats=track_running_stats)
    if has_fc_output:
      self._fc = Tensor.glorot_uniform(out_channels, classes)
      self._fc_bias = Tensor.zeros(classes)
    else:
      self._fc = None

  def forward(self, x):
    x = self._bn0(x.conv2d(self._conv_stem, padding=(0,1,0,1), stride=2)).swish()
    x = x.sequential(self._blocks)
    x = self._bn1(x.conv2d(self._conv_head)).swish()
    x = x.avg_pool2d(kernel_size=x.shape[2:4])
    x = x.reshape(shape=(-1, x.shape[1]))
    return x.linear(self._fc, self._fc_bias) if self._fc is not None else x

  def load_from_pretrained(self):
    model_urls = {
      0: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
      1: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
      2: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
      3: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
      4: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
      5: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
      6: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
      7: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth"
    }

    b0 = torch_load(fetch(model_urls[self.number]))
    for k,v in b0.items():
      if k.endswith("num_batches_tracked"): continue
      for cat in ['_conv_head', '_conv_stem', '_depthwise_conv', '_expand_conv', '_fc', '_project_conv', '_se_reduce', '_se_expand']:
        if cat in k:
          k = k.replace('.bias', '_bias')
          k = k.replace('.weight', '')

      #print(k, v.shape)
      mv:Tensor = get_child(self, k)
      vnp = v #.astype(np.float32)
      vnp = vnp if k != '_fc' else vnp.T
      #vnp = vnp if vnp.shape != () else np.array([vnp])

      if mv.shape == vnp.shape:
        mv.replace(vnp.to(mv.device))
      else:
        print("MISMATCH SHAPE IN %s, %r %r" % (k, mv.shape, vnp.shape))

