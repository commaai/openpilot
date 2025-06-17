from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, LayerNorm, LayerNorm2d, Linear
from tinygrad.helpers import fetch, get_child

class Block:
  def __init__(self, dim):
    self.dwconv = Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
    self.norm = LayerNorm(dim, eps=1e-6)
    self.pwconv1 = Linear(dim, 4 * dim)
    self.pwconv2 = Linear(4 * dim, dim)
    self.gamma = Tensor.ones(dim)

  def __call__(self, x:Tensor):
    return x + x.sequential([
      self.dwconv, lambda x: x.permute(0, 2, 3, 1), self.norm,
      self.pwconv1, Tensor.gelu, self.pwconv2, lambda x: (self.gamma * x).permute(0, 3, 1, 2)
    ])

class ConvNeXt:
  def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
    self.downsample_layers = [
      [Conv2d(in_chans, dims[0], kernel_size=4, stride=4), LayerNorm2d(dims[0], eps=1e-6)],
      *[[LayerNorm2d(dims[i], eps=1e-6), Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)] for i in range(len(dims)-1)]
    ]
    self.stages = [[Block(dims[i]) for _ in range(depths[i])] for i in range(len(dims))]
    self.norm = LayerNorm(dims[-1])
    self.head = Linear(dims[-1], num_classes)

  def __call__(self, x:Tensor):
    for downsample, stage in zip(self.downsample_layers, self.stages):
      x = x.sequential(downsample).sequential(stage)
    return x.mean([-2, -1]).sequential([self.norm, self.head])

# *** model definition is done ***

versions = {
  "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
  "small": {"depths": [3, 3, 27, 3], "dims": [96, 192, 384, 768]},
  "base": {"depths": [3, 3, 9, 3], "dims": [128, 256, 512, 1024]},
  "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
  "xlarge": {"depths": [3, 3, 27, 3], "dims": [256, 512, 1024, 2048]}
}

def get_model(version, load_weights=False):
  model = ConvNeXt(**versions[version])
  if load_weights:
    from tinygrad.nn.state import torch_load
    weights = torch_load(fetch(f'https://dl.fbaipublicfiles.com/convnext/convnext_{version}_1k_224_ema.pth'))['model']
    for k,v in weights.items():
      mv = get_child(model, k)
      mv.assign(v.reshape(mv.shape).to(mv.device)).realize()
  return model

if __name__ == "__main__":
  model = get_model("tiny", True)

  # load image
  from test.models.test_efficientnet import chicken_img, preprocess, _LABELS
  img = Tensor(preprocess(chicken_img))

  Tensor.training = False
  Tensor.no_grad = True

  out = model(img).numpy()
  print(_LABELS[out.argmax()])
