from huggingface_hub import snapshot_download
from tinygrad import nn, Tensor, TinyJit, Device, GlobalCounters, Context
import time

class Block:
  def __init__(self, in_dims, dims, stride=1):
    super().__init__()

    self.conv1 = nn.Conv2d(
      in_dims, dims, kernel_size=3, stride=stride, padding=1, bias=False
    )
    self.bn1 = nn.BatchNorm(dims)

    self.conv2 = nn.Conv2d(
      dims, dims, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn2 = nn.BatchNorm(dims)

    self.downsample = []
    if stride != 1:
      self.downsample = [
        nn.Conv2d(in_dims, dims, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm(dims)
      ]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out))
    for l in self.downsample:
      x = l(x)
    out += x
    return out.relu()


class ResNet:
  def __init__(self, block, num_blocks, num_classes=10):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm(64)

    self.layer1 = self._make_layer(block, 64, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 64, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 128, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 256, 512, num_blocks[3], stride=2)

    self.fc = nn.Linear(512, num_classes)

  def _make_layer(self, block, in_dims, dims, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(in_dims, dims, stride))
      in_dims = dims
    return layers

  def __call__(self, x:Tensor):
    x = self.bn1(self.conv1(x)).relu().max_pool2d()
    x = x.sequential(self.layer1)
    with Context(WINO=1): x = x.sequential(self.layer2 + self.layer3 + self.layer4)
    x = x.mean([2, 3])
    x = self.fc(x)
    return x



def load():
  model = ResNet(Block, [2, 2, 2, 2], num_classes=1000)
  file = "model.safetensors"
  model_path = snapshot_download(
    repo_id="awni/resnet18-mlx",
    allow_patterns=[file],
  )
  state = nn.state.safe_load(model_path + "/" + file)
  # mlx is NHWC, tinygrad is NCHW
  nn.state.load_state_dict(model, {k:v if len(v.shape) != 4 else v.to(None).permute(0,3,1,2).contiguous() for k,v in state.items()}, strict=False)
  return model

if __name__ == "__main__":

  resnet18 = load()

  @Tensor.test()
  def _forward(im): return resnet18(im)
  forward = TinyJit(_forward, prune=True)

  batch_sizes = [1, 2, 4, 8, 16, 32, 64]
  #its = 200
  #batch_sizes = [64]
  its = 20
  print(f"Batch Size | Images-per-second | Milliseconds-per-image")
  print(f"---- | ---- | ---- ")
  for N in batch_sizes:
    forward.reset()   # reset the JIT for a new batch size (could make automatic)
    image = Tensor.uniform(N, 3, 288, 288)

    # Warmup
    for _ in range(5):
      GlobalCounters.reset()
      output = forward(image)
    Device.default.synchronize()

    tic = time.time()
    for _ in range(its):
      GlobalCounters.reset()
      output = forward(image)
    Device.default.synchronize()
    toc = time.time()
    ims_per_sec = N * its / (toc - tic)
    ms_per_im = 1e3 / ims_per_sec
    print(f"{N} | {ims_per_sec:.3f} | {ms_per_im:.3f}")