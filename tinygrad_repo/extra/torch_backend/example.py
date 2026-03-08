from PIL import Image
from tinygrad.helpers import getenv, GlobalCounters
import torch, torchvision, pathlib, warnings
import torchvision.transforms as transforms
import extra.torch_backend.backend
device = "tiny"
torch.set_default_device(device)

if __name__ == "__main__":
  GlobalCounters.reset()
  img = Image.open(pathlib.Path(__file__).parent.parent.parent / "test/models/efficientnet/Chicken.jpg").convert('RGB')
  transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  img = transform(img).unsqueeze(0).to(device)

  model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
  if getenv("EVAL", 1): model.eval()
  out = model(img).detach().cpu().numpy()
  print("output:", out.shape, out.argmax())
  assert out.argmax() == 7  # cock

  kernel_count = GlobalCounters.kernel_count
  assert kernel_count > 0, "No kernels, test failed"
  expected_kernels = 228
  expectation = f"ResNet18 kernels are {kernel_count} vs {expected_kernels} expected."
  if kernel_count < expected_kernels: warnings.warn(f"{expectation} Expectation can be lowered.", UserWarning)
  assert kernel_count <= expected_kernels, f"{expectation}"