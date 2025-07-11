import pathlib
from tinygrad import Tensor, Device, Context

if __name__ == "__main__":
  with Context(DEBUG=2):
    disk_llama = Tensor(pathlib.Path("/raid/weights/LLaMA-3/8B/consolidated.00.pth"))
    device_llama = disk_llama.to(Device.DEFAULT).realize()
