from tinygrad.device import Compiled, HostAllocator

class NpyDevice(Compiled):
  def __init__(self, device:str): super().__init__(device, HostAllocator(self), [], None)
