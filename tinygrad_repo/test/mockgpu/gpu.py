class VirtGPU:
  def __init__(self, gpuid): self.gpuid = gpuid
  def map_range(self, vaddr, size): raise NotImplementedError()
  def unmap_range(self, vaddr, size): raise NotImplementedError()
