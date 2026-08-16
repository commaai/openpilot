from tinygrad.helpers import Profiling
from tinygrad import Device

if __name__ == "__main__":
  am = Device["AMD"]

  # kfd is 0.55ms!
  with Profiling("allocation 127.7mb"):
    am.allocator.alloc(int(127.7*1024*1024))
