import random
from typing import Dict, Optional
from tinygrad.helpers import getenv
from tinygrad.runtime.support.allocator import TLSFAllocator

class AllocatorFuzzer:
  def __init__(self, total_size):
    self.total_size = total_size
    self.alloc_payload = 0
    self.mv = memoryview(bytearray(total_size))
    self.alloctor = TLSFAllocator(total_size, block_size=16)

    self.allocations: Dict[int, tuple[int, int]] = {} # ptr -> (size, pattern)

    self.min_alloc_size = 16
    self.max_alloc_size = int(total_size * 0.3)
    self.alloc_probability = 0.7

  def generate_pattern(self, ptr: int, size: int) -> int: return (ptr * 31 + size * 17) & 0xFF

  def fill_memory(self, ptr: int, size: int, pattern: int):
    for i in range(min(size, 32)):
      self.mv[ptr + i] = pattern
      self.mv[ptr + (size - 1 - i)] = pattern

  def verify_memory(self, ptr: int, size: int, pattern: int) -> bool:
    for i in range(min(size, 32)):
      assert self.mv[ptr + i] == pattern
      assert self.mv[ptr + (size - 1 - i)] == pattern
    return True

  def random_alloc(self) -> Optional[int]:
    size = random.randint(self.min_alloc_size, min(self.max_alloc_size, self.total_size - self.alloc_payload))

    try:
      ptr = self.alloctor.alloc(size)
    except MemoryError:
      print(f"Failed to allocate {size} bytes. Payload size is {self.alloc_payload}, so fragmenation is {(size / self.total_size)*100.0:.2f}%")
      return None

    pattern = self.generate_pattern(ptr, size)
    self.fill_memory(ptr, size, pattern)
    self.allocations[ptr] = (size, pattern)
    self.alloc_payload += size
    print(f"Allocated {size} bytes at {ptr:x}, pattern: {pattern:02x}")
    return ptr

  def random_free(self) -> bool:
    if not self.allocations: return False

    ptr = random.choice(list(self.allocations.keys()))
    size, pattern = self.allocations[ptr]

    # Verify pattern before freeing
    if not self.verify_memory(ptr, size, pattern):
      raise RuntimeError(f"Memory corruption detected at {ptr:x}!")

    print(f"Freeing {size} bytes at {ptr:x}, pattern verified: {pattern:02x}")
    self.alloc_payload -= size
    self.alloctor.free(ptr)
    del self.allocations[ptr]
    return True

  def run(self):
    for i in range(getenv("ITERS", 100000)):
      if (random.random() < self.alloc_probability or not self.allocations): self.random_alloc()
      else: self.random_free()

    print("\nCleaning up remaining allocations...")
    while self.allocations: self.random_free()

    print("Fuzzing completed successfully!")

if __name__ == "__main__":
  SEED = getenv("SEED", 42)
  random.seed(SEED)

  fuzzer = AllocatorFuzzer(1 << 30)
  fuzzer.run()
