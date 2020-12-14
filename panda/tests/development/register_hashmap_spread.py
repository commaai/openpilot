#!/usr/bin/env python3
import matplotlib.pyplot as plt  # pylint: disable=import-error

HASHING_PRIME = 23
REGISTER_MAP_SIZE = 0x3FF
BYTES_PER_REG = 4

# From ST32F413 datasheet
REGISTER_ADDRESS_REGIONS = [
  (0x40000000, 0x40007FFF),
  (0x40010000, 0x400107FF),
  (0x40011000, 0x400123FF),
  (0x40012C00, 0x40014BFF),
  (0x40015000, 0x400153FF),
  (0x40015800, 0x40015BFF),
  (0x40016000, 0x400167FF),
  (0x40020000, 0x40021FFF),
  (0x40023000, 0x400233FF),
  (0x40023800, 0x40023FFF),
  (0x40026000, 0x400267FF),
  (0x50000000, 0x5003FFFF),
  (0x50060000, 0x500603FF),
  (0x50060800, 0x50060BFF),
  (0x50060800, 0x50060BFF),
  (0xE0000000, 0xE00FFFFF)
]

def _hash(reg_addr):
  return (((reg_addr >> 16) ^ ((((reg_addr + 1) & 0xFFFF) * HASHING_PRIME) & 0xFFFF)) & REGISTER_MAP_SIZE)

# Calculate hash for each address
hashes = []
double_hashes = []
for (start_addr, stop_addr) in REGISTER_ADDRESS_REGIONS:
  for addr in range(start_addr, stop_addr + 1, BYTES_PER_REG):
    h = _hash(addr)
    hashes.append(h)
    double_hashes.append(_hash(h))

# Make histograms
plt.subplot(2, 1, 1)
plt.hist(hashes, bins=REGISTER_MAP_SIZE)
plt.title("Number of collisions per _hash")
plt.xlabel("Address")

plt.subplot(2, 1, 2)
plt.hist(double_hashes, bins=REGISTER_MAP_SIZE)
plt.title("Number of collisions per double _hash")
plt.xlabel("Address")
plt.show()
