import chryslercan
from values import CAR

# bad checksum to reproduce: 14 00 00 00 20 cc
# dat = '\x14\x00\x00\x00\x20' # good checksum: \xcc
[addr, _, dat, _] = chryslercan.create_292(0, 2, True)
print ' '.join('{:02x}'.format(ord(c)) for c in dat)
[addr, _, dat, _] = chryslercan.create_292(-10, 3, True)
print ' '.join('{:02x}'.format(ord(c)) for c in dat)
