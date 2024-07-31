# Cython, now uses scons to build
from openpilot.selfdrive.pandad.pandad_api_impl import can_list_to_can_capnp, can_capnp_to_list
assert can_list_to_can_capnp
assert can_capnp_to_list

def can_capnp_to_can_list(can, src_filter=None):
  ret = []
  for msg in can:
    if src_filter is None or msg.src in src_filter:
      ret.append((msg.address, msg.dat, msg.src))
  return ret
