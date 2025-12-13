# C++ extension, now uses scons to build
from openpilot.selfdrive.pandad.pandad import can_list_to_can_capnp, can_capnp_to_list
assert can_list_to_can_capnp  # type: ignore[truthy-function]
assert can_capnp_to_list  # type: ignore[truthy-function]
