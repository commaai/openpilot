import os
import capnp

CEREAL_PATH = os.path.dirname(os.path.abspath(__file__))
capnp.remove_import_hook()

_capnp_imports = [CEREAL_PATH, os.path.join(CEREAL_PATH, "../../opendbc_repo/opendbc/car")]
log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"), imports=_capnp_imports)
car = capnp.load(os.path.join(CEREAL_PATH, "../../opendbc_repo/opendbc/car", "car.capnp"), imports=_capnp_imports)
custom = capnp.load(os.path.join(CEREAL_PATH, "custom.capnp"), imports=_capnp_imports)
