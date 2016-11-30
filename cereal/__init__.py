import os
import capnp
capnp.remove_import_hook()

CEREAL_PATH = os.path.dirname(os.path.abspath(__file__))
log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))

