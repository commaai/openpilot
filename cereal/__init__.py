# pylint: skip-file
import os
import capnp

CEREAL_PATH = os.path.dirname(os.path.abspath(__file__))
capnp.remove_import_hook()

log = capnp.load(os.path.join(CEREAL_PATH, "log.capnp"))
car = capnp.load(os.path.join(CEREAL_PATH, "car.capnp"))
