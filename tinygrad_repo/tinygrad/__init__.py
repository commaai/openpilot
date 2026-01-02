import os
if int(os.getenv("TYPED", "0")):
  from typeguard import install_import_hook
  install_import_hook(__name__)
from tinygrad.tensor import Tensor                                    # noqa: F401
from tinygrad.engine.jit import TinyJit                               # noqa: F401
from tinygrad.uop.ops import UOp
Variable = UOp.variable
from tinygrad.dtype import dtypes                                     # noqa: F401
from tinygrad.helpers import GlobalCounters, fetch, Context, getenv   # noqa: F401
from tinygrad.device import Device                                    # noqa: F401
