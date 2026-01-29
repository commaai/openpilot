# CUPTI autogen loader for nv_pma
# To regenerate: REGEN=1 python -c "import extra.nv_pma.cupti"
import importlib, pathlib
from tinygrad.helpers import getenv

root = pathlib.Path(__file__).parents[3]
here = pathlib.Path(__file__).parent

def load(name, dll, files, **kwargs):
  if not (f:=here/f"{name}.py").exists() or getenv('REGEN'):
    kwargs['args'] = kwargs.get('args', [])
    f.write_text(importlib.import_module("tinygrad.runtime.support.autogen").gen(name, dll, files, **kwargs))
  return importlib.import_module(f"extra.nv_pma.cupti.{name}")

def __getattr__(nm):
  match nm:
    case "cupti":
      return load("cupti", "'/usr/local/cuda/targets/x86_64-linux/lib/libcupti.so'", [
        "/usr/local/cuda/include/cupti_result.h", "/usr/local/cuda/include/cupti_activity.h",
        "/usr/local/cuda/include/cupti_callbacks.h", "/usr/local/cuda/include/cupti_events.h",
        "/usr/local/cuda/include/cupti_metrics.h", "/usr/local/cuda/include/cupti_driver_cbid.h",
        "/usr/local/cuda/include/cupti_runtime_cbid.h", "/usr/local/cuda/include/cupti_profiler_target.h",
        "/usr/local/cuda/include/cupti_profiler_host.h", "/usr/local/cuda/include/cupti_pmsampling.h",
        "/usr/local/cuda/include/generated_cuda_meta.h", "/usr/local/cuda/include/generated_cuda_runtime_api_meta.h"
      ], args=["-D__CUDA_API_VERSION_INTERNAL", "-I/usr/local/cuda/include"], parse_macros=False)
    case _: raise AttributeError(f"no such autogen: {nm}")
