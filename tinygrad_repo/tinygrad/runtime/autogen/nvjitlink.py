# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
import sysconfig
dll = c.DLL('nvjitlink', 'nvJitLink', [f'/{pre}/cuda/targets/{sysconfig.get_config_vars().get("MULTIARCH", "").rsplit("-", 1)[0]}/lib' for pre in ['opt', 'usr/local']])
class nvJitLinkResult(Annotated[int, ctypes.c_uint32], c.Enum): pass
NVJITLINK_SUCCESS = nvJitLinkResult.define('NVJITLINK_SUCCESS', 0)
NVJITLINK_ERROR_UNRECOGNIZED_OPTION = nvJitLinkResult.define('NVJITLINK_ERROR_UNRECOGNIZED_OPTION', 1)
NVJITLINK_ERROR_MISSING_ARCH = nvJitLinkResult.define('NVJITLINK_ERROR_MISSING_ARCH', 2)
NVJITLINK_ERROR_INVALID_INPUT = nvJitLinkResult.define('NVJITLINK_ERROR_INVALID_INPUT', 3)
NVJITLINK_ERROR_PTX_COMPILE = nvJitLinkResult.define('NVJITLINK_ERROR_PTX_COMPILE', 4)
NVJITLINK_ERROR_NVVM_COMPILE = nvJitLinkResult.define('NVJITLINK_ERROR_NVVM_COMPILE', 5)
NVJITLINK_ERROR_INTERNAL = nvJitLinkResult.define('NVJITLINK_ERROR_INTERNAL', 6)

class nvJitLinkInputType(Annotated[int, ctypes.c_uint32], c.Enum): pass
NVJITLINK_INPUT_NONE = nvJitLinkInputType.define('NVJITLINK_INPUT_NONE', 0)
NVJITLINK_INPUT_CUBIN = nvJitLinkInputType.define('NVJITLINK_INPUT_CUBIN', 1)
NVJITLINK_INPUT_PTX = nvJitLinkInputType.define('NVJITLINK_INPUT_PTX', 2)
NVJITLINK_INPUT_LTOIR = nvJitLinkInputType.define('NVJITLINK_INPUT_LTOIR', 3)
NVJITLINK_INPUT_FATBIN = nvJitLinkInputType.define('NVJITLINK_INPUT_FATBIN', 4)
NVJITLINK_INPUT_OBJECT = nvJitLinkInputType.define('NVJITLINK_INPUT_OBJECT', 5)
NVJITLINK_INPUT_LIBRARY = nvJitLinkInputType.define('NVJITLINK_INPUT_LIBRARY', 6)

class struct_nvJitLink(ctypes.Structure): pass
nvJitLinkHandle: TypeAlias = c.POINTER[struct_nvJitLink]
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@dll.bind
def nvJitLinkCreate(handle:c.POINTER[nvJitLinkHandle], numOptions:uint32_t, options:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkDestroy(handle:c.POINTER[nvJitLinkHandle]) -> nvJitLinkResult: ...
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@dll.bind
def nvJitLinkAddData(handle:nvJitLinkHandle, inputType:nvJitLinkInputType, data:ctypes.c_void_p, size:size_t, name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkAddFile(handle:nvJitLinkHandle, inputType:nvJitLinkInputType, fileName:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkComplete(handle:nvJitLinkHandle) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetLinkedCubinSize(handle:nvJitLinkHandle, size:c.POINTER[size_t]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetLinkedCubin(handle:nvJitLinkHandle, cubin:ctypes.c_void_p) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetLinkedPtxSize(handle:nvJitLinkHandle, size:c.POINTER[size_t]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetLinkedPtx(handle:nvJitLinkHandle, ptx:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetErrorLogSize(handle:nvJitLinkHandle, size:c.POINTER[size_t]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetErrorLog(handle:nvJitLinkHandle, log:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetInfoLogSize(handle:nvJitLinkHandle, size:c.POINTER[size_t]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkGetInfoLog(handle:nvJitLinkHandle, log:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> nvJitLinkResult: ...
@dll.bind
def nvJitLinkVersion(major:c.POINTER[Annotated[int, ctypes.c_uint32]], minor:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> nvJitLinkResult: ...
c.init_records()
