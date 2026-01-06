# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
from tinygrad.helpers import WIN, OSX
dll = DLL('llvm', 'C:\\Program Files\\LLVM\\bin\\LLVM-C.dll' if WIN else '/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib' if OSX else ['LLVM', 'LLVM-21', 'LLVM-20', 'LLVM-19', 'LLVM-18', 'LLVM-17', 'LLVM-16', 'LLVM-15', 'LLVM-14'])
intmax_t = ctypes.c_int64
try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

class imaxdiv_t(Struct): pass
imaxdiv_t._fields_ = [
  ('quot', ctypes.c_int64),
  ('rem', ctypes.c_int64),
]
try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

uintmax_t = ctypes.c_uint64
try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

__gwchar_t = ctypes.c_int32
try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

class fd_set(Struct): pass
__fd_mask = ctypes.c_int64
fd_set._fields_ = [
  ('fds_bits', (ctypes.c_int64 * 16)),
]
class struct_timeval(Struct): pass
__time_t = ctypes.c_int64
__suseconds_t = ctypes.c_int64
struct_timeval._fields_ = [
  ('tv_sec', ctypes.c_int64),
  ('tv_usec', ctypes.c_int64),
]
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

class struct_timespec(Struct): pass
__syscall_slong_t = ctypes.c_int64
struct_timespec._fields_ = [
  ('tv_sec', ctypes.c_int64),
  ('tv_nsec', ctypes.c_int64),
]
class __sigset_t(Struct): pass
__sigset_t._fields_ = [
  ('__val', (ctypes.c_uint64 * 16)),
]
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMVerifierFailureAction = CEnum(ctypes.c_uint32)
LLVMAbortProcessAction = LLVMVerifierFailureAction.define('LLVMAbortProcessAction', 0)
LLVMPrintMessageAction = LLVMVerifierFailureAction.define('LLVMPrintMessageAction', 1)
LLVMReturnStatusAction = LLVMVerifierFailureAction.define('LLVMReturnStatusAction', 2)

LLVMBool = ctypes.c_int32
class struct_LLVMOpaqueModule(Struct): pass
LLVMModuleRef = ctypes.POINTER(struct_LLVMOpaqueModule)
try: (LLVMVerifyModule:=dll.LLVMVerifyModule).restype, LLVMVerifyModule.argtypes = LLVMBool, [LLVMModuleRef, LLVMVerifierFailureAction, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

class struct_LLVMOpaqueValue(Struct): pass
LLVMValueRef = ctypes.POINTER(struct_LLVMOpaqueValue)
try: (LLVMVerifyFunction:=dll.LLVMVerifyFunction).restype, LLVMVerifyFunction.argtypes = LLVMBool, [LLVMValueRef, LLVMVerifierFailureAction]
except AttributeError: pass

try: (LLVMViewFunctionCFG:=dll.LLVMViewFunctionCFG).restype, LLVMViewFunctionCFG.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (LLVMViewFunctionCFGOnly:=dll.LLVMViewFunctionCFGOnly).restype, LLVMViewFunctionCFGOnly.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

class struct_LLVMOpaqueMemoryBuffer(Struct): pass
LLVMMemoryBufferRef = ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer)
try: (LLVMParseBitcode:=dll.LLVMParseBitcode).restype, LLVMParseBitcode.argtypes = LLVMBool, [LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMParseBitcode2:=dll.LLVMParseBitcode2).restype, LLVMParseBitcode2.argtypes = LLVMBool, [LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef)]
except AttributeError: pass

class struct_LLVMOpaqueContext(Struct): pass
LLVMContextRef = ctypes.POINTER(struct_LLVMOpaqueContext)
try: (LLVMParseBitcodeInContext:=dll.LLVMParseBitcodeInContext).restype, LLVMParseBitcodeInContext.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMParseBitcodeInContext2:=dll.LLVMParseBitcodeInContext2).restype, LLVMParseBitcodeInContext2.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef)]
except AttributeError: pass

try: (LLVMGetBitcodeModuleInContext:=dll.LLVMGetBitcodeModuleInContext).restype, LLVMGetBitcodeModuleInContext.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetBitcodeModuleInContext2:=dll.LLVMGetBitcodeModuleInContext2).restype, LLVMGetBitcodeModuleInContext2.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef)]
except AttributeError: pass

try: (LLVMGetBitcodeModule:=dll.LLVMGetBitcodeModule).restype, LLVMGetBitcodeModule.argtypes = LLVMBool, [LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetBitcodeModule2:=dll.LLVMGetBitcodeModule2).restype, LLVMGetBitcodeModule2.argtypes = LLVMBool, [LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef)]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMWriteBitcodeToFile:=dll.LLVMWriteBitcodeToFile).restype, LLVMWriteBitcodeToFile.argtypes = ctypes.c_int32, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMWriteBitcodeToFD:=dll.LLVMWriteBitcodeToFD).restype, LLVMWriteBitcodeToFD.argtypes = ctypes.c_int32, [LLVMModuleRef, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

try: (LLVMWriteBitcodeToFileHandle:=dll.LLVMWriteBitcodeToFileHandle).restype, LLVMWriteBitcodeToFileHandle.argtypes = ctypes.c_int32, [LLVMModuleRef, ctypes.c_int32]
except AttributeError: pass

try: (LLVMWriteBitcodeToMemoryBuffer:=dll.LLVMWriteBitcodeToMemoryBuffer).restype, LLVMWriteBitcodeToMemoryBuffer.argtypes = LLVMMemoryBufferRef, [LLVMModuleRef]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMComdatSelectionKind = CEnum(ctypes.c_uint32)
LLVMAnyComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMAnyComdatSelectionKind', 0)
LLVMExactMatchComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMExactMatchComdatSelectionKind', 1)
LLVMLargestComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMLargestComdatSelectionKind', 2)
LLVMNoDeduplicateComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMNoDeduplicateComdatSelectionKind', 3)
LLVMSameSizeComdatSelectionKind = LLVMComdatSelectionKind.define('LLVMSameSizeComdatSelectionKind', 4)

class struct_LLVMComdat(Struct): pass
LLVMComdatRef = ctypes.POINTER(struct_LLVMComdat)
try: (LLVMGetOrInsertComdat:=dll.LLVMGetOrInsertComdat).restype, LLVMGetOrInsertComdat.argtypes = LLVMComdatRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetComdat:=dll.LLVMGetComdat).restype, LLVMGetComdat.argtypes = LLVMComdatRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetComdat:=dll.LLVMSetComdat).restype, LLVMSetComdat.argtypes = None, [LLVMValueRef, LLVMComdatRef]
except AttributeError: pass

try: (LLVMGetComdatSelectionKind:=dll.LLVMGetComdatSelectionKind).restype, LLVMGetComdatSelectionKind.argtypes = LLVMComdatSelectionKind, [LLVMComdatRef]
except AttributeError: pass

try: (LLVMSetComdatSelectionKind:=dll.LLVMSetComdatSelectionKind).restype, LLVMSetComdatSelectionKind.argtypes = None, [LLVMComdatRef, LLVMComdatSelectionKind]
except AttributeError: pass

LLVMFatalErrorHandler = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char))
try: (LLVMInstallFatalErrorHandler:=dll.LLVMInstallFatalErrorHandler).restype, LLVMInstallFatalErrorHandler.argtypes = None, [LLVMFatalErrorHandler]
except AttributeError: pass

try: (LLVMResetFatalErrorHandler:=dll.LLVMResetFatalErrorHandler).restype, LLVMResetFatalErrorHandler.argtypes = None, []
except AttributeError: pass

try: (LLVMEnablePrettyStackTrace:=dll.LLVMEnablePrettyStackTrace).restype, LLVMEnablePrettyStackTrace.argtypes = None, []
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMOpcode = CEnum(ctypes.c_uint32)
LLVMRet = LLVMOpcode.define('LLVMRet', 1)
LLVMBr = LLVMOpcode.define('LLVMBr', 2)
LLVMSwitch = LLVMOpcode.define('LLVMSwitch', 3)
LLVMIndirectBr = LLVMOpcode.define('LLVMIndirectBr', 4)
LLVMInvoke = LLVMOpcode.define('LLVMInvoke', 5)
LLVMUnreachable = LLVMOpcode.define('LLVMUnreachable', 7)
LLVMCallBr = LLVMOpcode.define('LLVMCallBr', 67)
LLVMFNeg = LLVMOpcode.define('LLVMFNeg', 66)
LLVMAdd = LLVMOpcode.define('LLVMAdd', 8)
LLVMFAdd = LLVMOpcode.define('LLVMFAdd', 9)
LLVMSub = LLVMOpcode.define('LLVMSub', 10)
LLVMFSub = LLVMOpcode.define('LLVMFSub', 11)
LLVMMul = LLVMOpcode.define('LLVMMul', 12)
LLVMFMul = LLVMOpcode.define('LLVMFMul', 13)
LLVMUDiv = LLVMOpcode.define('LLVMUDiv', 14)
LLVMSDiv = LLVMOpcode.define('LLVMSDiv', 15)
LLVMFDiv = LLVMOpcode.define('LLVMFDiv', 16)
LLVMURem = LLVMOpcode.define('LLVMURem', 17)
LLVMSRem = LLVMOpcode.define('LLVMSRem', 18)
LLVMFRem = LLVMOpcode.define('LLVMFRem', 19)
LLVMShl = LLVMOpcode.define('LLVMShl', 20)
LLVMLShr = LLVMOpcode.define('LLVMLShr', 21)
LLVMAShr = LLVMOpcode.define('LLVMAShr', 22)
LLVMAnd = LLVMOpcode.define('LLVMAnd', 23)
LLVMOr = LLVMOpcode.define('LLVMOr', 24)
LLVMXor = LLVMOpcode.define('LLVMXor', 25)
LLVMAlloca = LLVMOpcode.define('LLVMAlloca', 26)
LLVMLoad = LLVMOpcode.define('LLVMLoad', 27)
LLVMStore = LLVMOpcode.define('LLVMStore', 28)
LLVMGetElementPtr = LLVMOpcode.define('LLVMGetElementPtr', 29)
LLVMTrunc = LLVMOpcode.define('LLVMTrunc', 30)
LLVMZExt = LLVMOpcode.define('LLVMZExt', 31)
LLVMSExt = LLVMOpcode.define('LLVMSExt', 32)
LLVMFPToUI = LLVMOpcode.define('LLVMFPToUI', 33)
LLVMFPToSI = LLVMOpcode.define('LLVMFPToSI', 34)
LLVMUIToFP = LLVMOpcode.define('LLVMUIToFP', 35)
LLVMSIToFP = LLVMOpcode.define('LLVMSIToFP', 36)
LLVMFPTrunc = LLVMOpcode.define('LLVMFPTrunc', 37)
LLVMFPExt = LLVMOpcode.define('LLVMFPExt', 38)
LLVMPtrToInt = LLVMOpcode.define('LLVMPtrToInt', 39)
LLVMIntToPtr = LLVMOpcode.define('LLVMIntToPtr', 40)
LLVMBitCast = LLVMOpcode.define('LLVMBitCast', 41)
LLVMAddrSpaceCast = LLVMOpcode.define('LLVMAddrSpaceCast', 60)
LLVMICmp = LLVMOpcode.define('LLVMICmp', 42)
LLVMFCmp = LLVMOpcode.define('LLVMFCmp', 43)
LLVMPHI = LLVMOpcode.define('LLVMPHI', 44)
LLVMCall = LLVMOpcode.define('LLVMCall', 45)
LLVMSelect = LLVMOpcode.define('LLVMSelect', 46)
LLVMUserOp1 = LLVMOpcode.define('LLVMUserOp1', 47)
LLVMUserOp2 = LLVMOpcode.define('LLVMUserOp2', 48)
LLVMVAArg = LLVMOpcode.define('LLVMVAArg', 49)
LLVMExtractElement = LLVMOpcode.define('LLVMExtractElement', 50)
LLVMInsertElement = LLVMOpcode.define('LLVMInsertElement', 51)
LLVMShuffleVector = LLVMOpcode.define('LLVMShuffleVector', 52)
LLVMExtractValue = LLVMOpcode.define('LLVMExtractValue', 53)
LLVMInsertValue = LLVMOpcode.define('LLVMInsertValue', 54)
LLVMFreeze = LLVMOpcode.define('LLVMFreeze', 68)
LLVMFence = LLVMOpcode.define('LLVMFence', 55)
LLVMAtomicCmpXchg = LLVMOpcode.define('LLVMAtomicCmpXchg', 56)
LLVMAtomicRMW = LLVMOpcode.define('LLVMAtomicRMW', 57)
LLVMResume = LLVMOpcode.define('LLVMResume', 58)
LLVMLandingPad = LLVMOpcode.define('LLVMLandingPad', 59)
LLVMCleanupRet = LLVMOpcode.define('LLVMCleanupRet', 61)
LLVMCatchRet = LLVMOpcode.define('LLVMCatchRet', 62)
LLVMCatchPad = LLVMOpcode.define('LLVMCatchPad', 63)
LLVMCleanupPad = LLVMOpcode.define('LLVMCleanupPad', 64)
LLVMCatchSwitch = LLVMOpcode.define('LLVMCatchSwitch', 65)

LLVMTypeKind = CEnum(ctypes.c_uint32)
LLVMVoidTypeKind = LLVMTypeKind.define('LLVMVoidTypeKind', 0)
LLVMHalfTypeKind = LLVMTypeKind.define('LLVMHalfTypeKind', 1)
LLVMFloatTypeKind = LLVMTypeKind.define('LLVMFloatTypeKind', 2)
LLVMDoubleTypeKind = LLVMTypeKind.define('LLVMDoubleTypeKind', 3)
LLVMX86_FP80TypeKind = LLVMTypeKind.define('LLVMX86_FP80TypeKind', 4)
LLVMFP128TypeKind = LLVMTypeKind.define('LLVMFP128TypeKind', 5)
LLVMPPC_FP128TypeKind = LLVMTypeKind.define('LLVMPPC_FP128TypeKind', 6)
LLVMLabelTypeKind = LLVMTypeKind.define('LLVMLabelTypeKind', 7)
LLVMIntegerTypeKind = LLVMTypeKind.define('LLVMIntegerTypeKind', 8)
LLVMFunctionTypeKind = LLVMTypeKind.define('LLVMFunctionTypeKind', 9)
LLVMStructTypeKind = LLVMTypeKind.define('LLVMStructTypeKind', 10)
LLVMArrayTypeKind = LLVMTypeKind.define('LLVMArrayTypeKind', 11)
LLVMPointerTypeKind = LLVMTypeKind.define('LLVMPointerTypeKind', 12)
LLVMVectorTypeKind = LLVMTypeKind.define('LLVMVectorTypeKind', 13)
LLVMMetadataTypeKind = LLVMTypeKind.define('LLVMMetadataTypeKind', 14)
LLVMTokenTypeKind = LLVMTypeKind.define('LLVMTokenTypeKind', 16)
LLVMScalableVectorTypeKind = LLVMTypeKind.define('LLVMScalableVectorTypeKind', 17)
LLVMBFloatTypeKind = LLVMTypeKind.define('LLVMBFloatTypeKind', 18)
LLVMX86_AMXTypeKind = LLVMTypeKind.define('LLVMX86_AMXTypeKind', 19)
LLVMTargetExtTypeKind = LLVMTypeKind.define('LLVMTargetExtTypeKind', 20)

LLVMLinkage = CEnum(ctypes.c_uint32)
LLVMExternalLinkage = LLVMLinkage.define('LLVMExternalLinkage', 0)
LLVMAvailableExternallyLinkage = LLVMLinkage.define('LLVMAvailableExternallyLinkage', 1)
LLVMLinkOnceAnyLinkage = LLVMLinkage.define('LLVMLinkOnceAnyLinkage', 2)
LLVMLinkOnceODRLinkage = LLVMLinkage.define('LLVMLinkOnceODRLinkage', 3)
LLVMLinkOnceODRAutoHideLinkage = LLVMLinkage.define('LLVMLinkOnceODRAutoHideLinkage', 4)
LLVMWeakAnyLinkage = LLVMLinkage.define('LLVMWeakAnyLinkage', 5)
LLVMWeakODRLinkage = LLVMLinkage.define('LLVMWeakODRLinkage', 6)
LLVMAppendingLinkage = LLVMLinkage.define('LLVMAppendingLinkage', 7)
LLVMInternalLinkage = LLVMLinkage.define('LLVMInternalLinkage', 8)
LLVMPrivateLinkage = LLVMLinkage.define('LLVMPrivateLinkage', 9)
LLVMDLLImportLinkage = LLVMLinkage.define('LLVMDLLImportLinkage', 10)
LLVMDLLExportLinkage = LLVMLinkage.define('LLVMDLLExportLinkage', 11)
LLVMExternalWeakLinkage = LLVMLinkage.define('LLVMExternalWeakLinkage', 12)
LLVMGhostLinkage = LLVMLinkage.define('LLVMGhostLinkage', 13)
LLVMCommonLinkage = LLVMLinkage.define('LLVMCommonLinkage', 14)
LLVMLinkerPrivateLinkage = LLVMLinkage.define('LLVMLinkerPrivateLinkage', 15)
LLVMLinkerPrivateWeakLinkage = LLVMLinkage.define('LLVMLinkerPrivateWeakLinkage', 16)

LLVMVisibility = CEnum(ctypes.c_uint32)
LLVMDefaultVisibility = LLVMVisibility.define('LLVMDefaultVisibility', 0)
LLVMHiddenVisibility = LLVMVisibility.define('LLVMHiddenVisibility', 1)
LLVMProtectedVisibility = LLVMVisibility.define('LLVMProtectedVisibility', 2)

LLVMUnnamedAddr = CEnum(ctypes.c_uint32)
LLVMNoUnnamedAddr = LLVMUnnamedAddr.define('LLVMNoUnnamedAddr', 0)
LLVMLocalUnnamedAddr = LLVMUnnamedAddr.define('LLVMLocalUnnamedAddr', 1)
LLVMGlobalUnnamedAddr = LLVMUnnamedAddr.define('LLVMGlobalUnnamedAddr', 2)

LLVMDLLStorageClass = CEnum(ctypes.c_uint32)
LLVMDefaultStorageClass = LLVMDLLStorageClass.define('LLVMDefaultStorageClass', 0)
LLVMDLLImportStorageClass = LLVMDLLStorageClass.define('LLVMDLLImportStorageClass', 1)
LLVMDLLExportStorageClass = LLVMDLLStorageClass.define('LLVMDLLExportStorageClass', 2)

LLVMCallConv = CEnum(ctypes.c_uint32)
LLVMCCallConv = LLVMCallConv.define('LLVMCCallConv', 0)
LLVMFastCallConv = LLVMCallConv.define('LLVMFastCallConv', 8)
LLVMColdCallConv = LLVMCallConv.define('LLVMColdCallConv', 9)
LLVMGHCCallConv = LLVMCallConv.define('LLVMGHCCallConv', 10)
LLVMHiPECallConv = LLVMCallConv.define('LLVMHiPECallConv', 11)
LLVMAnyRegCallConv = LLVMCallConv.define('LLVMAnyRegCallConv', 13)
LLVMPreserveMostCallConv = LLVMCallConv.define('LLVMPreserveMostCallConv', 14)
LLVMPreserveAllCallConv = LLVMCallConv.define('LLVMPreserveAllCallConv', 15)
LLVMSwiftCallConv = LLVMCallConv.define('LLVMSwiftCallConv', 16)
LLVMCXXFASTTLSCallConv = LLVMCallConv.define('LLVMCXXFASTTLSCallConv', 17)
LLVMX86StdcallCallConv = LLVMCallConv.define('LLVMX86StdcallCallConv', 64)
LLVMX86FastcallCallConv = LLVMCallConv.define('LLVMX86FastcallCallConv', 65)
LLVMARMAPCSCallConv = LLVMCallConv.define('LLVMARMAPCSCallConv', 66)
LLVMARMAAPCSCallConv = LLVMCallConv.define('LLVMARMAAPCSCallConv', 67)
LLVMARMAAPCSVFPCallConv = LLVMCallConv.define('LLVMARMAAPCSVFPCallConv', 68)
LLVMMSP430INTRCallConv = LLVMCallConv.define('LLVMMSP430INTRCallConv', 69)
LLVMX86ThisCallCallConv = LLVMCallConv.define('LLVMX86ThisCallCallConv', 70)
LLVMPTXKernelCallConv = LLVMCallConv.define('LLVMPTXKernelCallConv', 71)
LLVMPTXDeviceCallConv = LLVMCallConv.define('LLVMPTXDeviceCallConv', 72)
LLVMSPIRFUNCCallConv = LLVMCallConv.define('LLVMSPIRFUNCCallConv', 75)
LLVMSPIRKERNELCallConv = LLVMCallConv.define('LLVMSPIRKERNELCallConv', 76)
LLVMIntelOCLBICallConv = LLVMCallConv.define('LLVMIntelOCLBICallConv', 77)
LLVMX8664SysVCallConv = LLVMCallConv.define('LLVMX8664SysVCallConv', 78)
LLVMWin64CallConv = LLVMCallConv.define('LLVMWin64CallConv', 79)
LLVMX86VectorCallCallConv = LLVMCallConv.define('LLVMX86VectorCallCallConv', 80)
LLVMHHVMCallConv = LLVMCallConv.define('LLVMHHVMCallConv', 81)
LLVMHHVMCCallConv = LLVMCallConv.define('LLVMHHVMCCallConv', 82)
LLVMX86INTRCallConv = LLVMCallConv.define('LLVMX86INTRCallConv', 83)
LLVMAVRINTRCallConv = LLVMCallConv.define('LLVMAVRINTRCallConv', 84)
LLVMAVRSIGNALCallConv = LLVMCallConv.define('LLVMAVRSIGNALCallConv', 85)
LLVMAVRBUILTINCallConv = LLVMCallConv.define('LLVMAVRBUILTINCallConv', 86)
LLVMAMDGPUVSCallConv = LLVMCallConv.define('LLVMAMDGPUVSCallConv', 87)
LLVMAMDGPUGSCallConv = LLVMCallConv.define('LLVMAMDGPUGSCallConv', 88)
LLVMAMDGPUPSCallConv = LLVMCallConv.define('LLVMAMDGPUPSCallConv', 89)
LLVMAMDGPUCSCallConv = LLVMCallConv.define('LLVMAMDGPUCSCallConv', 90)
LLVMAMDGPUKERNELCallConv = LLVMCallConv.define('LLVMAMDGPUKERNELCallConv', 91)
LLVMX86RegCallCallConv = LLVMCallConv.define('LLVMX86RegCallCallConv', 92)
LLVMAMDGPUHSCallConv = LLVMCallConv.define('LLVMAMDGPUHSCallConv', 93)
LLVMMSP430BUILTINCallConv = LLVMCallConv.define('LLVMMSP430BUILTINCallConv', 94)
LLVMAMDGPULSCallConv = LLVMCallConv.define('LLVMAMDGPULSCallConv', 95)
LLVMAMDGPUESCallConv = LLVMCallConv.define('LLVMAMDGPUESCallConv', 96)

LLVMValueKind = CEnum(ctypes.c_uint32)
LLVMArgumentValueKind = LLVMValueKind.define('LLVMArgumentValueKind', 0)
LLVMBasicBlockValueKind = LLVMValueKind.define('LLVMBasicBlockValueKind', 1)
LLVMMemoryUseValueKind = LLVMValueKind.define('LLVMMemoryUseValueKind', 2)
LLVMMemoryDefValueKind = LLVMValueKind.define('LLVMMemoryDefValueKind', 3)
LLVMMemoryPhiValueKind = LLVMValueKind.define('LLVMMemoryPhiValueKind', 4)
LLVMFunctionValueKind = LLVMValueKind.define('LLVMFunctionValueKind', 5)
LLVMGlobalAliasValueKind = LLVMValueKind.define('LLVMGlobalAliasValueKind', 6)
LLVMGlobalIFuncValueKind = LLVMValueKind.define('LLVMGlobalIFuncValueKind', 7)
LLVMGlobalVariableValueKind = LLVMValueKind.define('LLVMGlobalVariableValueKind', 8)
LLVMBlockAddressValueKind = LLVMValueKind.define('LLVMBlockAddressValueKind', 9)
LLVMConstantExprValueKind = LLVMValueKind.define('LLVMConstantExprValueKind', 10)
LLVMConstantArrayValueKind = LLVMValueKind.define('LLVMConstantArrayValueKind', 11)
LLVMConstantStructValueKind = LLVMValueKind.define('LLVMConstantStructValueKind', 12)
LLVMConstantVectorValueKind = LLVMValueKind.define('LLVMConstantVectorValueKind', 13)
LLVMUndefValueValueKind = LLVMValueKind.define('LLVMUndefValueValueKind', 14)
LLVMConstantAggregateZeroValueKind = LLVMValueKind.define('LLVMConstantAggregateZeroValueKind', 15)
LLVMConstantDataArrayValueKind = LLVMValueKind.define('LLVMConstantDataArrayValueKind', 16)
LLVMConstantDataVectorValueKind = LLVMValueKind.define('LLVMConstantDataVectorValueKind', 17)
LLVMConstantIntValueKind = LLVMValueKind.define('LLVMConstantIntValueKind', 18)
LLVMConstantFPValueKind = LLVMValueKind.define('LLVMConstantFPValueKind', 19)
LLVMConstantPointerNullValueKind = LLVMValueKind.define('LLVMConstantPointerNullValueKind', 20)
LLVMConstantTokenNoneValueKind = LLVMValueKind.define('LLVMConstantTokenNoneValueKind', 21)
LLVMMetadataAsValueValueKind = LLVMValueKind.define('LLVMMetadataAsValueValueKind', 22)
LLVMInlineAsmValueKind = LLVMValueKind.define('LLVMInlineAsmValueKind', 23)
LLVMInstructionValueKind = LLVMValueKind.define('LLVMInstructionValueKind', 24)
LLVMPoisonValueValueKind = LLVMValueKind.define('LLVMPoisonValueValueKind', 25)
LLVMConstantTargetNoneValueKind = LLVMValueKind.define('LLVMConstantTargetNoneValueKind', 26)
LLVMConstantPtrAuthValueKind = LLVMValueKind.define('LLVMConstantPtrAuthValueKind', 27)

LLVMIntPredicate = CEnum(ctypes.c_uint32)
LLVMIntEQ = LLVMIntPredicate.define('LLVMIntEQ', 32)
LLVMIntNE = LLVMIntPredicate.define('LLVMIntNE', 33)
LLVMIntUGT = LLVMIntPredicate.define('LLVMIntUGT', 34)
LLVMIntUGE = LLVMIntPredicate.define('LLVMIntUGE', 35)
LLVMIntULT = LLVMIntPredicate.define('LLVMIntULT', 36)
LLVMIntULE = LLVMIntPredicate.define('LLVMIntULE', 37)
LLVMIntSGT = LLVMIntPredicate.define('LLVMIntSGT', 38)
LLVMIntSGE = LLVMIntPredicate.define('LLVMIntSGE', 39)
LLVMIntSLT = LLVMIntPredicate.define('LLVMIntSLT', 40)
LLVMIntSLE = LLVMIntPredicate.define('LLVMIntSLE', 41)

LLVMRealPredicate = CEnum(ctypes.c_uint32)
LLVMRealPredicateFalse = LLVMRealPredicate.define('LLVMRealPredicateFalse', 0)
LLVMRealOEQ = LLVMRealPredicate.define('LLVMRealOEQ', 1)
LLVMRealOGT = LLVMRealPredicate.define('LLVMRealOGT', 2)
LLVMRealOGE = LLVMRealPredicate.define('LLVMRealOGE', 3)
LLVMRealOLT = LLVMRealPredicate.define('LLVMRealOLT', 4)
LLVMRealOLE = LLVMRealPredicate.define('LLVMRealOLE', 5)
LLVMRealONE = LLVMRealPredicate.define('LLVMRealONE', 6)
LLVMRealORD = LLVMRealPredicate.define('LLVMRealORD', 7)
LLVMRealUNO = LLVMRealPredicate.define('LLVMRealUNO', 8)
LLVMRealUEQ = LLVMRealPredicate.define('LLVMRealUEQ', 9)
LLVMRealUGT = LLVMRealPredicate.define('LLVMRealUGT', 10)
LLVMRealUGE = LLVMRealPredicate.define('LLVMRealUGE', 11)
LLVMRealULT = LLVMRealPredicate.define('LLVMRealULT', 12)
LLVMRealULE = LLVMRealPredicate.define('LLVMRealULE', 13)
LLVMRealUNE = LLVMRealPredicate.define('LLVMRealUNE', 14)
LLVMRealPredicateTrue = LLVMRealPredicate.define('LLVMRealPredicateTrue', 15)

LLVMLandingPadClauseTy = CEnum(ctypes.c_uint32)
LLVMLandingPadCatch = LLVMLandingPadClauseTy.define('LLVMLandingPadCatch', 0)
LLVMLandingPadFilter = LLVMLandingPadClauseTy.define('LLVMLandingPadFilter', 1)

LLVMThreadLocalMode = CEnum(ctypes.c_uint32)
LLVMNotThreadLocal = LLVMThreadLocalMode.define('LLVMNotThreadLocal', 0)
LLVMGeneralDynamicTLSModel = LLVMThreadLocalMode.define('LLVMGeneralDynamicTLSModel', 1)
LLVMLocalDynamicTLSModel = LLVMThreadLocalMode.define('LLVMLocalDynamicTLSModel', 2)
LLVMInitialExecTLSModel = LLVMThreadLocalMode.define('LLVMInitialExecTLSModel', 3)
LLVMLocalExecTLSModel = LLVMThreadLocalMode.define('LLVMLocalExecTLSModel', 4)

LLVMAtomicOrdering = CEnum(ctypes.c_uint32)
LLVMAtomicOrderingNotAtomic = LLVMAtomicOrdering.define('LLVMAtomicOrderingNotAtomic', 0)
LLVMAtomicOrderingUnordered = LLVMAtomicOrdering.define('LLVMAtomicOrderingUnordered', 1)
LLVMAtomicOrderingMonotonic = LLVMAtomicOrdering.define('LLVMAtomicOrderingMonotonic', 2)
LLVMAtomicOrderingAcquire = LLVMAtomicOrdering.define('LLVMAtomicOrderingAcquire', 4)
LLVMAtomicOrderingRelease = LLVMAtomicOrdering.define('LLVMAtomicOrderingRelease', 5)
LLVMAtomicOrderingAcquireRelease = LLVMAtomicOrdering.define('LLVMAtomicOrderingAcquireRelease', 6)
LLVMAtomicOrderingSequentiallyConsistent = LLVMAtomicOrdering.define('LLVMAtomicOrderingSequentiallyConsistent', 7)

LLVMAtomicRMWBinOp = CEnum(ctypes.c_uint32)
LLVMAtomicRMWBinOpXchg = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpXchg', 0)
LLVMAtomicRMWBinOpAdd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpAdd', 1)
LLVMAtomicRMWBinOpSub = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpSub', 2)
LLVMAtomicRMWBinOpAnd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpAnd', 3)
LLVMAtomicRMWBinOpNand = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpNand', 4)
LLVMAtomicRMWBinOpOr = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpOr', 5)
LLVMAtomicRMWBinOpXor = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpXor', 6)
LLVMAtomicRMWBinOpMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpMax', 7)
LLVMAtomicRMWBinOpMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpMin', 8)
LLVMAtomicRMWBinOpUMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUMax', 9)
LLVMAtomicRMWBinOpUMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUMin', 10)
LLVMAtomicRMWBinOpFAdd = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFAdd', 11)
LLVMAtomicRMWBinOpFSub = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFSub', 12)
LLVMAtomicRMWBinOpFMax = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFMax', 13)
LLVMAtomicRMWBinOpFMin = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpFMin', 14)
LLVMAtomicRMWBinOpUIncWrap = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUIncWrap', 15)
LLVMAtomicRMWBinOpUDecWrap = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUDecWrap', 16)
LLVMAtomicRMWBinOpUSubCond = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUSubCond', 17)
LLVMAtomicRMWBinOpUSubSat = LLVMAtomicRMWBinOp.define('LLVMAtomicRMWBinOpUSubSat', 18)

LLVMDiagnosticSeverity = CEnum(ctypes.c_uint32)
LLVMDSError = LLVMDiagnosticSeverity.define('LLVMDSError', 0)
LLVMDSWarning = LLVMDiagnosticSeverity.define('LLVMDSWarning', 1)
LLVMDSRemark = LLVMDiagnosticSeverity.define('LLVMDSRemark', 2)
LLVMDSNote = LLVMDiagnosticSeverity.define('LLVMDSNote', 3)

LLVMInlineAsmDialect = CEnum(ctypes.c_uint32)
LLVMInlineAsmDialectATT = LLVMInlineAsmDialect.define('LLVMInlineAsmDialectATT', 0)
LLVMInlineAsmDialectIntel = LLVMInlineAsmDialect.define('LLVMInlineAsmDialectIntel', 1)

LLVMModuleFlagBehavior = CEnum(ctypes.c_uint32)
LLVMModuleFlagBehaviorError = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorError', 0)
LLVMModuleFlagBehaviorWarning = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorWarning', 1)
LLVMModuleFlagBehaviorRequire = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorRequire', 2)
LLVMModuleFlagBehaviorOverride = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorOverride', 3)
LLVMModuleFlagBehaviorAppend = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorAppend', 4)
LLVMModuleFlagBehaviorAppendUnique = LLVMModuleFlagBehavior.define('LLVMModuleFlagBehaviorAppendUnique', 5)

_anonenum0 = CEnum(ctypes.c_int32)
LLVMAttributeReturnIndex = _anonenum0.define('LLVMAttributeReturnIndex', 0)
LLVMAttributeFunctionIndex = _anonenum0.define('LLVMAttributeFunctionIndex', -1)

LLVMAttributeIndex = ctypes.c_uint32
LLVMTailCallKind = CEnum(ctypes.c_uint32)
LLVMTailCallKindNone = LLVMTailCallKind.define('LLVMTailCallKindNone', 0)
LLVMTailCallKindTail = LLVMTailCallKind.define('LLVMTailCallKindTail', 1)
LLVMTailCallKindMustTail = LLVMTailCallKind.define('LLVMTailCallKindMustTail', 2)
LLVMTailCallKindNoTail = LLVMTailCallKind.define('LLVMTailCallKindNoTail', 3)

_anonenum1 = CEnum(ctypes.c_uint32)
LLVMFastMathAllowReassoc = _anonenum1.define('LLVMFastMathAllowReassoc', 1)
LLVMFastMathNoNaNs = _anonenum1.define('LLVMFastMathNoNaNs', 2)
LLVMFastMathNoInfs = _anonenum1.define('LLVMFastMathNoInfs', 4)
LLVMFastMathNoSignedZeros = _anonenum1.define('LLVMFastMathNoSignedZeros', 8)
LLVMFastMathAllowReciprocal = _anonenum1.define('LLVMFastMathAllowReciprocal', 16)
LLVMFastMathAllowContract = _anonenum1.define('LLVMFastMathAllowContract', 32)
LLVMFastMathApproxFunc = _anonenum1.define('LLVMFastMathApproxFunc', 64)
LLVMFastMathNone = _anonenum1.define('LLVMFastMathNone', 0)
LLVMFastMathAll = _anonenum1.define('LLVMFastMathAll', 127)

LLVMFastMathFlags = ctypes.c_uint32
_anonenum2 = CEnum(ctypes.c_uint32)
LLVMGEPFlagInBounds = _anonenum2.define('LLVMGEPFlagInBounds', 1)
LLVMGEPFlagNUSW = _anonenum2.define('LLVMGEPFlagNUSW', 2)
LLVMGEPFlagNUW = _anonenum2.define('LLVMGEPFlagNUW', 4)

LLVMGEPNoWrapFlags = ctypes.c_uint32
try: (LLVMShutdown:=dll.LLVMShutdown).restype, LLVMShutdown.argtypes = None, []
except AttributeError: pass

try: (LLVMGetVersion:=dll.LLVMGetVersion).restype, LLVMGetVersion.argtypes = None, [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (LLVMCreateMessage:=dll.LLVMCreateMessage).restype, LLVMCreateMessage.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeMessage:=dll.LLVMDisposeMessage).restype, LLVMDisposeMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOpaqueDiagnosticInfo(Struct): pass
LLVMDiagnosticHandler = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_LLVMOpaqueDiagnosticInfo), ctypes.c_void_p)
LLVMYieldCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_LLVMOpaqueContext), ctypes.c_void_p)
try: (LLVMContextCreate:=dll.LLVMContextCreate).restype, LLVMContextCreate.argtypes = LLVMContextRef, []
except AttributeError: pass

try: (LLVMGetGlobalContext:=dll.LLVMGetGlobalContext).restype, LLVMGetGlobalContext.argtypes = LLVMContextRef, []
except AttributeError: pass

try: (LLVMContextSetDiagnosticHandler:=dll.LLVMContextSetDiagnosticHandler).restype, LLVMContextSetDiagnosticHandler.argtypes = None, [LLVMContextRef, LLVMDiagnosticHandler, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMContextGetDiagnosticHandler:=dll.LLVMContextGetDiagnosticHandler).restype, LLVMContextGetDiagnosticHandler.argtypes = LLVMDiagnosticHandler, [LLVMContextRef]
except AttributeError: pass

try: (LLVMContextGetDiagnosticContext:=dll.LLVMContextGetDiagnosticContext).restype, LLVMContextGetDiagnosticContext.argtypes = ctypes.c_void_p, [LLVMContextRef]
except AttributeError: pass

try: (LLVMContextSetYieldCallback:=dll.LLVMContextSetYieldCallback).restype, LLVMContextSetYieldCallback.argtypes = None, [LLVMContextRef, LLVMYieldCallback, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMContextShouldDiscardValueNames:=dll.LLVMContextShouldDiscardValueNames).restype, LLVMContextShouldDiscardValueNames.argtypes = LLVMBool, [LLVMContextRef]
except AttributeError: pass

try: (LLVMContextSetDiscardValueNames:=dll.LLVMContextSetDiscardValueNames).restype, LLVMContextSetDiscardValueNames.argtypes = None, [LLVMContextRef, LLVMBool]
except AttributeError: pass

try: (LLVMContextDispose:=dll.LLVMContextDispose).restype, LLVMContextDispose.argtypes = None, [LLVMContextRef]
except AttributeError: pass

LLVMDiagnosticInfoRef = ctypes.POINTER(struct_LLVMOpaqueDiagnosticInfo)
try: (LLVMGetDiagInfoDescription:=dll.LLVMGetDiagInfoDescription).restype, LLVMGetDiagInfoDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMDiagnosticInfoRef]
except AttributeError: pass

try: (LLVMGetDiagInfoSeverity:=dll.LLVMGetDiagInfoSeverity).restype, LLVMGetDiagInfoSeverity.argtypes = LLVMDiagnosticSeverity, [LLVMDiagnosticInfoRef]
except AttributeError: pass

try: (LLVMGetMDKindIDInContext:=dll.LLVMGetMDKindIDInContext).restype, LLVMGetMDKindIDInContext.argtypes = ctypes.c_uint32, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetMDKindID:=dll.LLVMGetMDKindID).restype, LLVMGetMDKindID.argtypes = ctypes.c_uint32, [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

size_t = ctypes.c_uint64
try: (LLVMGetSyncScopeID:=dll.LLVMGetSyncScopeID).restype, LLVMGetSyncScopeID.argtypes = ctypes.c_uint32, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetEnumAttributeKindForName:=dll.LLVMGetEnumAttributeKindForName).restype, LLVMGetEnumAttributeKindForName.argtypes = ctypes.c_uint32, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetLastEnumAttributeKind:=dll.LLVMGetLastEnumAttributeKind).restype, LLVMGetLastEnumAttributeKind.argtypes = ctypes.c_uint32, []
except AttributeError: pass

class struct_LLVMOpaqueAttributeRef(Struct): pass
LLVMAttributeRef = ctypes.POINTER(struct_LLVMOpaqueAttributeRef)
uint64_t = ctypes.c_uint64
try: (LLVMCreateEnumAttribute:=dll.LLVMCreateEnumAttribute).restype, LLVMCreateEnumAttribute.argtypes = LLVMAttributeRef, [LLVMContextRef, ctypes.c_uint32, uint64_t]
except AttributeError: pass

try: (LLVMGetEnumAttributeKind:=dll.LLVMGetEnumAttributeKind).restype, LLVMGetEnumAttributeKind.argtypes = ctypes.c_uint32, [LLVMAttributeRef]
except AttributeError: pass

try: (LLVMGetEnumAttributeValue:=dll.LLVMGetEnumAttributeValue).restype, LLVMGetEnumAttributeValue.argtypes = uint64_t, [LLVMAttributeRef]
except AttributeError: pass

class struct_LLVMOpaqueType(Struct): pass
LLVMTypeRef = ctypes.POINTER(struct_LLVMOpaqueType)
try: (LLVMCreateTypeAttribute:=dll.LLVMCreateTypeAttribute).restype, LLVMCreateTypeAttribute.argtypes = LLVMAttributeRef, [LLVMContextRef, ctypes.c_uint32, LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetTypeAttributeValue:=dll.LLVMGetTypeAttributeValue).restype, LLVMGetTypeAttributeValue.argtypes = LLVMTypeRef, [LLVMAttributeRef]
except AttributeError: pass

try: (LLVMCreateConstantRangeAttribute:=dll.LLVMCreateConstantRangeAttribute).restype, LLVMCreateConstantRangeAttribute.argtypes = LLVMAttributeRef, [LLVMContextRef, ctypes.c_uint32, ctypes.c_uint32, (uint64_t * 0), (uint64_t * 0)]
except AttributeError: pass

try: (LLVMCreateStringAttribute:=dll.LLVMCreateStringAttribute).restype, LLVMCreateStringAttribute.argtypes = LLVMAttributeRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetStringAttributeKind:=dll.LLVMGetStringAttributeKind).restype, LLVMGetStringAttributeKind.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMAttributeRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (LLVMGetStringAttributeValue:=dll.LLVMGetStringAttributeValue).restype, LLVMGetStringAttributeValue.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMAttributeRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (LLVMIsEnumAttribute:=dll.LLVMIsEnumAttribute).restype, LLVMIsEnumAttribute.argtypes = LLVMBool, [LLVMAttributeRef]
except AttributeError: pass

try: (LLVMIsStringAttribute:=dll.LLVMIsStringAttribute).restype, LLVMIsStringAttribute.argtypes = LLVMBool, [LLVMAttributeRef]
except AttributeError: pass

try: (LLVMIsTypeAttribute:=dll.LLVMIsTypeAttribute).restype, LLVMIsTypeAttribute.argtypes = LLVMBool, [LLVMAttributeRef]
except AttributeError: pass

try: (LLVMGetTypeByName2:=dll.LLVMGetTypeByName2).restype, LLVMGetTypeByName2.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMModuleCreateWithName:=dll.LLVMModuleCreateWithName).restype, LLVMModuleCreateWithName.argtypes = LLVMModuleRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMModuleCreateWithNameInContext:=dll.LLVMModuleCreateWithNameInContext).restype, LLVMModuleCreateWithNameInContext.argtypes = LLVMModuleRef, [ctypes.POINTER(ctypes.c_char), LLVMContextRef]
except AttributeError: pass

try: (LLVMCloneModule:=dll.LLVMCloneModule).restype, LLVMCloneModule.argtypes = LLVMModuleRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMDisposeModule:=dll.LLVMDisposeModule).restype, LLVMDisposeModule.argtypes = None, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMIsNewDbgInfoFormat:=dll.LLVMIsNewDbgInfoFormat).restype, LLVMIsNewDbgInfoFormat.argtypes = LLVMBool, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetIsNewDbgInfoFormat:=dll.LLVMSetIsNewDbgInfoFormat).restype, LLVMSetIsNewDbgInfoFormat.argtypes = None, [LLVMModuleRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetModuleIdentifier:=dll.LLVMGetModuleIdentifier).restype, LLVMGetModuleIdentifier.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMSetModuleIdentifier:=dll.LLVMSetModuleIdentifier).restype, LLVMSetModuleIdentifier.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetSourceFileName:=dll.LLVMGetSourceFileName).restype, LLVMGetSourceFileName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMSetSourceFileName:=dll.LLVMSetSourceFileName).restype, LLVMSetSourceFileName.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetDataLayoutStr:=dll.LLVMGetDataLayoutStr).restype, LLVMGetDataLayoutStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetDataLayout:=dll.LLVMGetDataLayout).restype, LLVMGetDataLayout.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetDataLayout:=dll.LLVMSetDataLayout).restype, LLVMSetDataLayout.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetTarget:=dll.LLVMGetTarget).restype, LLVMGetTarget.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetTarget:=dll.LLVMSetTarget).restype, LLVMSetTarget.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOpaqueModuleFlagEntry(Struct): pass
LLVMModuleFlagEntry = struct_LLVMOpaqueModuleFlagEntry
try: (LLVMCopyModuleFlagsMetadata:=dll.LLVMCopyModuleFlagsMetadata).restype, LLVMCopyModuleFlagsMetadata.argtypes = ctypes.POINTER(LLVMModuleFlagEntry), [LLVMModuleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMDisposeModuleFlagsMetadata:=dll.LLVMDisposeModuleFlagsMetadata).restype, LLVMDisposeModuleFlagsMetadata.argtypes = None, [ctypes.POINTER(LLVMModuleFlagEntry)]
except AttributeError: pass

try: (LLVMModuleFlagEntriesGetFlagBehavior:=dll.LLVMModuleFlagEntriesGetFlagBehavior).restype, LLVMModuleFlagEntriesGetFlagBehavior.argtypes = LLVMModuleFlagBehavior, [ctypes.POINTER(LLVMModuleFlagEntry), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMModuleFlagEntriesGetKey:=dll.LLVMModuleFlagEntriesGetKey).restype, LLVMModuleFlagEntriesGetKey.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(LLVMModuleFlagEntry), ctypes.c_uint32, ctypes.POINTER(size_t)]
except AttributeError: pass

class struct_LLVMOpaqueMetadata(Struct): pass
LLVMMetadataRef = ctypes.POINTER(struct_LLVMOpaqueMetadata)
try: (LLVMModuleFlagEntriesGetMetadata:=dll.LLVMModuleFlagEntriesGetMetadata).restype, LLVMModuleFlagEntriesGetMetadata.argtypes = LLVMMetadataRef, [ctypes.POINTER(LLVMModuleFlagEntry), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetModuleFlag:=dll.LLVMGetModuleFlag).restype, LLVMGetModuleFlag.argtypes = LLVMMetadataRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMAddModuleFlag:=dll.LLVMAddModuleFlag).restype, LLVMAddModuleFlag.argtypes = None, [LLVMModuleRef, LLVMModuleFlagBehavior, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDumpModule:=dll.LLVMDumpModule).restype, LLVMDumpModule.argtypes = None, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMPrintModuleToFile:=dll.LLVMPrintModuleToFile).restype, LLVMPrintModuleToFile.argtypes = LLVMBool, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMPrintModuleToString:=dll.LLVMPrintModuleToString).restype, LLVMPrintModuleToString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetModuleInlineAsm:=dll.LLVMGetModuleInlineAsm).restype, LLVMGetModuleInlineAsm.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMSetModuleInlineAsm2:=dll.LLVMSetModuleInlineAsm2).restype, LLVMSetModuleInlineAsm2.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMAppendModuleInlineAsm:=dll.LLVMAppendModuleInlineAsm).restype, LLVMAppendModuleInlineAsm.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetInlineAsm:=dll.LLVMGetInlineAsm).restype, LLVMGetInlineAsm.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool, LLVMBool, LLVMInlineAsmDialect, LLVMBool]
except AttributeError: pass

try: (LLVMGetInlineAsmAsmString:=dll.LLVMGetInlineAsmAsmString).restype, LLVMGetInlineAsmAsmString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMGetInlineAsmConstraintString:=dll.LLVMGetInlineAsmConstraintString).restype, LLVMGetInlineAsmConstraintString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMGetInlineAsmDialect:=dll.LLVMGetInlineAsmDialect).restype, LLVMGetInlineAsmDialect.argtypes = LLVMInlineAsmDialect, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetInlineAsmFunctionType:=dll.LLVMGetInlineAsmFunctionType).restype, LLVMGetInlineAsmFunctionType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetInlineAsmHasSideEffects:=dll.LLVMGetInlineAsmHasSideEffects).restype, LLVMGetInlineAsmHasSideEffects.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetInlineAsmNeedsAlignedStack:=dll.LLVMGetInlineAsmNeedsAlignedStack).restype, LLVMGetInlineAsmNeedsAlignedStack.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetInlineAsmCanUnwind:=dll.LLVMGetInlineAsmCanUnwind).restype, LLVMGetInlineAsmCanUnwind.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetModuleContext:=dll.LLVMGetModuleContext).restype, LLVMGetModuleContext.argtypes = LLVMContextRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetTypeByName:=dll.LLVMGetTypeByName).restype, LLVMGetTypeByName.argtypes = LLVMTypeRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOpaqueNamedMDNode(Struct): pass
LLVMNamedMDNodeRef = ctypes.POINTER(struct_LLVMOpaqueNamedMDNode)
try: (LLVMGetFirstNamedMetadata:=dll.LLVMGetFirstNamedMetadata).restype, LLVMGetFirstNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetLastNamedMetadata:=dll.LLVMGetLastNamedMetadata).restype, LLVMGetLastNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetNextNamedMetadata:=dll.LLVMGetNextNamedMetadata).restype, LLVMGetNextNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMNamedMDNodeRef]
except AttributeError: pass

try: (LLVMGetPreviousNamedMetadata:=dll.LLVMGetPreviousNamedMetadata).restype, LLVMGetPreviousNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMNamedMDNodeRef]
except AttributeError: pass

try: (LLVMGetNamedMetadata:=dll.LLVMGetNamedMetadata).restype, LLVMGetNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetOrInsertNamedMetadata:=dll.LLVMGetOrInsertNamedMetadata).restype, LLVMGetOrInsertNamedMetadata.argtypes = LLVMNamedMDNodeRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetNamedMetadataName:=dll.LLVMGetNamedMetadataName).restype, LLVMGetNamedMetadataName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMNamedMDNodeRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMGetNamedMetadataNumOperands:=dll.LLVMGetNamedMetadataNumOperands).restype, LLVMGetNamedMetadataNumOperands.argtypes = ctypes.c_uint32, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetNamedMetadataOperands:=dll.LLVMGetNamedMetadataOperands).restype, LLVMGetNamedMetadataOperands.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

try: (LLVMAddNamedMetadataOperand:=dll.LLVMAddNamedMetadataOperand).restype, LLVMAddNamedMetadataOperand.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMValueRef]
except AttributeError: pass

try: (LLVMGetDebugLocDirectory:=dll.LLVMGetDebugLocDirectory).restype, LLVMGetDebugLocDirectory.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (LLVMGetDebugLocFilename:=dll.LLVMGetDebugLocFilename).restype, LLVMGetDebugLocFilename.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (LLVMGetDebugLocLine:=dll.LLVMGetDebugLocLine).restype, LLVMGetDebugLocLine.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetDebugLocColumn:=dll.LLVMGetDebugLocColumn).restype, LLVMGetDebugLocColumn.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMAddFunction:=dll.LLVMAddFunction).restype, LLVMAddFunction.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetNamedFunction:=dll.LLVMGetNamedFunction).restype, LLVMGetNamedFunction.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetNamedFunctionWithLength:=dll.LLVMGetNamedFunctionWithLength).restype, LLVMGetNamedFunctionWithLength.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetFirstFunction:=dll.LLVMGetFirstFunction).restype, LLVMGetFirstFunction.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetLastFunction:=dll.LLVMGetLastFunction).restype, LLVMGetLastFunction.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetNextFunction:=dll.LLVMGetNextFunction).restype, LLVMGetNextFunction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetPreviousFunction:=dll.LLVMGetPreviousFunction).restype, LLVMGetPreviousFunction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetModuleInlineAsm:=dll.LLVMSetModuleInlineAsm).restype, LLVMSetModuleInlineAsm.argtypes = None, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetTypeKind:=dll.LLVMGetTypeKind).restype, LLVMGetTypeKind.argtypes = LLVMTypeKind, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMTypeIsSized:=dll.LLVMTypeIsSized).restype, LLVMTypeIsSized.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetTypeContext:=dll.LLVMGetTypeContext).restype, LLVMGetTypeContext.argtypes = LLVMContextRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMDumpType:=dll.LLVMDumpType).restype, LLVMDumpType.argtypes = None, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMPrintTypeToString:=dll.LLVMPrintTypeToString).restype, LLVMPrintTypeToString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTypeRef]
except AttributeError: pass

try: (LLVMInt1TypeInContext:=dll.LLVMInt1TypeInContext).restype, LLVMInt1TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMInt8TypeInContext:=dll.LLVMInt8TypeInContext).restype, LLVMInt8TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMInt16TypeInContext:=dll.LLVMInt16TypeInContext).restype, LLVMInt16TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMInt32TypeInContext:=dll.LLVMInt32TypeInContext).restype, LLVMInt32TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMInt64TypeInContext:=dll.LLVMInt64TypeInContext).restype, LLVMInt64TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMInt128TypeInContext:=dll.LLVMInt128TypeInContext).restype, LLVMInt128TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMIntTypeInContext:=dll.LLVMIntTypeInContext).restype, LLVMIntTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMInt1Type:=dll.LLVMInt1Type).restype, LLVMInt1Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMInt8Type:=dll.LLVMInt8Type).restype, LLVMInt8Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMInt16Type:=dll.LLVMInt16Type).restype, LLVMInt16Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMInt32Type:=dll.LLVMInt32Type).restype, LLVMInt32Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMInt64Type:=dll.LLVMInt64Type).restype, LLVMInt64Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMInt128Type:=dll.LLVMInt128Type).restype, LLVMInt128Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMIntType:=dll.LLVMIntType).restype, LLVMIntType.argtypes = LLVMTypeRef, [ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetIntTypeWidth:=dll.LLVMGetIntTypeWidth).restype, LLVMGetIntTypeWidth.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMHalfTypeInContext:=dll.LLVMHalfTypeInContext).restype, LLVMHalfTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMBFloatTypeInContext:=dll.LLVMBFloatTypeInContext).restype, LLVMBFloatTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMFloatTypeInContext:=dll.LLVMFloatTypeInContext).restype, LLVMFloatTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMDoubleTypeInContext:=dll.LLVMDoubleTypeInContext).restype, LLVMDoubleTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMX86FP80TypeInContext:=dll.LLVMX86FP80TypeInContext).restype, LLVMX86FP80TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMFP128TypeInContext:=dll.LLVMFP128TypeInContext).restype, LLVMFP128TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMPPCFP128TypeInContext:=dll.LLVMPPCFP128TypeInContext).restype, LLVMPPCFP128TypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMHalfType:=dll.LLVMHalfType).restype, LLVMHalfType.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMBFloatType:=dll.LLVMBFloatType).restype, LLVMBFloatType.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMFloatType:=dll.LLVMFloatType).restype, LLVMFloatType.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMDoubleType:=dll.LLVMDoubleType).restype, LLVMDoubleType.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMX86FP80Type:=dll.LLVMX86FP80Type).restype, LLVMX86FP80Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMFP128Type:=dll.LLVMFP128Type).restype, LLVMFP128Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMPPCFP128Type:=dll.LLVMPPCFP128Type).restype, LLVMPPCFP128Type.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMFunctionType:=dll.LLVMFunctionType).restype, LLVMFunctionType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

try: (LLVMIsFunctionVarArg:=dll.LLVMIsFunctionVarArg).restype, LLVMIsFunctionVarArg.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetReturnType:=dll.LLVMGetReturnType).restype, LLVMGetReturnType.argtypes = LLVMTypeRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMCountParamTypes:=dll.LLVMCountParamTypes).restype, LLVMCountParamTypes.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetParamTypes:=dll.LLVMGetParamTypes).restype, LLVMGetParamTypes.argtypes = None, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef)]
except AttributeError: pass

try: (LLVMStructTypeInContext:=dll.LLVMStructTypeInContext).restype, LLVMStructTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

try: (LLVMStructType:=dll.LLVMStructType).restype, LLVMStructType.argtypes = LLVMTypeRef, [ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

try: (LLVMStructCreateNamed:=dll.LLVMStructCreateNamed).restype, LLVMStructCreateNamed.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetStructName:=dll.LLVMGetStructName).restype, LLVMGetStructName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTypeRef]
except AttributeError: pass

try: (LLVMStructSetBody:=dll.LLVMStructSetBody).restype, LLVMStructSetBody.argtypes = None, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

try: (LLVMCountStructElementTypes:=dll.LLVMCountStructElementTypes).restype, LLVMCountStructElementTypes.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetStructElementTypes:=dll.LLVMGetStructElementTypes).restype, LLVMGetStructElementTypes.argtypes = None, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef)]
except AttributeError: pass

try: (LLVMStructGetTypeAtIndex:=dll.LLVMStructGetTypeAtIndex).restype, LLVMStructGetTypeAtIndex.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIsPackedStruct:=dll.LLVMIsPackedStruct).restype, LLVMIsPackedStruct.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMIsOpaqueStruct:=dll.LLVMIsOpaqueStruct).restype, LLVMIsOpaqueStruct.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMIsLiteralStruct:=dll.LLVMIsLiteralStruct).restype, LLVMIsLiteralStruct.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetElementType:=dll.LLVMGetElementType).restype, LLVMGetElementType.argtypes = LLVMTypeRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetSubtypes:=dll.LLVMGetSubtypes).restype, LLVMGetSubtypes.argtypes = None, [LLVMTypeRef, ctypes.POINTER(LLVMTypeRef)]
except AttributeError: pass

try: (LLVMGetNumContainedTypes:=dll.LLVMGetNumContainedTypes).restype, LLVMGetNumContainedTypes.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMArrayType:=dll.LLVMArrayType).restype, LLVMArrayType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMArrayType2:=dll.LLVMArrayType2).restype, LLVMArrayType2.argtypes = LLVMTypeRef, [LLVMTypeRef, uint64_t]
except AttributeError: pass

try: (LLVMGetArrayLength:=dll.LLVMGetArrayLength).restype, LLVMGetArrayLength.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetArrayLength2:=dll.LLVMGetArrayLength2).restype, LLVMGetArrayLength2.argtypes = uint64_t, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMPointerType:=dll.LLVMPointerType).restype, LLVMPointerType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMPointerTypeIsOpaque:=dll.LLVMPointerTypeIsOpaque).restype, LLVMPointerTypeIsOpaque.argtypes = LLVMBool, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMPointerTypeInContext:=dll.LLVMPointerTypeInContext).restype, LLVMPointerTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetPointerAddressSpace:=dll.LLVMGetPointerAddressSpace).restype, LLVMGetPointerAddressSpace.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMVectorType:=dll.LLVMVectorType).restype, LLVMVectorType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMScalableVectorType:=dll.LLVMScalableVectorType).restype, LLVMScalableVectorType.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetVectorSize:=dll.LLVMGetVectorSize).restype, LLVMGetVectorSize.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetConstantPtrAuthPointer:=dll.LLVMGetConstantPtrAuthPointer).restype, LLVMGetConstantPtrAuthPointer.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetConstantPtrAuthKey:=dll.LLVMGetConstantPtrAuthKey).restype, LLVMGetConstantPtrAuthKey.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetConstantPtrAuthDiscriminator:=dll.LLVMGetConstantPtrAuthDiscriminator).restype, LLVMGetConstantPtrAuthDiscriminator.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetConstantPtrAuthAddrDiscriminator:=dll.LLVMGetConstantPtrAuthAddrDiscriminator).restype, LLVMGetConstantPtrAuthAddrDiscriminator.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMVoidTypeInContext:=dll.LLVMVoidTypeInContext).restype, LLVMVoidTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMLabelTypeInContext:=dll.LLVMLabelTypeInContext).restype, LLVMLabelTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMX86AMXTypeInContext:=dll.LLVMX86AMXTypeInContext).restype, LLVMX86AMXTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMTokenTypeInContext:=dll.LLVMTokenTypeInContext).restype, LLVMTokenTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMMetadataTypeInContext:=dll.LLVMMetadataTypeInContext).restype, LLVMMetadataTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMVoidType:=dll.LLVMVoidType).restype, LLVMVoidType.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMLabelType:=dll.LLVMLabelType).restype, LLVMLabelType.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMX86AMXType:=dll.LLVMX86AMXType).restype, LLVMX86AMXType.argtypes = LLVMTypeRef, []
except AttributeError: pass

try: (LLVMTargetExtTypeInContext:=dll.LLVMTargetExtTypeInContext).restype, LLVMTargetExtTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTypeRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetTargetExtTypeName:=dll.LLVMGetTargetExtTypeName).restype, LLVMGetTargetExtTypeName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetTargetExtTypeNumTypeParams:=dll.LLVMGetTargetExtTypeNumTypeParams).restype, LLVMGetTargetExtTypeNumTypeParams.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetTargetExtTypeTypeParam:=dll.LLVMGetTargetExtTypeTypeParam).restype, LLVMGetTargetExtTypeTypeParam.argtypes = LLVMTypeRef, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetTargetExtTypeNumIntParams:=dll.LLVMGetTargetExtTypeNumIntParams).restype, LLVMGetTargetExtTypeNumIntParams.argtypes = ctypes.c_uint32, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetTargetExtTypeIntParam:=dll.LLVMGetTargetExtTypeIntParam).restype, LLVMGetTargetExtTypeIntParam.argtypes = ctypes.c_uint32, [LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMTypeOf:=dll.LLVMTypeOf).restype, LLVMTypeOf.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetValueKind:=dll.LLVMGetValueKind).restype, LLVMGetValueKind.argtypes = LLVMValueKind, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetValueName2:=dll.LLVMGetValueName2).restype, LLVMGetValueName2.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMSetValueName2:=dll.LLVMSetValueName2).restype, LLVMSetValueName2.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDumpValue:=dll.LLVMDumpValue).restype, LLVMDumpValue.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (LLVMPrintValueToString:=dll.LLVMPrintValueToString).restype, LLVMPrintValueToString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetValueContext:=dll.LLVMGetValueContext).restype, LLVMGetValueContext.argtypes = LLVMContextRef, [LLVMValueRef]
except AttributeError: pass

class struct_LLVMOpaqueDbgRecord(Struct): pass
LLVMDbgRecordRef = ctypes.POINTER(struct_LLVMOpaqueDbgRecord)
try: (LLVMPrintDbgRecordToString:=dll.LLVMPrintDbgRecordToString).restype, LLVMPrintDbgRecordToString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMDbgRecordRef]
except AttributeError: pass

try: (LLVMReplaceAllUsesWith:=dll.LLVMReplaceAllUsesWith).restype, LLVMReplaceAllUsesWith.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMIsConstant:=dll.LLVMIsConstant).restype, LLVMIsConstant.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsUndef:=dll.LLVMIsUndef).restype, LLVMIsUndef.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsPoison:=dll.LLVMIsPoison).restype, LLVMIsPoison.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAArgument:=dll.LLVMIsAArgument).restype, LLVMIsAArgument.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsABasicBlock:=dll.LLVMIsABasicBlock).restype, LLVMIsABasicBlock.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAInlineAsm:=dll.LLVMIsAInlineAsm).restype, LLVMIsAInlineAsm.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAUser:=dll.LLVMIsAUser).restype, LLVMIsAUser.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstant:=dll.LLVMIsAConstant).restype, LLVMIsAConstant.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsABlockAddress:=dll.LLVMIsABlockAddress).restype, LLVMIsABlockAddress.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantAggregateZero:=dll.LLVMIsAConstantAggregateZero).restype, LLVMIsAConstantAggregateZero.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantArray:=dll.LLVMIsAConstantArray).restype, LLVMIsAConstantArray.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantDataSequential:=dll.LLVMIsAConstantDataSequential).restype, LLVMIsAConstantDataSequential.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantDataArray:=dll.LLVMIsAConstantDataArray).restype, LLVMIsAConstantDataArray.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantDataVector:=dll.LLVMIsAConstantDataVector).restype, LLVMIsAConstantDataVector.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantExpr:=dll.LLVMIsAConstantExpr).restype, LLVMIsAConstantExpr.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantFP:=dll.LLVMIsAConstantFP).restype, LLVMIsAConstantFP.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantInt:=dll.LLVMIsAConstantInt).restype, LLVMIsAConstantInt.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantPointerNull:=dll.LLVMIsAConstantPointerNull).restype, LLVMIsAConstantPointerNull.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantStruct:=dll.LLVMIsAConstantStruct).restype, LLVMIsAConstantStruct.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantTokenNone:=dll.LLVMIsAConstantTokenNone).restype, LLVMIsAConstantTokenNone.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantVector:=dll.LLVMIsAConstantVector).restype, LLVMIsAConstantVector.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAConstantPtrAuth:=dll.LLVMIsAConstantPtrAuth).restype, LLVMIsAConstantPtrAuth.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAGlobalValue:=dll.LLVMIsAGlobalValue).restype, LLVMIsAGlobalValue.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAGlobalAlias:=dll.LLVMIsAGlobalAlias).restype, LLVMIsAGlobalAlias.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAGlobalObject:=dll.LLVMIsAGlobalObject).restype, LLVMIsAGlobalObject.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAFunction:=dll.LLVMIsAFunction).restype, LLVMIsAFunction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAGlobalVariable:=dll.LLVMIsAGlobalVariable).restype, LLVMIsAGlobalVariable.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAGlobalIFunc:=dll.LLVMIsAGlobalIFunc).restype, LLVMIsAGlobalIFunc.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAUndefValue:=dll.LLVMIsAUndefValue).restype, LLVMIsAUndefValue.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAPoisonValue:=dll.LLVMIsAPoisonValue).restype, LLVMIsAPoisonValue.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAInstruction:=dll.LLVMIsAInstruction).restype, LLVMIsAInstruction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAUnaryOperator:=dll.LLVMIsAUnaryOperator).restype, LLVMIsAUnaryOperator.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsABinaryOperator:=dll.LLVMIsABinaryOperator).restype, LLVMIsABinaryOperator.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsACallInst:=dll.LLVMIsACallInst).restype, LLVMIsACallInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAIntrinsicInst:=dll.LLVMIsAIntrinsicInst).restype, LLVMIsAIntrinsicInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsADbgInfoIntrinsic:=dll.LLVMIsADbgInfoIntrinsic).restype, LLVMIsADbgInfoIntrinsic.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsADbgVariableIntrinsic:=dll.LLVMIsADbgVariableIntrinsic).restype, LLVMIsADbgVariableIntrinsic.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsADbgDeclareInst:=dll.LLVMIsADbgDeclareInst).restype, LLVMIsADbgDeclareInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsADbgLabelInst:=dll.LLVMIsADbgLabelInst).restype, LLVMIsADbgLabelInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAMemIntrinsic:=dll.LLVMIsAMemIntrinsic).restype, LLVMIsAMemIntrinsic.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAMemCpyInst:=dll.LLVMIsAMemCpyInst).restype, LLVMIsAMemCpyInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAMemMoveInst:=dll.LLVMIsAMemMoveInst).restype, LLVMIsAMemMoveInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAMemSetInst:=dll.LLVMIsAMemSetInst).restype, LLVMIsAMemSetInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsACmpInst:=dll.LLVMIsACmpInst).restype, LLVMIsACmpInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAFCmpInst:=dll.LLVMIsAFCmpInst).restype, LLVMIsAFCmpInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAICmpInst:=dll.LLVMIsAICmpInst).restype, LLVMIsAICmpInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAExtractElementInst:=dll.LLVMIsAExtractElementInst).restype, LLVMIsAExtractElementInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAGetElementPtrInst:=dll.LLVMIsAGetElementPtrInst).restype, LLVMIsAGetElementPtrInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAInsertElementInst:=dll.LLVMIsAInsertElementInst).restype, LLVMIsAInsertElementInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAInsertValueInst:=dll.LLVMIsAInsertValueInst).restype, LLVMIsAInsertValueInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsALandingPadInst:=dll.LLVMIsALandingPadInst).restype, LLVMIsALandingPadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAPHINode:=dll.LLVMIsAPHINode).restype, LLVMIsAPHINode.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsASelectInst:=dll.LLVMIsASelectInst).restype, LLVMIsASelectInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAShuffleVectorInst:=dll.LLVMIsAShuffleVectorInst).restype, LLVMIsAShuffleVectorInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAStoreInst:=dll.LLVMIsAStoreInst).restype, LLVMIsAStoreInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsABranchInst:=dll.LLVMIsABranchInst).restype, LLVMIsABranchInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAIndirectBrInst:=dll.LLVMIsAIndirectBrInst).restype, LLVMIsAIndirectBrInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAInvokeInst:=dll.LLVMIsAInvokeInst).restype, LLVMIsAInvokeInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAReturnInst:=dll.LLVMIsAReturnInst).restype, LLVMIsAReturnInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsASwitchInst:=dll.LLVMIsASwitchInst).restype, LLVMIsASwitchInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAUnreachableInst:=dll.LLVMIsAUnreachableInst).restype, LLVMIsAUnreachableInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAResumeInst:=dll.LLVMIsAResumeInst).restype, LLVMIsAResumeInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsACleanupReturnInst:=dll.LLVMIsACleanupReturnInst).restype, LLVMIsACleanupReturnInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsACatchReturnInst:=dll.LLVMIsACatchReturnInst).restype, LLVMIsACatchReturnInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsACatchSwitchInst:=dll.LLVMIsACatchSwitchInst).restype, LLVMIsACatchSwitchInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsACallBrInst:=dll.LLVMIsACallBrInst).restype, LLVMIsACallBrInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAFuncletPadInst:=dll.LLVMIsAFuncletPadInst).restype, LLVMIsAFuncletPadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsACatchPadInst:=dll.LLVMIsACatchPadInst).restype, LLVMIsACatchPadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsACleanupPadInst:=dll.LLVMIsACleanupPadInst).restype, LLVMIsACleanupPadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAUnaryInstruction:=dll.LLVMIsAUnaryInstruction).restype, LLVMIsAUnaryInstruction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAAllocaInst:=dll.LLVMIsAAllocaInst).restype, LLVMIsAAllocaInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsACastInst:=dll.LLVMIsACastInst).restype, LLVMIsACastInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAAddrSpaceCastInst:=dll.LLVMIsAAddrSpaceCastInst).restype, LLVMIsAAddrSpaceCastInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsABitCastInst:=dll.LLVMIsABitCastInst).restype, LLVMIsABitCastInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAFPExtInst:=dll.LLVMIsAFPExtInst).restype, LLVMIsAFPExtInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAFPToSIInst:=dll.LLVMIsAFPToSIInst).restype, LLVMIsAFPToSIInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAFPToUIInst:=dll.LLVMIsAFPToUIInst).restype, LLVMIsAFPToUIInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAFPTruncInst:=dll.LLVMIsAFPTruncInst).restype, LLVMIsAFPTruncInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAIntToPtrInst:=dll.LLVMIsAIntToPtrInst).restype, LLVMIsAIntToPtrInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAPtrToIntInst:=dll.LLVMIsAPtrToIntInst).restype, LLVMIsAPtrToIntInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsASExtInst:=dll.LLVMIsASExtInst).restype, LLVMIsASExtInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsASIToFPInst:=dll.LLVMIsASIToFPInst).restype, LLVMIsASIToFPInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsATruncInst:=dll.LLVMIsATruncInst).restype, LLVMIsATruncInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAUIToFPInst:=dll.LLVMIsAUIToFPInst).restype, LLVMIsAUIToFPInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAZExtInst:=dll.LLVMIsAZExtInst).restype, LLVMIsAZExtInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAExtractValueInst:=dll.LLVMIsAExtractValueInst).restype, LLVMIsAExtractValueInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsALoadInst:=dll.LLVMIsALoadInst).restype, LLVMIsALoadInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAVAArgInst:=dll.LLVMIsAVAArgInst).restype, LLVMIsAVAArgInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAFreezeInst:=dll.LLVMIsAFreezeInst).restype, LLVMIsAFreezeInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAAtomicCmpXchgInst:=dll.LLVMIsAAtomicCmpXchgInst).restype, LLVMIsAAtomicCmpXchgInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAAtomicRMWInst:=dll.LLVMIsAAtomicRMWInst).restype, LLVMIsAAtomicRMWInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAFenceInst:=dll.LLVMIsAFenceInst).restype, LLVMIsAFenceInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAMDNode:=dll.LLVMIsAMDNode).restype, LLVMIsAMDNode.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAValueAsMetadata:=dll.LLVMIsAValueAsMetadata).restype, LLVMIsAValueAsMetadata.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsAMDString:=dll.LLVMIsAMDString).restype, LLVMIsAMDString.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetValueName:=dll.LLVMGetValueName).restype, LLVMGetValueName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetValueName:=dll.LLVMSetValueName).restype, LLVMSetValueName.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOpaqueUse(Struct): pass
LLVMUseRef = ctypes.POINTER(struct_LLVMOpaqueUse)
try: (LLVMGetFirstUse:=dll.LLVMGetFirstUse).restype, LLVMGetFirstUse.argtypes = LLVMUseRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetNextUse:=dll.LLVMGetNextUse).restype, LLVMGetNextUse.argtypes = LLVMUseRef, [LLVMUseRef]
except AttributeError: pass

try: (LLVMGetUser:=dll.LLVMGetUser).restype, LLVMGetUser.argtypes = LLVMValueRef, [LLVMUseRef]
except AttributeError: pass

try: (LLVMGetUsedValue:=dll.LLVMGetUsedValue).restype, LLVMGetUsedValue.argtypes = LLVMValueRef, [LLVMUseRef]
except AttributeError: pass

try: (LLVMGetOperand:=dll.LLVMGetOperand).restype, LLVMGetOperand.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetOperandUse:=dll.LLVMGetOperandUse).restype, LLVMGetOperandUse.argtypes = LLVMUseRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSetOperand:=dll.LLVMSetOperand).restype, LLVMSetOperand.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetNumOperands:=dll.LLVMGetNumOperands).restype, LLVMGetNumOperands.argtypes = ctypes.c_int32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNull:=dll.LLVMConstNull).restype, LLVMConstNull.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstAllOnes:=dll.LLVMConstAllOnes).restype, LLVMConstAllOnes.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetUndef:=dll.LLVMGetUndef).restype, LLVMGetUndef.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMGetPoison:=dll.LLVMGetPoison).restype, LLVMGetPoison.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMIsNull:=dll.LLVMIsNull).restype, LLVMIsNull.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMConstPointerNull:=dll.LLVMConstPointerNull).restype, LLVMConstPointerNull.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstInt:=dll.LLVMConstInt).restype, LLVMConstInt.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.c_uint64, LLVMBool]
except AttributeError: pass

try: (LLVMConstIntOfArbitraryPrecision:=dll.LLVMConstIntOfArbitraryPrecision).restype, LLVMConstIntOfArbitraryPrecision.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.c_uint32, (uint64_t * 0)]
except AttributeError: pass

uint8_t = ctypes.c_ubyte
try: (LLVMConstIntOfString:=dll.LLVMConstIntOfString).restype, LLVMConstIntOfString.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), uint8_t]
except AttributeError: pass

try: (LLVMConstIntOfStringAndSize:=dll.LLVMConstIntOfStringAndSize).restype, LLVMConstIntOfStringAndSize.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, uint8_t]
except AttributeError: pass

try: (LLVMConstReal:=dll.LLVMConstReal).restype, LLVMConstReal.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.c_double]
except AttributeError: pass

try: (LLVMConstRealOfString:=dll.LLVMConstRealOfString).restype, LLVMConstRealOfString.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMConstRealOfStringAndSize:=dll.LLVMConstRealOfStringAndSize).restype, LLVMConstRealOfStringAndSize.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMConstIntGetZExtValue:=dll.LLVMConstIntGetZExtValue).restype, LLVMConstIntGetZExtValue.argtypes = ctypes.c_uint64, [LLVMValueRef]
except AttributeError: pass

try: (LLVMConstIntGetSExtValue:=dll.LLVMConstIntGetSExtValue).restype, LLVMConstIntGetSExtValue.argtypes = ctypes.c_int64, [LLVMValueRef]
except AttributeError: pass

try: (LLVMConstRealGetDouble:=dll.LLVMConstRealGetDouble).restype, LLVMConstRealGetDouble.argtypes = ctypes.c_double, [LLVMValueRef, ctypes.POINTER(LLVMBool)]
except AttributeError: pass

try: (LLVMConstStringInContext:=dll.LLVMConstStringInContext).restype, LLVMConstStringInContext.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

try: (LLVMConstStringInContext2:=dll.LLVMConstStringInContext2).restype, LLVMConstStringInContext2.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool]
except AttributeError: pass

try: (LLVMConstString:=dll.LLVMConstString).restype, LLVMConstString.argtypes = LLVMValueRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

try: (LLVMIsConstantString:=dll.LLVMIsConstantString).restype, LLVMIsConstantString.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetAsString:=dll.LLVMGetAsString).restype, LLVMGetAsString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMConstStructInContext:=dll.LLVMConstStructInContext).restype, LLVMConstStructInContext.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

try: (LLVMConstStruct:=dll.LLVMConstStruct).restype, LLVMConstStruct.argtypes = LLVMValueRef, [ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMBool]
except AttributeError: pass

try: (LLVMConstArray:=dll.LLVMConstArray).restype, LLVMConstArray.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMConstArray2:=dll.LLVMConstArray2).restype, LLVMConstArray2.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(LLVMValueRef), uint64_t]
except AttributeError: pass

try: (LLVMConstNamedStruct:=dll.LLVMConstNamedStruct).restype, LLVMConstNamedStruct.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetAggregateElement:=dll.LLVMGetAggregateElement).restype, LLVMGetAggregateElement.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetElementAsConstant:=dll.LLVMGetElementAsConstant).restype, LLVMGetElementAsConstant.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMConstVector:=dll.LLVMConstVector).restype, LLVMConstVector.argtypes = LLVMValueRef, [ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMConstantPtrAuth:=dll.LLVMConstantPtrAuth).restype, LLVMConstantPtrAuth.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetConstOpcode:=dll.LLVMGetConstOpcode).restype, LLVMGetConstOpcode.argtypes = LLVMOpcode, [LLVMValueRef]
except AttributeError: pass

try: (LLVMAlignOf:=dll.LLVMAlignOf).restype, LLVMAlignOf.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMSizeOf:=dll.LLVMSizeOf).restype, LLVMSizeOf.argtypes = LLVMValueRef, [LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstNeg:=dll.LLVMConstNeg).restype, LLVMConstNeg.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNSWNeg:=dll.LLVMConstNSWNeg).restype, LLVMConstNSWNeg.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNUWNeg:=dll.LLVMConstNUWNeg).restype, LLVMConstNUWNeg.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNot:=dll.LLVMConstNot).restype, LLVMConstNot.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMConstAdd:=dll.LLVMConstAdd).restype, LLVMConstAdd.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNSWAdd:=dll.LLVMConstNSWAdd).restype, LLVMConstNSWAdd.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNUWAdd:=dll.LLVMConstNUWAdd).restype, LLVMConstNUWAdd.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstSub:=dll.LLVMConstSub).restype, LLVMConstSub.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNSWSub:=dll.LLVMConstNSWSub).restype, LLVMConstNSWSub.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNUWSub:=dll.LLVMConstNUWSub).restype, LLVMConstNUWSub.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstMul:=dll.LLVMConstMul).restype, LLVMConstMul.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNSWMul:=dll.LLVMConstNSWMul).restype, LLVMConstNSWMul.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstNUWMul:=dll.LLVMConstNUWMul).restype, LLVMConstNUWMul.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstXor:=dll.LLVMConstXor).restype, LLVMConstXor.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstGEP2:=dll.LLVMConstGEP2).restype, LLVMConstGEP2.argtypes = LLVMValueRef, [LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMConstInBoundsGEP2:=dll.LLVMConstInBoundsGEP2).restype, LLVMConstInBoundsGEP2.argtypes = LLVMValueRef, [LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMConstGEPWithNoWrapFlags:=dll.LLVMConstGEPWithNoWrapFlags).restype, LLVMConstGEPWithNoWrapFlags.argtypes = LLVMValueRef, [LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMGEPNoWrapFlags]
except AttributeError: pass

try: (LLVMConstTrunc:=dll.LLVMConstTrunc).restype, LLVMConstTrunc.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstPtrToInt:=dll.LLVMConstPtrToInt).restype, LLVMConstPtrToInt.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstIntToPtr:=dll.LLVMConstIntToPtr).restype, LLVMConstIntToPtr.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstBitCast:=dll.LLVMConstBitCast).restype, LLVMConstBitCast.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstAddrSpaceCast:=dll.LLVMConstAddrSpaceCast).restype, LLVMConstAddrSpaceCast.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstTruncOrBitCast:=dll.LLVMConstTruncOrBitCast).restype, LLVMConstTruncOrBitCast.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstPointerCast:=dll.LLVMConstPointerCast).restype, LLVMConstPointerCast.argtypes = LLVMValueRef, [LLVMValueRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMConstExtractElement:=dll.LLVMConstExtractElement).restype, LLVMConstExtractElement.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstInsertElement:=dll.LLVMConstInsertElement).restype, LLVMConstInsertElement.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMConstShuffleVector:=dll.LLVMConstShuffleVector).restype, LLVMConstShuffleVector.argtypes = LLVMValueRef, [LLVMValueRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

class struct_LLVMOpaqueBasicBlock(Struct): pass
LLVMBasicBlockRef = ctypes.POINTER(struct_LLVMOpaqueBasicBlock)
try: (LLVMBlockAddress:=dll.LLVMBlockAddress).restype, LLVMBlockAddress.argtypes = LLVMValueRef, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetBlockAddressFunction:=dll.LLVMGetBlockAddressFunction).restype, LLVMGetBlockAddressFunction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetBlockAddressBasicBlock:=dll.LLVMGetBlockAddressBasicBlock).restype, LLVMGetBlockAddressBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMConstInlineAsm:=dll.LLVMConstInlineAsm).restype, LLVMConstInlineAsm.argtypes = LLVMValueRef, [LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMBool, LLVMBool]
except AttributeError: pass

try: (LLVMGetGlobalParent:=dll.LLVMGetGlobalParent).restype, LLVMGetGlobalParent.argtypes = LLVMModuleRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsDeclaration:=dll.LLVMIsDeclaration).restype, LLVMIsDeclaration.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetLinkage:=dll.LLVMGetLinkage).restype, LLVMGetLinkage.argtypes = LLVMLinkage, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetLinkage:=dll.LLVMSetLinkage).restype, LLVMSetLinkage.argtypes = None, [LLVMValueRef, LLVMLinkage]
except AttributeError: pass

try: (LLVMGetSection:=dll.LLVMGetSection).restype, LLVMGetSection.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetSection:=dll.LLVMSetSection).restype, LLVMSetSection.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetVisibility:=dll.LLVMGetVisibility).restype, LLVMGetVisibility.argtypes = LLVMVisibility, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetVisibility:=dll.LLVMSetVisibility).restype, LLVMSetVisibility.argtypes = None, [LLVMValueRef, LLVMVisibility]
except AttributeError: pass

try: (LLVMGetDLLStorageClass:=dll.LLVMGetDLLStorageClass).restype, LLVMGetDLLStorageClass.argtypes = LLVMDLLStorageClass, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetDLLStorageClass:=dll.LLVMSetDLLStorageClass).restype, LLVMSetDLLStorageClass.argtypes = None, [LLVMValueRef, LLVMDLLStorageClass]
except AttributeError: pass

try: (LLVMGetUnnamedAddress:=dll.LLVMGetUnnamedAddress).restype, LLVMGetUnnamedAddress.argtypes = LLVMUnnamedAddr, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetUnnamedAddress:=dll.LLVMSetUnnamedAddress).restype, LLVMSetUnnamedAddress.argtypes = None, [LLVMValueRef, LLVMUnnamedAddr]
except AttributeError: pass

try: (LLVMGlobalGetValueType:=dll.LLVMGlobalGetValueType).restype, LLVMGlobalGetValueType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMHasUnnamedAddr:=dll.LLVMHasUnnamedAddr).restype, LLVMHasUnnamedAddr.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetUnnamedAddr:=dll.LLVMSetUnnamedAddr).restype, LLVMSetUnnamedAddr.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetAlignment:=dll.LLVMGetAlignment).restype, LLVMGetAlignment.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetAlignment:=dll.LLVMSetAlignment).restype, LLVMSetAlignment.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGlobalSetMetadata:=dll.LLVMGlobalSetMetadata).restype, LLVMGlobalSetMetadata.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMGlobalEraseMetadata:=dll.LLVMGlobalEraseMetadata).restype, LLVMGlobalEraseMetadata.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGlobalClearMetadata:=dll.LLVMGlobalClearMetadata).restype, LLVMGlobalClearMetadata.argtypes = None, [LLVMValueRef]
except AttributeError: pass

class struct_LLVMOpaqueValueMetadataEntry(Struct): pass
LLVMValueMetadataEntry = struct_LLVMOpaqueValueMetadataEntry
try: (LLVMGlobalCopyAllMetadata:=dll.LLVMGlobalCopyAllMetadata).restype, LLVMGlobalCopyAllMetadata.argtypes = ctypes.POINTER(LLVMValueMetadataEntry), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMDisposeValueMetadataEntries:=dll.LLVMDisposeValueMetadataEntries).restype, LLVMDisposeValueMetadataEntries.argtypes = None, [ctypes.POINTER(LLVMValueMetadataEntry)]
except AttributeError: pass

try: (LLVMValueMetadataEntriesGetKind:=dll.LLVMValueMetadataEntriesGetKind).restype, LLVMValueMetadataEntriesGetKind.argtypes = ctypes.c_uint32, [ctypes.POINTER(LLVMValueMetadataEntry), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMValueMetadataEntriesGetMetadata:=dll.LLVMValueMetadataEntriesGetMetadata).restype, LLVMValueMetadataEntriesGetMetadata.argtypes = LLVMMetadataRef, [ctypes.POINTER(LLVMValueMetadataEntry), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMAddGlobal:=dll.LLVMAddGlobal).restype, LLVMAddGlobal.argtypes = LLVMValueRef, [LLVMModuleRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMAddGlobalInAddressSpace:=dll.LLVMAddGlobalInAddressSpace).restype, LLVMAddGlobalInAddressSpace.argtypes = LLVMValueRef, [LLVMModuleRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetNamedGlobal:=dll.LLVMGetNamedGlobal).restype, LLVMGetNamedGlobal.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetNamedGlobalWithLength:=dll.LLVMGetNamedGlobalWithLength).restype, LLVMGetNamedGlobalWithLength.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetFirstGlobal:=dll.LLVMGetFirstGlobal).restype, LLVMGetFirstGlobal.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetLastGlobal:=dll.LLVMGetLastGlobal).restype, LLVMGetLastGlobal.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetNextGlobal:=dll.LLVMGetNextGlobal).restype, LLVMGetNextGlobal.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetPreviousGlobal:=dll.LLVMGetPreviousGlobal).restype, LLVMGetPreviousGlobal.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMDeleteGlobal:=dll.LLVMDeleteGlobal).restype, LLVMDeleteGlobal.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetInitializer:=dll.LLVMGetInitializer).restype, LLVMGetInitializer.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetInitializer:=dll.LLVMSetInitializer).restype, LLVMSetInitializer.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMIsThreadLocal:=dll.LLVMIsThreadLocal).restype, LLVMIsThreadLocal.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetThreadLocal:=dll.LLVMSetThreadLocal).restype, LLVMSetThreadLocal.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMIsGlobalConstant:=dll.LLVMIsGlobalConstant).restype, LLVMIsGlobalConstant.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetGlobalConstant:=dll.LLVMSetGlobalConstant).restype, LLVMSetGlobalConstant.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetThreadLocalMode:=dll.LLVMGetThreadLocalMode).restype, LLVMGetThreadLocalMode.argtypes = LLVMThreadLocalMode, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetThreadLocalMode:=dll.LLVMSetThreadLocalMode).restype, LLVMSetThreadLocalMode.argtypes = None, [LLVMValueRef, LLVMThreadLocalMode]
except AttributeError: pass

try: (LLVMIsExternallyInitialized:=dll.LLVMIsExternallyInitialized).restype, LLVMIsExternallyInitialized.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetExternallyInitialized:=dll.LLVMSetExternallyInitialized).restype, LLVMSetExternallyInitialized.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMAddAlias2:=dll.LLVMAddAlias2).restype, LLVMAddAlias2.argtypes = LLVMValueRef, [LLVMModuleRef, LLVMTypeRef, ctypes.c_uint32, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetNamedGlobalAlias:=dll.LLVMGetNamedGlobalAlias).restype, LLVMGetNamedGlobalAlias.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetFirstGlobalAlias:=dll.LLVMGetFirstGlobalAlias).restype, LLVMGetFirstGlobalAlias.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetLastGlobalAlias:=dll.LLVMGetLastGlobalAlias).restype, LLVMGetLastGlobalAlias.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetNextGlobalAlias:=dll.LLVMGetNextGlobalAlias).restype, LLVMGetNextGlobalAlias.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetPreviousGlobalAlias:=dll.LLVMGetPreviousGlobalAlias).restype, LLVMGetPreviousGlobalAlias.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMAliasGetAliasee:=dll.LLVMAliasGetAliasee).restype, LLVMAliasGetAliasee.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMAliasSetAliasee:=dll.LLVMAliasSetAliasee).restype, LLVMAliasSetAliasee.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMDeleteFunction:=dll.LLVMDeleteFunction).restype, LLVMDeleteFunction.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (LLVMHasPersonalityFn:=dll.LLVMHasPersonalityFn).restype, LLVMHasPersonalityFn.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetPersonalityFn:=dll.LLVMGetPersonalityFn).restype, LLVMGetPersonalityFn.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetPersonalityFn:=dll.LLVMSetPersonalityFn).restype, LLVMSetPersonalityFn.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMLookupIntrinsicID:=dll.LLVMLookupIntrinsicID).restype, LLVMLookupIntrinsicID.argtypes = ctypes.c_uint32, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetIntrinsicID:=dll.LLVMGetIntrinsicID).restype, LLVMGetIntrinsicID.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetIntrinsicDeclaration:=dll.LLVMGetIntrinsicDeclaration).restype, LLVMGetIntrinsicDeclaration.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(LLVMTypeRef), size_t]
except AttributeError: pass

try: (LLVMIntrinsicGetType:=dll.LLVMIntrinsicGetType).restype, LLVMIntrinsicGetType.argtypes = LLVMTypeRef, [LLVMContextRef, ctypes.c_uint32, ctypes.POINTER(LLVMTypeRef), size_t]
except AttributeError: pass

try: (LLVMIntrinsicGetName:=dll.LLVMIntrinsicGetName).restype, LLVMIntrinsicGetName.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_uint32, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMIntrinsicCopyOverloadedName:=dll.LLVMIntrinsicCopyOverloadedName).restype, LLVMIntrinsicCopyOverloadedName.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_uint32, ctypes.POINTER(LLVMTypeRef), size_t, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMIntrinsicCopyOverloadedName2:=dll.LLVMIntrinsicCopyOverloadedName2).restype, LLVMIntrinsicCopyOverloadedName2.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(LLVMTypeRef), size_t, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMIntrinsicIsOverloaded:=dll.LLVMIntrinsicIsOverloaded).restype, LLVMIntrinsicIsOverloaded.argtypes = LLVMBool, [ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetFunctionCallConv:=dll.LLVMGetFunctionCallConv).restype, LLVMGetFunctionCallConv.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetFunctionCallConv:=dll.LLVMSetFunctionCallConv).restype, LLVMSetFunctionCallConv.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetGC:=dll.LLVMGetGC).restype, LLVMGetGC.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetGC:=dll.LLVMSetGC).restype, LLVMSetGC.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetPrefixData:=dll.LLVMGetPrefixData).restype, LLVMGetPrefixData.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMHasPrefixData:=dll.LLVMHasPrefixData).restype, LLVMHasPrefixData.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetPrefixData:=dll.LLVMSetPrefixData).restype, LLVMSetPrefixData.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetPrologueData:=dll.LLVMGetPrologueData).restype, LLVMGetPrologueData.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMHasPrologueData:=dll.LLVMHasPrologueData).restype, LLVMHasPrologueData.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetPrologueData:=dll.LLVMSetPrologueData).restype, LLVMSetPrologueData.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMAddAttributeAtIndex:=dll.LLVMAddAttributeAtIndex).restype, LLVMAddAttributeAtIndex.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, LLVMAttributeRef]
except AttributeError: pass

try: (LLVMGetAttributeCountAtIndex:=dll.LLVMGetAttributeCountAtIndex).restype, LLVMGetAttributeCountAtIndex.argtypes = ctypes.c_uint32, [LLVMValueRef, LLVMAttributeIndex]
except AttributeError: pass

try: (LLVMGetAttributesAtIndex:=dll.LLVMGetAttributesAtIndex).restype, LLVMGetAttributesAtIndex.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(LLVMAttributeRef)]
except AttributeError: pass

try: (LLVMGetEnumAttributeAtIndex:=dll.LLVMGetEnumAttributeAtIndex).restype, LLVMGetEnumAttributeAtIndex.argtypes = LLVMAttributeRef, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetStringAttributeAtIndex:=dll.LLVMGetStringAttributeAtIndex).restype, LLVMGetStringAttributeAtIndex.argtypes = LLVMAttributeRef, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMRemoveEnumAttributeAtIndex:=dll.LLVMRemoveEnumAttributeAtIndex).restype, LLVMRemoveEnumAttributeAtIndex.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMRemoveStringAttributeAtIndex:=dll.LLVMRemoveStringAttributeAtIndex).restype, LLVMRemoveStringAttributeAtIndex.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMAddTargetDependentFunctionAttr:=dll.LLVMAddTargetDependentFunctionAttr).restype, LLVMAddTargetDependentFunctionAttr.argtypes = None, [LLVMValueRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMCountParams:=dll.LLVMCountParams).restype, LLVMCountParams.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetParams:=dll.LLVMGetParams).restype, LLVMGetParams.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

try: (LLVMGetParam:=dll.LLVMGetParam).restype, LLVMGetParam.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetParamParent:=dll.LLVMGetParamParent).restype, LLVMGetParamParent.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetFirstParam:=dll.LLVMGetFirstParam).restype, LLVMGetFirstParam.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetLastParam:=dll.LLVMGetLastParam).restype, LLVMGetLastParam.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetNextParam:=dll.LLVMGetNextParam).restype, LLVMGetNextParam.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetPreviousParam:=dll.LLVMGetPreviousParam).restype, LLVMGetPreviousParam.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetParamAlignment:=dll.LLVMSetParamAlignment).restype, LLVMSetParamAlignment.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMAddGlobalIFunc:=dll.LLVMAddGlobalIFunc).restype, LLVMAddGlobalIFunc.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMTypeRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetNamedGlobalIFunc:=dll.LLVMGetNamedGlobalIFunc).restype, LLVMGetNamedGlobalIFunc.argtypes = LLVMValueRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMGetFirstGlobalIFunc:=dll.LLVMGetFirstGlobalIFunc).restype, LLVMGetFirstGlobalIFunc.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetLastGlobalIFunc:=dll.LLVMGetLastGlobalIFunc).restype, LLVMGetLastGlobalIFunc.argtypes = LLVMValueRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMGetNextGlobalIFunc:=dll.LLVMGetNextGlobalIFunc).restype, LLVMGetNextGlobalIFunc.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetPreviousGlobalIFunc:=dll.LLVMGetPreviousGlobalIFunc).restype, LLVMGetPreviousGlobalIFunc.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetGlobalIFuncResolver:=dll.LLVMGetGlobalIFuncResolver).restype, LLVMGetGlobalIFuncResolver.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetGlobalIFuncResolver:=dll.LLVMSetGlobalIFuncResolver).restype, LLVMSetGlobalIFuncResolver.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMEraseGlobalIFunc:=dll.LLVMEraseGlobalIFunc).restype, LLVMEraseGlobalIFunc.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (LLVMRemoveGlobalIFunc:=dll.LLVMRemoveGlobalIFunc).restype, LLVMRemoveGlobalIFunc.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (LLVMMDStringInContext2:=dll.LLVMMDStringInContext2).restype, LLVMMDStringInContext2.argtypes = LLVMMetadataRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMMDNodeInContext2:=dll.LLVMMDNodeInContext2).restype, LLVMMDNodeInContext2.argtypes = LLVMMetadataRef, [LLVMContextRef, ctypes.POINTER(LLVMMetadataRef), size_t]
except AttributeError: pass

try: (LLVMMetadataAsValue:=dll.LLVMMetadataAsValue).restype, LLVMMetadataAsValue.argtypes = LLVMValueRef, [LLVMContextRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMValueAsMetadata:=dll.LLVMValueAsMetadata).restype, LLVMValueAsMetadata.argtypes = LLVMMetadataRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetMDString:=dll.LLVMGetMDString).restype, LLVMGetMDString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMValueRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (LLVMGetMDNodeNumOperands:=dll.LLVMGetMDNodeNumOperands).restype, LLVMGetMDNodeNumOperands.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetMDNodeOperands:=dll.LLVMGetMDNodeOperands).restype, LLVMGetMDNodeOperands.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

try: (LLVMReplaceMDNodeOperandWith:=dll.LLVMReplaceMDNodeOperandWith).restype, LLVMReplaceMDNodeOperandWith.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMMDStringInContext:=dll.LLVMMDStringInContext).restype, LLVMMDStringInContext.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMMDString:=dll.LLVMMDString).restype, LLVMMDString.argtypes = LLVMValueRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMMDNodeInContext:=dll.LLVMMDNodeInContext).restype, LLVMMDNodeInContext.argtypes = LLVMValueRef, [LLVMContextRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMMDNode:=dll.LLVMMDNode).restype, LLVMMDNode.argtypes = LLVMValueRef, [ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

class struct_LLVMOpaqueOperandBundle(Struct): pass
LLVMOperandBundleRef = ctypes.POINTER(struct_LLVMOpaqueOperandBundle)
try: (LLVMCreateOperandBundle:=dll.LLVMCreateOperandBundle).restype, LLVMCreateOperandBundle.argtypes = LLVMOperandBundleRef, [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMDisposeOperandBundle:=dll.LLVMDisposeOperandBundle).restype, LLVMDisposeOperandBundle.argtypes = None, [LLVMOperandBundleRef]
except AttributeError: pass

try: (LLVMGetOperandBundleTag:=dll.LLVMGetOperandBundleTag).restype, LLVMGetOperandBundleTag.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOperandBundleRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMGetNumOperandBundleArgs:=dll.LLVMGetNumOperandBundleArgs).restype, LLVMGetNumOperandBundleArgs.argtypes = ctypes.c_uint32, [LLVMOperandBundleRef]
except AttributeError: pass

try: (LLVMGetOperandBundleArgAtIndex:=dll.LLVMGetOperandBundleArgAtIndex).restype, LLVMGetOperandBundleArgAtIndex.argtypes = LLVMValueRef, [LLVMOperandBundleRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMBasicBlockAsValue:=dll.LLVMBasicBlockAsValue).restype, LLVMBasicBlockAsValue.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMValueIsBasicBlock:=dll.LLVMValueIsBasicBlock).restype, LLVMValueIsBasicBlock.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMValueAsBasicBlock:=dll.LLVMValueAsBasicBlock).restype, LLVMValueAsBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetBasicBlockName:=dll.LLVMGetBasicBlockName).restype, LLVMGetBasicBlockName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetBasicBlockParent:=dll.LLVMGetBasicBlockParent).restype, LLVMGetBasicBlockParent.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetBasicBlockTerminator:=dll.LLVMGetBasicBlockTerminator).restype, LLVMGetBasicBlockTerminator.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMCountBasicBlocks:=dll.LLVMCountBasicBlocks).restype, LLVMCountBasicBlocks.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetBasicBlocks:=dll.LLVMGetBasicBlocks).restype, LLVMGetBasicBlocks.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMBasicBlockRef)]
except AttributeError: pass

try: (LLVMGetFirstBasicBlock:=dll.LLVMGetFirstBasicBlock).restype, LLVMGetFirstBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetLastBasicBlock:=dll.LLVMGetLastBasicBlock).restype, LLVMGetLastBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetNextBasicBlock:=dll.LLVMGetNextBasicBlock).restype, LLVMGetNextBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetPreviousBasicBlock:=dll.LLVMGetPreviousBasicBlock).restype, LLVMGetPreviousBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetEntryBasicBlock:=dll.LLVMGetEntryBasicBlock).restype, LLVMGetEntryBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

class struct_LLVMOpaqueBuilder(Struct): pass
LLVMBuilderRef = ctypes.POINTER(struct_LLVMOpaqueBuilder)
try: (LLVMInsertExistingBasicBlockAfterInsertBlock:=dll.LLVMInsertExistingBasicBlockAfterInsertBlock).restype, LLVMInsertExistingBasicBlockAfterInsertBlock.argtypes = None, [LLVMBuilderRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMAppendExistingBasicBlock:=dll.LLVMAppendExistingBasicBlock).restype, LLVMAppendExistingBasicBlock.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMCreateBasicBlockInContext:=dll.LLVMCreateBasicBlockInContext).restype, LLVMCreateBasicBlockInContext.argtypes = LLVMBasicBlockRef, [LLVMContextRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMAppendBasicBlockInContext:=dll.LLVMAppendBasicBlockInContext).restype, LLVMAppendBasicBlockInContext.argtypes = LLVMBasicBlockRef, [LLVMContextRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMAppendBasicBlock:=dll.LLVMAppendBasicBlock).restype, LLVMAppendBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMInsertBasicBlockInContext:=dll.LLVMInsertBasicBlockInContext).restype, LLVMInsertBasicBlockInContext.argtypes = LLVMBasicBlockRef, [LLVMContextRef, LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMInsertBasicBlock:=dll.LLVMInsertBasicBlock).restype, LLVMInsertBasicBlock.argtypes = LLVMBasicBlockRef, [LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDeleteBasicBlock:=dll.LLVMDeleteBasicBlock).restype, LLVMDeleteBasicBlock.argtypes = None, [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMRemoveBasicBlockFromParent:=dll.LLVMRemoveBasicBlockFromParent).restype, LLVMRemoveBasicBlockFromParent.argtypes = None, [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMMoveBasicBlockBefore:=dll.LLVMMoveBasicBlockBefore).restype, LLVMMoveBasicBlockBefore.argtypes = None, [LLVMBasicBlockRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMMoveBasicBlockAfter:=dll.LLVMMoveBasicBlockAfter).restype, LLVMMoveBasicBlockAfter.argtypes = None, [LLVMBasicBlockRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetFirstInstruction:=dll.LLVMGetFirstInstruction).restype, LLVMGetFirstInstruction.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetLastInstruction:=dll.LLVMGetLastInstruction).restype, LLVMGetLastInstruction.argtypes = LLVMValueRef, [LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMHasMetadata:=dll.LLVMHasMetadata).restype, LLVMHasMetadata.argtypes = ctypes.c_int32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetMetadata:=dll.LLVMGetMetadata).restype, LLVMGetMetadata.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSetMetadata:=dll.LLVMSetMetadata).restype, LLVMSetMetadata.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

try: (LLVMInstructionGetAllMetadataOtherThanDebugLoc:=dll.LLVMInstructionGetAllMetadataOtherThanDebugLoc).restype, LLVMInstructionGetAllMetadataOtherThanDebugLoc.argtypes = ctypes.POINTER(LLVMValueMetadataEntry), [LLVMValueRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMGetInstructionParent:=dll.LLVMGetInstructionParent).restype, LLVMGetInstructionParent.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetNextInstruction:=dll.LLVMGetNextInstruction).restype, LLVMGetNextInstruction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetPreviousInstruction:=dll.LLVMGetPreviousInstruction).restype, LLVMGetPreviousInstruction.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMInstructionRemoveFromParent:=dll.LLVMInstructionRemoveFromParent).restype, LLVMInstructionRemoveFromParent.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (LLVMInstructionEraseFromParent:=dll.LLVMInstructionEraseFromParent).restype, LLVMInstructionEraseFromParent.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (LLVMDeleteInstruction:=dll.LLVMDeleteInstruction).restype, LLVMDeleteInstruction.argtypes = None, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetInstructionOpcode:=dll.LLVMGetInstructionOpcode).restype, LLVMGetInstructionOpcode.argtypes = LLVMOpcode, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetICmpPredicate:=dll.LLVMGetICmpPredicate).restype, LLVMGetICmpPredicate.argtypes = LLVMIntPredicate, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetFCmpPredicate:=dll.LLVMGetFCmpPredicate).restype, LLVMGetFCmpPredicate.argtypes = LLVMRealPredicate, [LLVMValueRef]
except AttributeError: pass

try: (LLVMInstructionClone:=dll.LLVMInstructionClone).restype, LLVMInstructionClone.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsATerminatorInst:=dll.LLVMIsATerminatorInst).restype, LLVMIsATerminatorInst.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetFirstDbgRecord:=dll.LLVMGetFirstDbgRecord).restype, LLVMGetFirstDbgRecord.argtypes = LLVMDbgRecordRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetLastDbgRecord:=dll.LLVMGetLastDbgRecord).restype, LLVMGetLastDbgRecord.argtypes = LLVMDbgRecordRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetNextDbgRecord:=dll.LLVMGetNextDbgRecord).restype, LLVMGetNextDbgRecord.argtypes = LLVMDbgRecordRef, [LLVMDbgRecordRef]
except AttributeError: pass

try: (LLVMGetPreviousDbgRecord:=dll.LLVMGetPreviousDbgRecord).restype, LLVMGetPreviousDbgRecord.argtypes = LLVMDbgRecordRef, [LLVMDbgRecordRef]
except AttributeError: pass

try: (LLVMGetNumArgOperands:=dll.LLVMGetNumArgOperands).restype, LLVMGetNumArgOperands.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetInstructionCallConv:=dll.LLVMSetInstructionCallConv).restype, LLVMSetInstructionCallConv.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetInstructionCallConv:=dll.LLVMGetInstructionCallConv).restype, LLVMGetInstructionCallConv.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetInstrParamAlignment:=dll.LLVMSetInstrParamAlignment).restype, LLVMSetInstrParamAlignment.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMAddCallSiteAttribute:=dll.LLVMAddCallSiteAttribute).restype, LLVMAddCallSiteAttribute.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, LLVMAttributeRef]
except AttributeError: pass

try: (LLVMGetCallSiteAttributeCount:=dll.LLVMGetCallSiteAttributeCount).restype, LLVMGetCallSiteAttributeCount.argtypes = ctypes.c_uint32, [LLVMValueRef, LLVMAttributeIndex]
except AttributeError: pass

try: (LLVMGetCallSiteAttributes:=dll.LLVMGetCallSiteAttributes).restype, LLVMGetCallSiteAttributes.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(LLVMAttributeRef)]
except AttributeError: pass

try: (LLVMGetCallSiteEnumAttribute:=dll.LLVMGetCallSiteEnumAttribute).restype, LLVMGetCallSiteEnumAttribute.argtypes = LLVMAttributeRef, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetCallSiteStringAttribute:=dll.LLVMGetCallSiteStringAttribute).restype, LLVMGetCallSiteStringAttribute.argtypes = LLVMAttributeRef, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMRemoveCallSiteEnumAttribute:=dll.LLVMRemoveCallSiteEnumAttribute).restype, LLVMRemoveCallSiteEnumAttribute.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMRemoveCallSiteStringAttribute:=dll.LLVMRemoveCallSiteStringAttribute).restype, LLVMRemoveCallSiteStringAttribute.argtypes = None, [LLVMValueRef, LLVMAttributeIndex, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetCalledFunctionType:=dll.LLVMGetCalledFunctionType).restype, LLVMGetCalledFunctionType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetCalledValue:=dll.LLVMGetCalledValue).restype, LLVMGetCalledValue.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetNumOperandBundles:=dll.LLVMGetNumOperandBundles).restype, LLVMGetNumOperandBundles.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetOperandBundleAtIndex:=dll.LLVMGetOperandBundleAtIndex).restype, LLVMGetOperandBundleAtIndex.argtypes = LLVMOperandBundleRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIsTailCall:=dll.LLVMIsTailCall).restype, LLVMIsTailCall.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetTailCall:=dll.LLVMSetTailCall).restype, LLVMSetTailCall.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetTailCallKind:=dll.LLVMGetTailCallKind).restype, LLVMGetTailCallKind.argtypes = LLVMTailCallKind, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetTailCallKind:=dll.LLVMSetTailCallKind).restype, LLVMSetTailCallKind.argtypes = None, [LLVMValueRef, LLVMTailCallKind]
except AttributeError: pass

try: (LLVMGetNormalDest:=dll.LLVMGetNormalDest).restype, LLVMGetNormalDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetUnwindDest:=dll.LLVMGetUnwindDest).restype, LLVMGetUnwindDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetNormalDest:=dll.LLVMSetNormalDest).restype, LLVMSetNormalDest.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMSetUnwindDest:=dll.LLVMSetUnwindDest).restype, LLVMSetUnwindDest.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetCallBrDefaultDest:=dll.LLVMGetCallBrDefaultDest).restype, LLVMGetCallBrDefaultDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetCallBrNumIndirectDests:=dll.LLVMGetCallBrNumIndirectDests).restype, LLVMGetCallBrNumIndirectDests.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetCallBrIndirectDest:=dll.LLVMGetCallBrIndirectDest).restype, LLVMGetCallBrIndirectDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetNumSuccessors:=dll.LLVMGetNumSuccessors).restype, LLVMGetNumSuccessors.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetSuccessor:=dll.LLVMGetSuccessor).restype, LLVMGetSuccessor.argtypes = LLVMBasicBlockRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSetSuccessor:=dll.LLVMSetSuccessor).restype, LLVMSetSuccessor.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMIsConditional:=dll.LLVMIsConditional).restype, LLVMIsConditional.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetCondition:=dll.LLVMGetCondition).restype, LLVMGetCondition.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetCondition:=dll.LLVMSetCondition).restype, LLVMSetCondition.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetSwitchDefaultDest:=dll.LLVMGetSwitchDefaultDest).restype, LLVMGetSwitchDefaultDest.argtypes = LLVMBasicBlockRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetAllocatedType:=dll.LLVMGetAllocatedType).restype, LLVMGetAllocatedType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMIsInBounds:=dll.LLVMIsInBounds).restype, LLVMIsInBounds.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetIsInBounds:=dll.LLVMSetIsInBounds).restype, LLVMSetIsInBounds.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetGEPSourceElementType:=dll.LLVMGetGEPSourceElementType).restype, LLVMGetGEPSourceElementType.argtypes = LLVMTypeRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGEPGetNoWrapFlags:=dll.LLVMGEPGetNoWrapFlags).restype, LLVMGEPGetNoWrapFlags.argtypes = LLVMGEPNoWrapFlags, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGEPSetNoWrapFlags:=dll.LLVMGEPSetNoWrapFlags).restype, LLVMGEPSetNoWrapFlags.argtypes = None, [LLVMValueRef, LLVMGEPNoWrapFlags]
except AttributeError: pass

try: (LLVMAddIncoming:=dll.LLVMAddIncoming).restype, LLVMAddIncoming.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.POINTER(LLVMBasicBlockRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMCountIncoming:=dll.LLVMCountIncoming).restype, LLVMCountIncoming.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetIncomingValue:=dll.LLVMGetIncomingValue).restype, LLVMGetIncomingValue.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetIncomingBlock:=dll.LLVMGetIncomingBlock).restype, LLVMGetIncomingBlock.argtypes = LLVMBasicBlockRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetNumIndices:=dll.LLVMGetNumIndices).restype, LLVMGetNumIndices.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetIndices:=dll.LLVMGetIndices).restype, LLVMGetIndices.argtypes = ctypes.POINTER(ctypes.c_uint32), [LLVMValueRef]
except AttributeError: pass

try: (LLVMCreateBuilderInContext:=dll.LLVMCreateBuilderInContext).restype, LLVMCreateBuilderInContext.argtypes = LLVMBuilderRef, [LLVMContextRef]
except AttributeError: pass

try: (LLVMCreateBuilder:=dll.LLVMCreateBuilder).restype, LLVMCreateBuilder.argtypes = LLVMBuilderRef, []
except AttributeError: pass

try: (LLVMPositionBuilder:=dll.LLVMPositionBuilder).restype, LLVMPositionBuilder.argtypes = None, [LLVMBuilderRef, LLVMBasicBlockRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMPositionBuilderBeforeDbgRecords:=dll.LLVMPositionBuilderBeforeDbgRecords).restype, LLVMPositionBuilderBeforeDbgRecords.argtypes = None, [LLVMBuilderRef, LLVMBasicBlockRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMPositionBuilderBefore:=dll.LLVMPositionBuilderBefore).restype, LLVMPositionBuilderBefore.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMPositionBuilderBeforeInstrAndDbgRecords:=dll.LLVMPositionBuilderBeforeInstrAndDbgRecords).restype, LLVMPositionBuilderBeforeInstrAndDbgRecords.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMPositionBuilderAtEnd:=dll.LLVMPositionBuilderAtEnd).restype, LLVMPositionBuilderAtEnd.argtypes = None, [LLVMBuilderRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetInsertBlock:=dll.LLVMGetInsertBlock).restype, LLVMGetInsertBlock.argtypes = LLVMBasicBlockRef, [LLVMBuilderRef]
except AttributeError: pass

try: (LLVMClearInsertionPosition:=dll.LLVMClearInsertionPosition).restype, LLVMClearInsertionPosition.argtypes = None, [LLVMBuilderRef]
except AttributeError: pass

try: (LLVMInsertIntoBuilder:=dll.LLVMInsertIntoBuilder).restype, LLVMInsertIntoBuilder.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMInsertIntoBuilderWithName:=dll.LLVMInsertIntoBuilderWithName).restype, LLVMInsertIntoBuilderWithName.argtypes = None, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeBuilder:=dll.LLVMDisposeBuilder).restype, LLVMDisposeBuilder.argtypes = None, [LLVMBuilderRef]
except AttributeError: pass

try: (LLVMGetCurrentDebugLocation2:=dll.LLVMGetCurrentDebugLocation2).restype, LLVMGetCurrentDebugLocation2.argtypes = LLVMMetadataRef, [LLVMBuilderRef]
except AttributeError: pass

try: (LLVMSetCurrentDebugLocation2:=dll.LLVMSetCurrentDebugLocation2).restype, LLVMSetCurrentDebugLocation2.argtypes = None, [LLVMBuilderRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMSetInstDebugLocation:=dll.LLVMSetInstDebugLocation).restype, LLVMSetInstDebugLocation.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMAddMetadataToInst:=dll.LLVMAddMetadataToInst).restype, LLVMAddMetadataToInst.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMBuilderGetDefaultFPMathTag:=dll.LLVMBuilderGetDefaultFPMathTag).restype, LLVMBuilderGetDefaultFPMathTag.argtypes = LLVMMetadataRef, [LLVMBuilderRef]
except AttributeError: pass

try: (LLVMBuilderSetDefaultFPMathTag:=dll.LLVMBuilderSetDefaultFPMathTag).restype, LLVMBuilderSetDefaultFPMathTag.argtypes = None, [LLVMBuilderRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMGetBuilderContext:=dll.LLVMGetBuilderContext).restype, LLVMGetBuilderContext.argtypes = LLVMContextRef, [LLVMBuilderRef]
except AttributeError: pass

try: (LLVMSetCurrentDebugLocation:=dll.LLVMSetCurrentDebugLocation).restype, LLVMSetCurrentDebugLocation.argtypes = None, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetCurrentDebugLocation:=dll.LLVMGetCurrentDebugLocation).restype, LLVMGetCurrentDebugLocation.argtypes = LLVMValueRef, [LLVMBuilderRef]
except AttributeError: pass

try: (LLVMBuildRetVoid:=dll.LLVMBuildRetVoid).restype, LLVMBuildRetVoid.argtypes = LLVMValueRef, [LLVMBuilderRef]
except AttributeError: pass

try: (LLVMBuildRet:=dll.LLVMBuildRet).restype, LLVMBuildRet.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMBuildAggregateRet:=dll.LLVMBuildAggregateRet).restype, LLVMBuildAggregateRet.argtypes = LLVMValueRef, [LLVMBuilderRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMBuildBr:=dll.LLVMBuildBr).restype, LLVMBuildBr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMBuildCondBr:=dll.LLVMBuildCondBr).restype, LLVMBuildCondBr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMBuildSwitch:=dll.LLVMBuildSwitch).restype, LLVMBuildSwitch.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMBuildIndirectBr:=dll.LLVMBuildIndirectBr).restype, LLVMBuildIndirectBr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMBuildCallBr:=dll.LLVMBuildCallBr).restype, LLVMBuildCallBr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMBasicBlockRef, ctypes.POINTER(LLVMBasicBlockRef), ctypes.c_uint32, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(LLVMOperandBundleRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildInvoke2:=dll.LLVMBuildInvoke2).restype, LLVMBuildInvoke2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMBasicBlockRef, LLVMBasicBlockRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildInvokeWithOperandBundles:=dll.LLVMBuildInvokeWithOperandBundles).restype, LLVMBuildInvokeWithOperandBundles.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, LLVMBasicBlockRef, LLVMBasicBlockRef, ctypes.POINTER(LLVMOperandBundleRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildUnreachable:=dll.LLVMBuildUnreachable).restype, LLVMBuildUnreachable.argtypes = LLVMValueRef, [LLVMBuilderRef]
except AttributeError: pass

try: (LLVMBuildResume:=dll.LLVMBuildResume).restype, LLVMBuildResume.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMBuildLandingPad:=dll.LLVMBuildLandingPad).restype, LLVMBuildLandingPad.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildCleanupRet:=dll.LLVMBuildCleanupRet).restype, LLVMBuildCleanupRet.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMBuildCatchRet:=dll.LLVMBuildCatchRet).restype, LLVMBuildCatchRet.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMBuildCatchPad:=dll.LLVMBuildCatchPad).restype, LLVMBuildCatchPad.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildCleanupPad:=dll.LLVMBuildCleanupPad).restype, LLVMBuildCleanupPad.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildCatchSwitch:=dll.LLVMBuildCatchSwitch).restype, LLVMBuildCatchSwitch.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMBasicBlockRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMAddCase:=dll.LLVMAddCase).restype, LLVMAddCase.argtypes = None, [LLVMValueRef, LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMAddDestination:=dll.LLVMAddDestination).restype, LLVMAddDestination.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetNumClauses:=dll.LLVMGetNumClauses).restype, LLVMGetNumClauses.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetClause:=dll.LLVMGetClause).restype, LLVMGetClause.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMAddClause:=dll.LLVMAddClause).restype, LLVMAddClause.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMIsCleanup:=dll.LLVMIsCleanup).restype, LLVMIsCleanup.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetCleanup:=dll.LLVMSetCleanup).restype, LLVMSetCleanup.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMAddHandler:=dll.LLVMAddHandler).restype, LLVMAddHandler.argtypes = None, [LLVMValueRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetNumHandlers:=dll.LLVMGetNumHandlers).restype, LLVMGetNumHandlers.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetHandlers:=dll.LLVMGetHandlers).restype, LLVMGetHandlers.argtypes = None, [LLVMValueRef, ctypes.POINTER(LLVMBasicBlockRef)]
except AttributeError: pass

try: (LLVMGetArgOperand:=dll.LLVMGetArgOperand).restype, LLVMGetArgOperand.argtypes = LLVMValueRef, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSetArgOperand:=dll.LLVMSetArgOperand).restype, LLVMSetArgOperand.argtypes = None, [LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetParentCatchSwitch:=dll.LLVMGetParentCatchSwitch).restype, LLVMGetParentCatchSwitch.argtypes = LLVMValueRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetParentCatchSwitch:=dll.LLVMSetParentCatchSwitch).restype, LLVMSetParentCatchSwitch.argtypes = None, [LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMBuildAdd:=dll.LLVMBuildAdd).restype, LLVMBuildAdd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNSWAdd:=dll.LLVMBuildNSWAdd).restype, LLVMBuildNSWAdd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNUWAdd:=dll.LLVMBuildNUWAdd).restype, LLVMBuildNUWAdd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFAdd:=dll.LLVMBuildFAdd).restype, LLVMBuildFAdd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildSub:=dll.LLVMBuildSub).restype, LLVMBuildSub.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNSWSub:=dll.LLVMBuildNSWSub).restype, LLVMBuildNSWSub.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNUWSub:=dll.LLVMBuildNUWSub).restype, LLVMBuildNUWSub.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFSub:=dll.LLVMBuildFSub).restype, LLVMBuildFSub.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildMul:=dll.LLVMBuildMul).restype, LLVMBuildMul.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNSWMul:=dll.LLVMBuildNSWMul).restype, LLVMBuildNSWMul.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNUWMul:=dll.LLVMBuildNUWMul).restype, LLVMBuildNUWMul.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFMul:=dll.LLVMBuildFMul).restype, LLVMBuildFMul.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildUDiv:=dll.LLVMBuildUDiv).restype, LLVMBuildUDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildExactUDiv:=dll.LLVMBuildExactUDiv).restype, LLVMBuildExactUDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildSDiv:=dll.LLVMBuildSDiv).restype, LLVMBuildSDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildExactSDiv:=dll.LLVMBuildExactSDiv).restype, LLVMBuildExactSDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFDiv:=dll.LLVMBuildFDiv).restype, LLVMBuildFDiv.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildURem:=dll.LLVMBuildURem).restype, LLVMBuildURem.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildSRem:=dll.LLVMBuildSRem).restype, LLVMBuildSRem.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFRem:=dll.LLVMBuildFRem).restype, LLVMBuildFRem.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildShl:=dll.LLVMBuildShl).restype, LLVMBuildShl.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildLShr:=dll.LLVMBuildLShr).restype, LLVMBuildLShr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildAShr:=dll.LLVMBuildAShr).restype, LLVMBuildAShr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildAnd:=dll.LLVMBuildAnd).restype, LLVMBuildAnd.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildOr:=dll.LLVMBuildOr).restype, LLVMBuildOr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildXor:=dll.LLVMBuildXor).restype, LLVMBuildXor.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildBinOp:=dll.LLVMBuildBinOp).restype, LLVMBuildBinOp.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMOpcode, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNeg:=dll.LLVMBuildNeg).restype, LLVMBuildNeg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNSWNeg:=dll.LLVMBuildNSWNeg).restype, LLVMBuildNSWNeg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNUWNeg:=dll.LLVMBuildNUWNeg).restype, LLVMBuildNUWNeg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFNeg:=dll.LLVMBuildFNeg).restype, LLVMBuildFNeg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildNot:=dll.LLVMBuildNot).restype, LLVMBuildNot.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetNUW:=dll.LLVMGetNUW).restype, LLVMGetNUW.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetNUW:=dll.LLVMSetNUW).restype, LLVMSetNUW.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetNSW:=dll.LLVMGetNSW).restype, LLVMGetNSW.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetNSW:=dll.LLVMSetNSW).restype, LLVMSetNSW.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetExact:=dll.LLVMGetExact).restype, LLVMGetExact.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetExact:=dll.LLVMSetExact).restype, LLVMSetExact.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetNNeg:=dll.LLVMGetNNeg).restype, LLVMGetNNeg.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetNNeg:=dll.LLVMSetNNeg).restype, LLVMSetNNeg.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetFastMathFlags:=dll.LLVMGetFastMathFlags).restype, LLVMGetFastMathFlags.argtypes = LLVMFastMathFlags, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetFastMathFlags:=dll.LLVMSetFastMathFlags).restype, LLVMSetFastMathFlags.argtypes = None, [LLVMValueRef, LLVMFastMathFlags]
except AttributeError: pass

try: (LLVMCanValueUseFastMathFlags:=dll.LLVMCanValueUseFastMathFlags).restype, LLVMCanValueUseFastMathFlags.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetIsDisjoint:=dll.LLVMGetIsDisjoint).restype, LLVMGetIsDisjoint.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetIsDisjoint:=dll.LLVMSetIsDisjoint).restype, LLVMSetIsDisjoint.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMBuildMalloc:=dll.LLVMBuildMalloc).restype, LLVMBuildMalloc.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildArrayMalloc:=dll.LLVMBuildArrayMalloc).restype, LLVMBuildArrayMalloc.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildMemSet:=dll.LLVMBuildMemSet).restype, LLVMBuildMemSet.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMBuildMemCpy:=dll.LLVMBuildMemCpy).restype, LLVMBuildMemCpy.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

try: (LLVMBuildMemMove:=dll.LLVMBuildMemMove).restype, LLVMBuildMemMove.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, LLVMValueRef, ctypes.c_uint32, LLVMValueRef]
except AttributeError: pass

try: (LLVMBuildAlloca:=dll.LLVMBuildAlloca).restype, LLVMBuildAlloca.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildArrayAlloca:=dll.LLVMBuildArrayAlloca).restype, LLVMBuildArrayAlloca.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFree:=dll.LLVMBuildFree).restype, LLVMBuildFree.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMBuildLoad2:=dll.LLVMBuildLoad2).restype, LLVMBuildLoad2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildStore:=dll.LLVMBuildStore).restype, LLVMBuildStore.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMBuildGEP2:=dll.LLVMBuildGEP2).restype, LLVMBuildGEP2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildInBoundsGEP2:=dll.LLVMBuildInBoundsGEP2).restype, LLVMBuildInBoundsGEP2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildGEPWithNoWrapFlags:=dll.LLVMBuildGEPWithNoWrapFlags).restype, LLVMBuildGEPWithNoWrapFlags.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), LLVMGEPNoWrapFlags]
except AttributeError: pass

try: (LLVMBuildStructGEP2:=dll.LLVMBuildStructGEP2).restype, LLVMBuildStructGEP2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildGlobalString:=dll.LLVMBuildGlobalString).restype, LLVMBuildGlobalString.argtypes = LLVMValueRef, [LLVMBuilderRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildGlobalStringPtr:=dll.LLVMBuildGlobalStringPtr).restype, LLVMBuildGlobalStringPtr.argtypes = LLVMValueRef, [LLVMBuilderRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetVolatile:=dll.LLVMGetVolatile).restype, LLVMGetVolatile.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetVolatile:=dll.LLVMSetVolatile).restype, LLVMSetVolatile.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetWeak:=dll.LLVMGetWeak).restype, LLVMGetWeak.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetWeak:=dll.LLVMSetWeak).restype, LLVMSetWeak.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGetOrdering:=dll.LLVMGetOrdering).restype, LLVMGetOrdering.argtypes = LLVMAtomicOrdering, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetOrdering:=dll.LLVMSetOrdering).restype, LLVMSetOrdering.argtypes = None, [LLVMValueRef, LLVMAtomicOrdering]
except AttributeError: pass

try: (LLVMGetAtomicRMWBinOp:=dll.LLVMGetAtomicRMWBinOp).restype, LLVMGetAtomicRMWBinOp.argtypes = LLVMAtomicRMWBinOp, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetAtomicRMWBinOp:=dll.LLVMSetAtomicRMWBinOp).restype, LLVMSetAtomicRMWBinOp.argtypes = None, [LLVMValueRef, LLVMAtomicRMWBinOp]
except AttributeError: pass

try: (LLVMBuildTrunc:=dll.LLVMBuildTrunc).restype, LLVMBuildTrunc.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildZExt:=dll.LLVMBuildZExt).restype, LLVMBuildZExt.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildSExt:=dll.LLVMBuildSExt).restype, LLVMBuildSExt.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFPToUI:=dll.LLVMBuildFPToUI).restype, LLVMBuildFPToUI.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFPToSI:=dll.LLVMBuildFPToSI).restype, LLVMBuildFPToSI.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildUIToFP:=dll.LLVMBuildUIToFP).restype, LLVMBuildUIToFP.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildSIToFP:=dll.LLVMBuildSIToFP).restype, LLVMBuildSIToFP.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFPTrunc:=dll.LLVMBuildFPTrunc).restype, LLVMBuildFPTrunc.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFPExt:=dll.LLVMBuildFPExt).restype, LLVMBuildFPExt.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildPtrToInt:=dll.LLVMBuildPtrToInt).restype, LLVMBuildPtrToInt.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildIntToPtr:=dll.LLVMBuildIntToPtr).restype, LLVMBuildIntToPtr.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildBitCast:=dll.LLVMBuildBitCast).restype, LLVMBuildBitCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildAddrSpaceCast:=dll.LLVMBuildAddrSpaceCast).restype, LLVMBuildAddrSpaceCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildZExtOrBitCast:=dll.LLVMBuildZExtOrBitCast).restype, LLVMBuildZExtOrBitCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildSExtOrBitCast:=dll.LLVMBuildSExtOrBitCast).restype, LLVMBuildSExtOrBitCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildTruncOrBitCast:=dll.LLVMBuildTruncOrBitCast).restype, LLVMBuildTruncOrBitCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildCast:=dll.LLVMBuildCast).restype, LLVMBuildCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMOpcode, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildPointerCast:=dll.LLVMBuildPointerCast).restype, LLVMBuildPointerCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildIntCast2:=dll.LLVMBuildIntCast2).restype, LLVMBuildIntCast2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, LLVMBool, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFPCast:=dll.LLVMBuildFPCast).restype, LLVMBuildFPCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildIntCast:=dll.LLVMBuildIntCast).restype, LLVMBuildIntCast.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetCastOpcode:=dll.LLVMGetCastOpcode).restype, LLVMGetCastOpcode.argtypes = LLVMOpcode, [LLVMValueRef, LLVMBool, LLVMTypeRef, LLVMBool]
except AttributeError: pass

try: (LLVMBuildICmp:=dll.LLVMBuildICmp).restype, LLVMBuildICmp.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMIntPredicate, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFCmp:=dll.LLVMBuildFCmp).restype, LLVMBuildFCmp.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMRealPredicate, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildPhi:=dll.LLVMBuildPhi).restype, LLVMBuildPhi.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildCall2:=dll.LLVMBuildCall2).restype, LLVMBuildCall2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildCallWithOperandBundles:=dll.LLVMBuildCallWithOperandBundles).restype, LLVMBuildCallWithOperandBundles.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, ctypes.POINTER(LLVMValueRef), ctypes.c_uint32, ctypes.POINTER(LLVMOperandBundleRef), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildSelect:=dll.LLVMBuildSelect).restype, LLVMBuildSelect.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildVAArg:=dll.LLVMBuildVAArg).restype, LLVMBuildVAArg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildExtractElement:=dll.LLVMBuildExtractElement).restype, LLVMBuildExtractElement.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildInsertElement:=dll.LLVMBuildInsertElement).restype, LLVMBuildInsertElement.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildShuffleVector:=dll.LLVMBuildShuffleVector).restype, LLVMBuildShuffleVector.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildExtractValue:=dll.LLVMBuildExtractValue).restype, LLVMBuildExtractValue.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildInsertValue:=dll.LLVMBuildInsertValue).restype, LLVMBuildInsertValue.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFreeze:=dll.LLVMBuildFreeze).restype, LLVMBuildFreeze.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildIsNull:=dll.LLVMBuildIsNull).restype, LLVMBuildIsNull.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildIsNotNull:=dll.LLVMBuildIsNotNull).restype, LLVMBuildIsNotNull.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildPtrDiff2:=dll.LLVMBuildPtrDiff2).restype, LLVMBuildPtrDiff2.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFence:=dll.LLVMBuildFence).restype, LLVMBuildFence.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMAtomicOrdering, LLVMBool, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildFenceSyncScope:=dll.LLVMBuildFenceSyncScope).restype, LLVMBuildFenceSyncScope.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMAtomicOrdering, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMBuildAtomicRMW:=dll.LLVMBuildAtomicRMW).restype, LLVMBuildAtomicRMW.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMAtomicRMWBinOp, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMBool]
except AttributeError: pass

try: (LLVMBuildAtomicRMWSyncScope:=dll.LLVMBuildAtomicRMWSyncScope).restype, LLVMBuildAtomicRMWSyncScope.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMAtomicRMWBinOp, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMBuildAtomicCmpXchg:=dll.LLVMBuildAtomicCmpXchg).restype, LLVMBuildAtomicCmpXchg.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMAtomicOrdering, LLVMBool]
except AttributeError: pass

try: (LLVMBuildAtomicCmpXchgSyncScope:=dll.LLVMBuildAtomicCmpXchgSyncScope).restype, LLVMBuildAtomicCmpXchgSyncScope.argtypes = LLVMValueRef, [LLVMBuilderRef, LLVMValueRef, LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMAtomicOrdering, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetNumMaskElements:=dll.LLVMGetNumMaskElements).restype, LLVMGetNumMaskElements.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetUndefMaskElem:=dll.LLVMGetUndefMaskElem).restype, LLVMGetUndefMaskElem.argtypes = ctypes.c_int32, []
except AttributeError: pass

try: (LLVMGetMaskValue:=dll.LLVMGetMaskValue).restype, LLVMGetMaskValue.argtypes = ctypes.c_int32, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIsAtomicSingleThread:=dll.LLVMIsAtomicSingleThread).restype, LLVMIsAtomicSingleThread.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetAtomicSingleThread:=dll.LLVMSetAtomicSingleThread).restype, LLVMSetAtomicSingleThread.argtypes = None, [LLVMValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMIsAtomic:=dll.LLVMIsAtomic).restype, LLVMIsAtomic.argtypes = LLVMBool, [LLVMValueRef]
except AttributeError: pass

try: (LLVMGetAtomicSyncScopeID:=dll.LLVMGetAtomicSyncScopeID).restype, LLVMGetAtomicSyncScopeID.argtypes = ctypes.c_uint32, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetAtomicSyncScopeID:=dll.LLVMSetAtomicSyncScopeID).restype, LLVMSetAtomicSyncScopeID.argtypes = None, [LLVMValueRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetCmpXchgSuccessOrdering:=dll.LLVMGetCmpXchgSuccessOrdering).restype, LLVMGetCmpXchgSuccessOrdering.argtypes = LLVMAtomicOrdering, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetCmpXchgSuccessOrdering:=dll.LLVMSetCmpXchgSuccessOrdering).restype, LLVMSetCmpXchgSuccessOrdering.argtypes = None, [LLVMValueRef, LLVMAtomicOrdering]
except AttributeError: pass

try: (LLVMGetCmpXchgFailureOrdering:=dll.LLVMGetCmpXchgFailureOrdering).restype, LLVMGetCmpXchgFailureOrdering.argtypes = LLVMAtomicOrdering, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetCmpXchgFailureOrdering:=dll.LLVMSetCmpXchgFailureOrdering).restype, LLVMSetCmpXchgFailureOrdering.argtypes = None, [LLVMValueRef, LLVMAtomicOrdering]
except AttributeError: pass

class struct_LLVMOpaqueModuleProvider(Struct): pass
LLVMModuleProviderRef = ctypes.POINTER(struct_LLVMOpaqueModuleProvider)
try: (LLVMCreateModuleProviderForExistingModule:=dll.LLVMCreateModuleProviderForExistingModule).restype, LLVMCreateModuleProviderForExistingModule.argtypes = LLVMModuleProviderRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMDisposeModuleProvider:=dll.LLVMDisposeModuleProvider).restype, LLVMDisposeModuleProvider.argtypes = None, [LLVMModuleProviderRef]
except AttributeError: pass

try: (LLVMCreateMemoryBufferWithContentsOfFile:=dll.LLVMCreateMemoryBufferWithContentsOfFile).restype, LLVMCreateMemoryBufferWithContentsOfFile.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMMemoryBufferRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMCreateMemoryBufferWithSTDIN:=dll.LLVMCreateMemoryBufferWithSTDIN).restype, LLVMCreateMemoryBufferWithSTDIN.argtypes = LLVMBool, [ctypes.POINTER(LLVMMemoryBufferRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMCreateMemoryBufferWithMemoryRange:=dll.LLVMCreateMemoryBufferWithMemoryRange).restype, LLVMCreateMemoryBufferWithMemoryRange.argtypes = LLVMMemoryBufferRef, [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), LLVMBool]
except AttributeError: pass

try: (LLVMCreateMemoryBufferWithMemoryRangeCopy:=dll.LLVMCreateMemoryBufferWithMemoryRangeCopy).restype, LLVMCreateMemoryBufferWithMemoryRangeCopy.argtypes = LLVMMemoryBufferRef, [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetBufferStart:=dll.LLVMGetBufferStart).restype, LLVMGetBufferStart.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMGetBufferSize:=dll.LLVMGetBufferSize).restype, LLVMGetBufferSize.argtypes = size_t, [LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMDisposeMemoryBuffer:=dll.LLVMDisposeMemoryBuffer).restype, LLVMDisposeMemoryBuffer.argtypes = None, [LLVMMemoryBufferRef]
except AttributeError: pass

class struct_LLVMOpaquePassManager(Struct): pass
LLVMPassManagerRef = ctypes.POINTER(struct_LLVMOpaquePassManager)
try: (LLVMCreatePassManager:=dll.LLVMCreatePassManager).restype, LLVMCreatePassManager.argtypes = LLVMPassManagerRef, []
except AttributeError: pass

try: (LLVMCreateFunctionPassManagerForModule:=dll.LLVMCreateFunctionPassManagerForModule).restype, LLVMCreateFunctionPassManagerForModule.argtypes = LLVMPassManagerRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMCreateFunctionPassManager:=dll.LLVMCreateFunctionPassManager).restype, LLVMCreateFunctionPassManager.argtypes = LLVMPassManagerRef, [LLVMModuleProviderRef]
except AttributeError: pass

try: (LLVMRunPassManager:=dll.LLVMRunPassManager).restype, LLVMRunPassManager.argtypes = LLVMBool, [LLVMPassManagerRef, LLVMModuleRef]
except AttributeError: pass

try: (LLVMInitializeFunctionPassManager:=dll.LLVMInitializeFunctionPassManager).restype, LLVMInitializeFunctionPassManager.argtypes = LLVMBool, [LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMRunFunctionPassManager:=dll.LLVMRunFunctionPassManager).restype, LLVMRunFunctionPassManager.argtypes = LLVMBool, [LLVMPassManagerRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMFinalizeFunctionPassManager:=dll.LLVMFinalizeFunctionPassManager).restype, LLVMFinalizeFunctionPassManager.argtypes = LLVMBool, [LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMDisposePassManager:=dll.LLVMDisposePassManager).restype, LLVMDisposePassManager.argtypes = None, [LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMStartMultithreaded:=dll.LLVMStartMultithreaded).restype, LLVMStartMultithreaded.argtypes = LLVMBool, []
except AttributeError: pass

try: (LLVMStopMultithreaded:=dll.LLVMStopMultithreaded).restype, LLVMStopMultithreaded.argtypes = None, []
except AttributeError: pass

try: (LLVMIsMultithreaded:=dll.LLVMIsMultithreaded).restype, LLVMIsMultithreaded.argtypes = LLVMBool, []
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMDIFlags = CEnum(ctypes.c_uint32)
LLVMDIFlagZero = LLVMDIFlags.define('LLVMDIFlagZero', 0)
LLVMDIFlagPrivate = LLVMDIFlags.define('LLVMDIFlagPrivate', 1)
LLVMDIFlagProtected = LLVMDIFlags.define('LLVMDIFlagProtected', 2)
LLVMDIFlagPublic = LLVMDIFlags.define('LLVMDIFlagPublic', 3)
LLVMDIFlagFwdDecl = LLVMDIFlags.define('LLVMDIFlagFwdDecl', 4)
LLVMDIFlagAppleBlock = LLVMDIFlags.define('LLVMDIFlagAppleBlock', 8)
LLVMDIFlagReservedBit4 = LLVMDIFlags.define('LLVMDIFlagReservedBit4', 16)
LLVMDIFlagVirtual = LLVMDIFlags.define('LLVMDIFlagVirtual', 32)
LLVMDIFlagArtificial = LLVMDIFlags.define('LLVMDIFlagArtificial', 64)
LLVMDIFlagExplicit = LLVMDIFlags.define('LLVMDIFlagExplicit', 128)
LLVMDIFlagPrototyped = LLVMDIFlags.define('LLVMDIFlagPrototyped', 256)
LLVMDIFlagObjcClassComplete = LLVMDIFlags.define('LLVMDIFlagObjcClassComplete', 512)
LLVMDIFlagObjectPointer = LLVMDIFlags.define('LLVMDIFlagObjectPointer', 1024)
LLVMDIFlagVector = LLVMDIFlags.define('LLVMDIFlagVector', 2048)
LLVMDIFlagStaticMember = LLVMDIFlags.define('LLVMDIFlagStaticMember', 4096)
LLVMDIFlagLValueReference = LLVMDIFlags.define('LLVMDIFlagLValueReference', 8192)
LLVMDIFlagRValueReference = LLVMDIFlags.define('LLVMDIFlagRValueReference', 16384)
LLVMDIFlagReserved = LLVMDIFlags.define('LLVMDIFlagReserved', 32768)
LLVMDIFlagSingleInheritance = LLVMDIFlags.define('LLVMDIFlagSingleInheritance', 65536)
LLVMDIFlagMultipleInheritance = LLVMDIFlags.define('LLVMDIFlagMultipleInheritance', 131072)
LLVMDIFlagVirtualInheritance = LLVMDIFlags.define('LLVMDIFlagVirtualInheritance', 196608)
LLVMDIFlagIntroducedVirtual = LLVMDIFlags.define('LLVMDIFlagIntroducedVirtual', 262144)
LLVMDIFlagBitField = LLVMDIFlags.define('LLVMDIFlagBitField', 524288)
LLVMDIFlagNoReturn = LLVMDIFlags.define('LLVMDIFlagNoReturn', 1048576)
LLVMDIFlagTypePassByValue = LLVMDIFlags.define('LLVMDIFlagTypePassByValue', 4194304)
LLVMDIFlagTypePassByReference = LLVMDIFlags.define('LLVMDIFlagTypePassByReference', 8388608)
LLVMDIFlagEnumClass = LLVMDIFlags.define('LLVMDIFlagEnumClass', 16777216)
LLVMDIFlagFixedEnum = LLVMDIFlags.define('LLVMDIFlagFixedEnum', 16777216)
LLVMDIFlagThunk = LLVMDIFlags.define('LLVMDIFlagThunk', 33554432)
LLVMDIFlagNonTrivial = LLVMDIFlags.define('LLVMDIFlagNonTrivial', 67108864)
LLVMDIFlagBigEndian = LLVMDIFlags.define('LLVMDIFlagBigEndian', 134217728)
LLVMDIFlagLittleEndian = LLVMDIFlags.define('LLVMDIFlagLittleEndian', 268435456)
LLVMDIFlagIndirectVirtualBase = LLVMDIFlags.define('LLVMDIFlagIndirectVirtualBase', 36)
LLVMDIFlagAccessibility = LLVMDIFlags.define('LLVMDIFlagAccessibility', 3)
LLVMDIFlagPtrToMemberRep = LLVMDIFlags.define('LLVMDIFlagPtrToMemberRep', 196608)

LLVMDWARFSourceLanguage = CEnum(ctypes.c_uint32)
LLVMDWARFSourceLanguageC89 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC89', 0)
LLVMDWARFSourceLanguageC = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC', 1)
LLVMDWARFSourceLanguageAda83 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAda83', 2)
LLVMDWARFSourceLanguageC_plus_plus = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus', 3)
LLVMDWARFSourceLanguageCobol74 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageCobol74', 4)
LLVMDWARFSourceLanguageCobol85 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageCobol85', 5)
LLVMDWARFSourceLanguageFortran77 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran77', 6)
LLVMDWARFSourceLanguageFortran90 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran90', 7)
LLVMDWARFSourceLanguagePascal83 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguagePascal83', 8)
LLVMDWARFSourceLanguageModula2 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageModula2', 9)
LLVMDWARFSourceLanguageJava = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageJava', 10)
LLVMDWARFSourceLanguageC99 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC99', 11)
LLVMDWARFSourceLanguageAda95 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAda95', 12)
LLVMDWARFSourceLanguageFortran95 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran95', 13)
LLVMDWARFSourceLanguagePLI = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguagePLI', 14)
LLVMDWARFSourceLanguageObjC = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageObjC', 15)
LLVMDWARFSourceLanguageObjC_plus_plus = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageObjC_plus_plus', 16)
LLVMDWARFSourceLanguageUPC = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageUPC', 17)
LLVMDWARFSourceLanguageD = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageD', 18)
LLVMDWARFSourceLanguagePython = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguagePython', 19)
LLVMDWARFSourceLanguageOpenCL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageOpenCL', 20)
LLVMDWARFSourceLanguageGo = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageGo', 21)
LLVMDWARFSourceLanguageModula3 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageModula3', 22)
LLVMDWARFSourceLanguageHaskell = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageHaskell', 23)
LLVMDWARFSourceLanguageC_plus_plus_03 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_03', 24)
LLVMDWARFSourceLanguageC_plus_plus_11 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_11', 25)
LLVMDWARFSourceLanguageOCaml = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageOCaml', 26)
LLVMDWARFSourceLanguageRust = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageRust', 27)
LLVMDWARFSourceLanguageC11 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC11', 28)
LLVMDWARFSourceLanguageSwift = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageSwift', 29)
LLVMDWARFSourceLanguageJulia = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageJulia', 30)
LLVMDWARFSourceLanguageDylan = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageDylan', 31)
LLVMDWARFSourceLanguageC_plus_plus_14 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_14', 32)
LLVMDWARFSourceLanguageFortran03 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran03', 33)
LLVMDWARFSourceLanguageFortran08 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran08', 34)
LLVMDWARFSourceLanguageRenderScript = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageRenderScript', 35)
LLVMDWARFSourceLanguageBLISS = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageBLISS', 36)
LLVMDWARFSourceLanguageKotlin = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageKotlin', 37)
LLVMDWARFSourceLanguageZig = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageZig', 38)
LLVMDWARFSourceLanguageCrystal = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageCrystal', 39)
LLVMDWARFSourceLanguageC_plus_plus_17 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_17', 40)
LLVMDWARFSourceLanguageC_plus_plus_20 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_plus_plus_20', 41)
LLVMDWARFSourceLanguageC17 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC17', 42)
LLVMDWARFSourceLanguageFortran18 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageFortran18', 43)
LLVMDWARFSourceLanguageAda2005 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAda2005', 44)
LLVMDWARFSourceLanguageAda2012 = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAda2012', 45)
LLVMDWARFSourceLanguageHIP = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageHIP', 46)
LLVMDWARFSourceLanguageAssembly = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageAssembly', 47)
LLVMDWARFSourceLanguageC_sharp = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageC_sharp', 48)
LLVMDWARFSourceLanguageMojo = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageMojo', 49)
LLVMDWARFSourceLanguageGLSL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageGLSL', 50)
LLVMDWARFSourceLanguageGLSL_ES = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageGLSL_ES', 51)
LLVMDWARFSourceLanguageHLSL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageHLSL', 52)
LLVMDWARFSourceLanguageOpenCL_CPP = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageOpenCL_CPP', 53)
LLVMDWARFSourceLanguageCPP_for_OpenCL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageCPP_for_OpenCL', 54)
LLVMDWARFSourceLanguageSYCL = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageSYCL', 55)
LLVMDWARFSourceLanguageRuby = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageRuby', 56)
LLVMDWARFSourceLanguageMove = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageMove', 57)
LLVMDWARFSourceLanguageHylo = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageHylo', 58)
LLVMDWARFSourceLanguageMetal = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageMetal', 59)
LLVMDWARFSourceLanguageMips_Assembler = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageMips_Assembler', 60)
LLVMDWARFSourceLanguageGOOGLE_RenderScript = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageGOOGLE_RenderScript', 61)
LLVMDWARFSourceLanguageBORLAND_Delphi = LLVMDWARFSourceLanguage.define('LLVMDWARFSourceLanguageBORLAND_Delphi', 62)

LLVMDWARFEmissionKind = CEnum(ctypes.c_uint32)
LLVMDWARFEmissionNone = LLVMDWARFEmissionKind.define('LLVMDWARFEmissionNone', 0)
LLVMDWARFEmissionFull = LLVMDWARFEmissionKind.define('LLVMDWARFEmissionFull', 1)
LLVMDWARFEmissionLineTablesOnly = LLVMDWARFEmissionKind.define('LLVMDWARFEmissionLineTablesOnly', 2)

_anonenum3 = CEnum(ctypes.c_uint32)
LLVMMDStringMetadataKind = _anonenum3.define('LLVMMDStringMetadataKind', 0)
LLVMConstantAsMetadataMetadataKind = _anonenum3.define('LLVMConstantAsMetadataMetadataKind', 1)
LLVMLocalAsMetadataMetadataKind = _anonenum3.define('LLVMLocalAsMetadataMetadataKind', 2)
LLVMDistinctMDOperandPlaceholderMetadataKind = _anonenum3.define('LLVMDistinctMDOperandPlaceholderMetadataKind', 3)
LLVMMDTupleMetadataKind = _anonenum3.define('LLVMMDTupleMetadataKind', 4)
LLVMDILocationMetadataKind = _anonenum3.define('LLVMDILocationMetadataKind', 5)
LLVMDIExpressionMetadataKind = _anonenum3.define('LLVMDIExpressionMetadataKind', 6)
LLVMDIGlobalVariableExpressionMetadataKind = _anonenum3.define('LLVMDIGlobalVariableExpressionMetadataKind', 7)
LLVMGenericDINodeMetadataKind = _anonenum3.define('LLVMGenericDINodeMetadataKind', 8)
LLVMDISubrangeMetadataKind = _anonenum3.define('LLVMDISubrangeMetadataKind', 9)
LLVMDIEnumeratorMetadataKind = _anonenum3.define('LLVMDIEnumeratorMetadataKind', 10)
LLVMDIBasicTypeMetadataKind = _anonenum3.define('LLVMDIBasicTypeMetadataKind', 11)
LLVMDIDerivedTypeMetadataKind = _anonenum3.define('LLVMDIDerivedTypeMetadataKind', 12)
LLVMDICompositeTypeMetadataKind = _anonenum3.define('LLVMDICompositeTypeMetadataKind', 13)
LLVMDISubroutineTypeMetadataKind = _anonenum3.define('LLVMDISubroutineTypeMetadataKind', 14)
LLVMDIFileMetadataKind = _anonenum3.define('LLVMDIFileMetadataKind', 15)
LLVMDICompileUnitMetadataKind = _anonenum3.define('LLVMDICompileUnitMetadataKind', 16)
LLVMDISubprogramMetadataKind = _anonenum3.define('LLVMDISubprogramMetadataKind', 17)
LLVMDILexicalBlockMetadataKind = _anonenum3.define('LLVMDILexicalBlockMetadataKind', 18)
LLVMDILexicalBlockFileMetadataKind = _anonenum3.define('LLVMDILexicalBlockFileMetadataKind', 19)
LLVMDINamespaceMetadataKind = _anonenum3.define('LLVMDINamespaceMetadataKind', 20)
LLVMDIModuleMetadataKind = _anonenum3.define('LLVMDIModuleMetadataKind', 21)
LLVMDITemplateTypeParameterMetadataKind = _anonenum3.define('LLVMDITemplateTypeParameterMetadataKind', 22)
LLVMDITemplateValueParameterMetadataKind = _anonenum3.define('LLVMDITemplateValueParameterMetadataKind', 23)
LLVMDIGlobalVariableMetadataKind = _anonenum3.define('LLVMDIGlobalVariableMetadataKind', 24)
LLVMDILocalVariableMetadataKind = _anonenum3.define('LLVMDILocalVariableMetadataKind', 25)
LLVMDILabelMetadataKind = _anonenum3.define('LLVMDILabelMetadataKind', 26)
LLVMDIObjCPropertyMetadataKind = _anonenum3.define('LLVMDIObjCPropertyMetadataKind', 27)
LLVMDIImportedEntityMetadataKind = _anonenum3.define('LLVMDIImportedEntityMetadataKind', 28)
LLVMDIMacroMetadataKind = _anonenum3.define('LLVMDIMacroMetadataKind', 29)
LLVMDIMacroFileMetadataKind = _anonenum3.define('LLVMDIMacroFileMetadataKind', 30)
LLVMDICommonBlockMetadataKind = _anonenum3.define('LLVMDICommonBlockMetadataKind', 31)
LLVMDIStringTypeMetadataKind = _anonenum3.define('LLVMDIStringTypeMetadataKind', 32)
LLVMDIGenericSubrangeMetadataKind = _anonenum3.define('LLVMDIGenericSubrangeMetadataKind', 33)
LLVMDIArgListMetadataKind = _anonenum3.define('LLVMDIArgListMetadataKind', 34)
LLVMDIAssignIDMetadataKind = _anonenum3.define('LLVMDIAssignIDMetadataKind', 35)

LLVMMetadataKind = ctypes.c_uint32
LLVMDWARFTypeEncoding = ctypes.c_uint32
LLVMDWARFMacinfoRecordType = CEnum(ctypes.c_uint32)
LLVMDWARFMacinfoRecordTypeDefine = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeDefine', 1)
LLVMDWARFMacinfoRecordTypeMacro = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeMacro', 2)
LLVMDWARFMacinfoRecordTypeStartFile = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeStartFile', 3)
LLVMDWARFMacinfoRecordTypeEndFile = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeEndFile', 4)
LLVMDWARFMacinfoRecordTypeVendorExt = LLVMDWARFMacinfoRecordType.define('LLVMDWARFMacinfoRecordTypeVendorExt', 255)

try: (LLVMDebugMetadataVersion:=dll.LLVMDebugMetadataVersion).restype, LLVMDebugMetadataVersion.argtypes = ctypes.c_uint32, []
except AttributeError: pass

try: (LLVMGetModuleDebugMetadataVersion:=dll.LLVMGetModuleDebugMetadataVersion).restype, LLVMGetModuleDebugMetadataVersion.argtypes = ctypes.c_uint32, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMStripModuleDebugInfo:=dll.LLVMStripModuleDebugInfo).restype, LLVMStripModuleDebugInfo.argtypes = LLVMBool, [LLVMModuleRef]
except AttributeError: pass

class struct_LLVMOpaqueDIBuilder(Struct): pass
LLVMDIBuilderRef = ctypes.POINTER(struct_LLVMOpaqueDIBuilder)
try: (LLVMCreateDIBuilderDisallowUnresolved:=dll.LLVMCreateDIBuilderDisallowUnresolved).restype, LLVMCreateDIBuilderDisallowUnresolved.argtypes = LLVMDIBuilderRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMCreateDIBuilder:=dll.LLVMCreateDIBuilder).restype, LLVMCreateDIBuilder.argtypes = LLVMDIBuilderRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMDisposeDIBuilder:=dll.LLVMDisposeDIBuilder).restype, LLVMDisposeDIBuilder.argtypes = None, [LLVMDIBuilderRef]
except AttributeError: pass

try: (LLVMDIBuilderFinalize:=dll.LLVMDIBuilderFinalize).restype, LLVMDIBuilderFinalize.argtypes = None, [LLVMDIBuilderRef]
except AttributeError: pass

try: (LLVMDIBuilderFinalizeSubprogram:=dll.LLVMDIBuilderFinalizeSubprogram).restype, LLVMDIBuilderFinalizeSubprogram.argtypes = None, [LLVMDIBuilderRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateCompileUnit:=dll.LLVMDIBuilderCreateCompileUnit).restype, LLVMDIBuilderCreateCompileUnit.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMDWARFSourceLanguage, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool, ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, LLVMDWARFEmissionKind, ctypes.c_uint32, LLVMBool, LLVMBool, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateFile:=dll.LLVMDIBuilderCreateFile).restype, LLVMDIBuilderCreateFile.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateModule:=dll.LLVMDIBuilderCreateModule).restype, LLVMDIBuilderCreateModule.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateNameSpace:=dll.LLVMDIBuilderCreateNameSpace).restype, LLVMDIBuilderCreateNameSpace.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMBool]
except AttributeError: pass

try: (LLVMDIBuilderCreateFunction:=dll.LLVMDIBuilderCreateFunction).restype, LLVMDIBuilderCreateFunction.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMBool, ctypes.c_uint32, LLVMDIFlags, LLVMBool]
except AttributeError: pass

try: (LLVMDIBuilderCreateLexicalBlock:=dll.LLVMDIBuilderCreateLexicalBlock).restype, LLVMDIBuilderCreateLexicalBlock.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMDIBuilderCreateLexicalBlockFile:=dll.LLVMDIBuilderCreateLexicalBlockFile).restype, LLVMDIBuilderCreateLexicalBlockFile.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMDIBuilderCreateImportedModuleFromNamespace:=dll.LLVMDIBuilderCreateImportedModuleFromNamespace).restype, LLVMDIBuilderCreateImportedModuleFromNamespace.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMDIBuilderCreateImportedModuleFromAlias:=dll.LLVMDIBuilderCreateImportedModuleFromAlias).restype, LLVMDIBuilderCreateImportedModuleFromAlias.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMDIBuilderCreateImportedModuleFromModule:=dll.LLVMDIBuilderCreateImportedModuleFromModule).restype, LLVMDIBuilderCreateImportedModuleFromModule.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMDIBuilderCreateImportedDeclaration:=dll.LLVMDIBuilderCreateImportedDeclaration).restype, LLVMDIBuilderCreateImportedDeclaration.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMDIBuilderCreateDebugLocation:=dll.LLVMDIBuilderCreateDebugLocation).restype, LLVMDIBuilderCreateDebugLocation.argtypes = LLVMMetadataRef, [LLVMContextRef, ctypes.c_uint32, ctypes.c_uint32, LLVMMetadataRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDILocationGetLine:=dll.LLVMDILocationGetLine).restype, LLVMDILocationGetLine.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDILocationGetColumn:=dll.LLVMDILocationGetColumn).restype, LLVMDILocationGetColumn.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDILocationGetScope:=dll.LLVMDILocationGetScope).restype, LLVMDILocationGetScope.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDILocationGetInlinedAt:=dll.LLVMDILocationGetInlinedAt).restype, LLVMDILocationGetInlinedAt.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIScopeGetFile:=dll.LLVMDIScopeGetFile).restype, LLVMDIScopeGetFile.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIFileGetDirectory:=dll.LLVMDIFileGetDirectory).restype, LLVMDIFileGetDirectory.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (LLVMDIFileGetFilename:=dll.LLVMDIFileGetFilename).restype, LLVMDIFileGetFilename.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (LLVMDIFileGetSource:=dll.LLVMDIFileGetSource).restype, LLVMDIFileGetSource.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMetadataRef, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (LLVMDIBuilderGetOrCreateTypeArray:=dll.LLVMDIBuilderGetOrCreateTypeArray).restype, LLVMDIBuilderGetOrCreateTypeArray.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(LLVMMetadataRef), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateSubroutineType:=dll.LLVMDIBuilderCreateSubroutineType).restype, LLVMDIBuilderCreateSubroutineType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, LLVMDIFlags]
except AttributeError: pass

try: (LLVMDIBuilderCreateMacro:=dll.LLVMDIBuilderCreateMacro).restype, LLVMDIBuilderCreateMacro.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.c_uint32, LLVMDWARFMacinfoRecordType, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateTempMacroFile:=dll.LLVMDIBuilderCreateTempMacroFile).restype, LLVMDIBuilderCreateTempMacroFile.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

int64_t = ctypes.c_int64
try: (LLVMDIBuilderCreateEnumerator:=dll.LLVMDIBuilderCreateEnumerator).restype, LLVMDIBuilderCreateEnumerator.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, int64_t, LLVMBool]
except AttributeError: pass

uint32_t = ctypes.c_uint32
try: (LLVMDIBuilderCreateEnumerationType:=dll.LLVMDIBuilderCreateEnumerationType).restype, LLVMDIBuilderCreateEnumerationType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateUnionType:=dll.LLVMDIBuilderCreateUnionType).restype, LLVMDIBuilderCreateUnionType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, LLVMDIFlags, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateArrayType:=dll.LLVMDIBuilderCreateArrayType).restype, LLVMDIBuilderCreateArrayType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, uint64_t, uint32_t, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMDIBuilderCreateVectorType:=dll.LLVMDIBuilderCreateVectorType).restype, LLVMDIBuilderCreateVectorType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, uint64_t, uint32_t, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32]
except AttributeError: pass

try: (LLVMDIBuilderCreateUnspecifiedType:=dll.LLVMDIBuilderCreateUnspecifiedType).restype, LLVMDIBuilderCreateUnspecifiedType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateBasicType:=dll.LLVMDIBuilderCreateBasicType).restype, LLVMDIBuilderCreateBasicType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, uint64_t, LLVMDWARFTypeEncoding, LLVMDIFlags]
except AttributeError: pass

try: (LLVMDIBuilderCreatePointerType:=dll.LLVMDIBuilderCreatePointerType).restype, LLVMDIBuilderCreatePointerType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, uint64_t, uint32_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateStructType:=dll.LLVMDIBuilderCreateStructType).restype, LLVMDIBuilderCreateStructType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, LLVMDIFlags, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, ctypes.c_uint32, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateMemberType:=dll.LLVMDIBuilderCreateMemberType).restype, LLVMDIBuilderCreateMemberType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, uint64_t, LLVMDIFlags, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateStaticMemberType:=dll.LLVMDIBuilderCreateStaticMemberType).restype, LLVMDIBuilderCreateStaticMemberType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMDIFlags, LLVMValueRef, uint32_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateMemberPointerType:=dll.LLVMDIBuilderCreateMemberPointerType).restype, LLVMDIBuilderCreateMemberPointerType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, uint64_t, uint32_t, LLVMDIFlags]
except AttributeError: pass

try: (LLVMDIBuilderCreateObjCIVar:=dll.LLVMDIBuilderCreateObjCIVar).restype, LLVMDIBuilderCreateObjCIVar.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, uint64_t, LLVMDIFlags, LLVMMetadataRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateObjCProperty:=dll.LLVMDIBuilderCreateObjCProperty).restype, LLVMDIBuilderCreateObjCProperty.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateObjectPointerType:=dll.LLVMDIBuilderCreateObjectPointerType).restype, LLVMDIBuilderCreateObjectPointerType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMBool]
except AttributeError: pass

try: (LLVMDIBuilderCreateQualifiedType:=dll.LLVMDIBuilderCreateQualifiedType).restype, LLVMDIBuilderCreateQualifiedType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateReferenceType:=dll.LLVMDIBuilderCreateReferenceType).restype, LLVMDIBuilderCreateReferenceType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.c_uint32, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateNullPtrType:=dll.LLVMDIBuilderCreateNullPtrType).restype, LLVMDIBuilderCreateNullPtrType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateTypedef:=dll.LLVMDIBuilderCreateTypedef).restype, LLVMDIBuilderCreateTypedef.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, uint32_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateInheritance:=dll.LLVMDIBuilderCreateInheritance).restype, LLVMDIBuilderCreateInheritance.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, uint64_t, uint32_t, LLVMDIFlags]
except AttributeError: pass

try: (LLVMDIBuilderCreateForwardDecl:=dll.LLVMDIBuilderCreateForwardDecl).restype, LLVMDIBuilderCreateForwardDecl.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.c_uint32, uint64_t, uint32_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateReplaceableCompositeType:=dll.LLVMDIBuilderCreateReplaceableCompositeType).restype, LLVMDIBuilderCreateReplaceableCompositeType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, LLVMMetadataRef, ctypes.c_uint32, ctypes.c_uint32, uint64_t, uint32_t, LLVMDIFlags, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateBitFieldMemberType:=dll.LLVMDIBuilderCreateBitFieldMemberType).restype, LLVMDIBuilderCreateBitFieldMemberType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint64_t, uint64_t, LLVMDIFlags, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateClassType:=dll.LLVMDIBuilderCreateClassType).restype, LLVMDIBuilderCreateClassType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, uint64_t, uint32_t, uint64_t, LLVMDIFlags, LLVMMetadataRef, ctypes.POINTER(LLVMMetadataRef), ctypes.c_uint32, LLVMMetadataRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateArtificialType:=dll.LLVMDIBuilderCreateArtificialType).restype, LLVMDIBuilderCreateArtificialType.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDITypeGetName:=dll.LLVMDITypeGetName).restype, LLVMDITypeGetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMMetadataRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMDITypeGetSizeInBits:=dll.LLVMDITypeGetSizeInBits).restype, LLVMDITypeGetSizeInBits.argtypes = uint64_t, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDITypeGetOffsetInBits:=dll.LLVMDITypeGetOffsetInBits).restype, LLVMDITypeGetOffsetInBits.argtypes = uint64_t, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDITypeGetAlignInBits:=dll.LLVMDITypeGetAlignInBits).restype, LLVMDITypeGetAlignInBits.argtypes = uint32_t, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDITypeGetLine:=dll.LLVMDITypeGetLine).restype, LLVMDITypeGetLine.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDITypeGetFlags:=dll.LLVMDITypeGetFlags).restype, LLVMDITypeGetFlags.argtypes = LLVMDIFlags, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderGetOrCreateSubrange:=dll.LLVMDIBuilderGetOrCreateSubrange).restype, LLVMDIBuilderGetOrCreateSubrange.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, int64_t, int64_t]
except AttributeError: pass

try: (LLVMDIBuilderGetOrCreateArray:=dll.LLVMDIBuilderGetOrCreateArray).restype, LLVMDIBuilderGetOrCreateArray.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(LLVMMetadataRef), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateExpression:=dll.LLVMDIBuilderCreateExpression).restype, LLVMDIBuilderCreateExpression.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, ctypes.POINTER(uint64_t), size_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateConstantValueExpression:=dll.LLVMDIBuilderCreateConstantValueExpression).restype, LLVMDIBuilderCreateConstantValueExpression.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, uint64_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateGlobalVariableExpression:=dll.LLVMDIBuilderCreateGlobalVariableExpression).restype, LLVMDIBuilderCreateGlobalVariableExpression.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMMetadataRef, LLVMMetadataRef, uint32_t]
except AttributeError: pass

uint16_t = ctypes.c_uint16
try: (LLVMGetDINodeTag:=dll.LLVMGetDINodeTag).restype, LLVMGetDINodeTag.argtypes = uint16_t, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIGlobalVariableExpressionGetVariable:=dll.LLVMDIGlobalVariableExpressionGetVariable).restype, LLVMDIGlobalVariableExpressionGetVariable.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIGlobalVariableExpressionGetExpression:=dll.LLVMDIGlobalVariableExpressionGetExpression).restype, LLVMDIGlobalVariableExpressionGetExpression.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIVariableGetFile:=dll.LLVMDIVariableGetFile).restype, LLVMDIVariableGetFile.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIVariableGetScope:=dll.LLVMDIVariableGetScope).restype, LLVMDIVariableGetScope.argtypes = LLVMMetadataRef, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIVariableGetLine:=dll.LLVMDIVariableGetLine).restype, LLVMDIVariableGetLine.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMTemporaryMDNode:=dll.LLVMTemporaryMDNode).restype, LLVMTemporaryMDNode.argtypes = LLVMMetadataRef, [LLVMContextRef, ctypes.POINTER(LLVMMetadataRef), size_t]
except AttributeError: pass

try: (LLVMDisposeTemporaryMDNode:=dll.LLVMDisposeTemporaryMDNode).restype, LLVMDisposeTemporaryMDNode.argtypes = None, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMMetadataReplaceAllUsesWith:=dll.LLVMMetadataReplaceAllUsesWith).restype, LLVMMetadataReplaceAllUsesWith.argtypes = None, [LLVMMetadataRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateTempGlobalVariableFwdDecl:=dll.LLVMDIBuilderCreateTempGlobalVariableFwdDecl).restype, LLVMDIBuilderCreateTempGlobalVariableFwdDecl.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMMetadataRef, uint32_t]
except AttributeError: pass

try: (LLVMDIBuilderInsertDeclareRecordBefore:=dll.LLVMDIBuilderInsertDeclareRecordBefore).restype, LLVMDIBuilderInsertDeclareRecordBefore.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMDIBuilderInsertDeclareRecordAtEnd:=dll.LLVMDIBuilderInsertDeclareRecordAtEnd).restype, LLVMDIBuilderInsertDeclareRecordAtEnd.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMDIBuilderInsertDbgValueRecordBefore:=dll.LLVMDIBuilderInsertDbgValueRecordBefore).restype, LLVMDIBuilderInsertDbgValueRecordBefore.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMDIBuilderInsertDbgValueRecordAtEnd:=dll.LLVMDIBuilderInsertDbgValueRecordAtEnd).restype, LLVMDIBuilderInsertDbgValueRecordAtEnd.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateAutoVariable:=dll.LLVMDIBuilderCreateAutoVariable).restype, LLVMDIBuilderCreateAutoVariable.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMDIFlags, uint32_t]
except AttributeError: pass

try: (LLVMDIBuilderCreateParameterVariable:=dll.LLVMDIBuilderCreateParameterVariable).restype, LLVMDIBuilderCreateParameterVariable.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_uint32, LLVMMetadataRef, ctypes.c_uint32, LLVMMetadataRef, LLVMBool, LLVMDIFlags]
except AttributeError: pass

try: (LLVMGetSubprogram:=dll.LLVMGetSubprogram).restype, LLVMGetSubprogram.argtypes = LLVMMetadataRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMSetSubprogram:=dll.LLVMSetSubprogram).restype, LLVMSetSubprogram.argtypes = None, [LLVMValueRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDISubprogramGetLine:=dll.LLVMDISubprogramGetLine).restype, LLVMDISubprogramGetLine.argtypes = ctypes.c_uint32, [LLVMMetadataRef]
except AttributeError: pass

try: (LLVMInstructionGetDebugLoc:=dll.LLVMInstructionGetDebugLoc).restype, LLVMInstructionGetDebugLoc.argtypes = LLVMMetadataRef, [LLVMValueRef]
except AttributeError: pass

try: (LLVMInstructionSetDebugLoc:=dll.LLVMInstructionSetDebugLoc).restype, LLVMInstructionSetDebugLoc.argtypes = None, [LLVMValueRef, LLVMMetadataRef]
except AttributeError: pass

try: (LLVMDIBuilderCreateLabel:=dll.LLVMDIBuilderCreateLabel).restype, LLVMDIBuilderCreateLabel.argtypes = LLVMMetadataRef, [LLVMDIBuilderRef, LLVMMetadataRef, ctypes.POINTER(ctypes.c_char), size_t, LLVMMetadataRef, ctypes.c_uint32, LLVMBool]
except AttributeError: pass

try: (LLVMDIBuilderInsertLabelBefore:=dll.LLVMDIBuilderInsertLabelBefore).restype, LLVMDIBuilderInsertLabelBefore.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMDIBuilderInsertLabelAtEnd:=dll.LLVMDIBuilderInsertLabelAtEnd).restype, LLVMDIBuilderInsertLabelAtEnd.argtypes = LLVMDbgRecordRef, [LLVMDIBuilderRef, LLVMMetadataRef, LLVMMetadataRef, LLVMBasicBlockRef]
except AttributeError: pass

try: (LLVMGetMetadataKind:=dll.LLVMGetMetadataKind).restype, LLVMGetMetadataKind.argtypes = LLVMMetadataKind, [LLVMMetadataRef]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMDisasmContextRef = ctypes.c_void_p
LLVMOpInfoCallback = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int32, ctypes.c_void_p)
LLVMSymbolLookupCallback = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
try: (LLVMCreateDisasm:=dll.LLVMCreateDisasm).restype, LLVMCreateDisasm.argtypes = LLVMDisasmContextRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_int32, LLVMOpInfoCallback, LLVMSymbolLookupCallback]
except AttributeError: pass

try: (LLVMCreateDisasmCPU:=dll.LLVMCreateDisasmCPU).restype, LLVMCreateDisasmCPU.argtypes = LLVMDisasmContextRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_int32, LLVMOpInfoCallback, LLVMSymbolLookupCallback]
except AttributeError: pass

try: (LLVMCreateDisasmCPUFeatures:=dll.LLVMCreateDisasmCPUFeatures).restype, LLVMCreateDisasmCPUFeatures.argtypes = LLVMDisasmContextRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_int32, LLVMOpInfoCallback, LLVMSymbolLookupCallback]
except AttributeError: pass

try: (LLVMSetDisasmOptions:=dll.LLVMSetDisasmOptions).restype, LLVMSetDisasmOptions.argtypes = ctypes.c_int32, [LLVMDisasmContextRef, uint64_t]
except AttributeError: pass

try: (LLVMDisasmDispose:=dll.LLVMDisasmDispose).restype, LLVMDisasmDispose.argtypes = None, [LLVMDisasmContextRef]
except AttributeError: pass

try: (LLVMDisasmInstruction:=dll.LLVMDisasmInstruction).restype, LLVMDisasmInstruction.argtypes = size_t, [LLVMDisasmContextRef, ctypes.POINTER(uint8_t), uint64_t, uint64_t, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

class struct_LLVMOpInfoSymbol1(Struct): pass
struct_LLVMOpInfoSymbol1._fields_ = [
  ('Present', uint64_t),
  ('Name', ctypes.POINTER(ctypes.c_char)),
  ('Value', uint64_t),
]
class struct_LLVMOpInfo1(Struct): pass
struct_LLVMOpInfo1._fields_ = [
  ('AddSymbol', struct_LLVMOpInfoSymbol1),
  ('SubtractSymbol', struct_LLVMOpInfoSymbol1),
  ('Value', uint64_t),
  ('VariantKind', uint64_t),
]
class struct_LLVMOpaqueError(Struct): pass
LLVMErrorRef = ctypes.POINTER(struct_LLVMOpaqueError)
LLVMErrorTypeId = ctypes.c_void_p
try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMInstallFatalErrorHandler:=dll.LLVMInstallFatalErrorHandler).restype, LLVMInstallFatalErrorHandler.argtypes = None, [LLVMFatalErrorHandler]
except AttributeError: pass

try: (LLVMResetFatalErrorHandler:=dll.LLVMResetFatalErrorHandler).restype, LLVMResetFatalErrorHandler.argtypes = None, []
except AttributeError: pass

try: (LLVMEnablePrettyStackTrace:=dll.LLVMEnablePrettyStackTrace).restype, LLVMEnablePrettyStackTrace.argtypes = None, []
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

class struct_LLVMOpaqueTargetData(Struct): pass
LLVMTargetDataRef = ctypes.POINTER(struct_LLVMOpaqueTargetData)
try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

class struct_LLVMOpaqueTargetLibraryInfotData(Struct): pass
LLVMTargetLibraryInfoRef = ctypes.POINTER(struct_LLVMOpaqueTargetLibraryInfotData)
try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

enum_LLVMByteOrdering = CEnum(ctypes.c_uint32)
LLVMBigEndian = enum_LLVMByteOrdering.define('LLVMBigEndian', 0)
LLVMLittleEndian = enum_LLVMByteOrdering.define('LLVMLittleEndian', 1)

try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

class struct_LLVMTarget(Struct): pass
LLVMTargetRef = ctypes.POINTER(struct_LLVMTarget)
try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

class struct_LLVMOpaqueTargetMachineOptions(Struct): pass
LLVMTargetMachineOptionsRef = ctypes.POINTER(struct_LLVMOpaqueTargetMachineOptions)
try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

LLVMCodeGenOptLevel = CEnum(ctypes.c_uint32)
LLVMCodeGenLevelNone = LLVMCodeGenOptLevel.define('LLVMCodeGenLevelNone', 0)
LLVMCodeGenLevelLess = LLVMCodeGenOptLevel.define('LLVMCodeGenLevelLess', 1)
LLVMCodeGenLevelDefault = LLVMCodeGenOptLevel.define('LLVMCodeGenLevelDefault', 2)
LLVMCodeGenLevelAggressive = LLVMCodeGenOptLevel.define('LLVMCodeGenLevelAggressive', 3)

try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

LLVMRelocMode = CEnum(ctypes.c_uint32)
LLVMRelocDefault = LLVMRelocMode.define('LLVMRelocDefault', 0)
LLVMRelocStatic = LLVMRelocMode.define('LLVMRelocStatic', 1)
LLVMRelocPIC = LLVMRelocMode.define('LLVMRelocPIC', 2)
LLVMRelocDynamicNoPic = LLVMRelocMode.define('LLVMRelocDynamicNoPic', 3)
LLVMRelocROPI = LLVMRelocMode.define('LLVMRelocROPI', 4)
LLVMRelocRWPI = LLVMRelocMode.define('LLVMRelocRWPI', 5)
LLVMRelocROPI_RWPI = LLVMRelocMode.define('LLVMRelocROPI_RWPI', 6)

try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

LLVMCodeModel = CEnum(ctypes.c_uint32)
LLVMCodeModelDefault = LLVMCodeModel.define('LLVMCodeModelDefault', 0)
LLVMCodeModelJITDefault = LLVMCodeModel.define('LLVMCodeModelJITDefault', 1)
LLVMCodeModelTiny = LLVMCodeModel.define('LLVMCodeModelTiny', 2)
LLVMCodeModelSmall = LLVMCodeModel.define('LLVMCodeModelSmall', 3)
LLVMCodeModelKernel = LLVMCodeModel.define('LLVMCodeModelKernel', 4)
LLVMCodeModelMedium = LLVMCodeModel.define('LLVMCodeModelMedium', 5)
LLVMCodeModelLarge = LLVMCodeModel.define('LLVMCodeModelLarge', 6)

try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

class struct_LLVMOpaqueTargetMachine(Struct): pass
LLVMTargetMachineRef = ctypes.POINTER(struct_LLVMOpaqueTargetMachine)
try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

LLVMGlobalISelAbortMode = CEnum(ctypes.c_uint32)
LLVMGlobalISelAbortEnable = LLVMGlobalISelAbortMode.define('LLVMGlobalISelAbortEnable', 0)
LLVMGlobalISelAbortDisable = LLVMGlobalISelAbortMode.define('LLVMGlobalISelAbortDisable', 1)
LLVMGlobalISelAbortDisableWithDiag = LLVMGlobalISelAbortMode.define('LLVMGlobalISelAbortDisableWithDiag', 2)

try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

LLVMCodeGenFileType = CEnum(ctypes.c_uint32)
LLVMAssemblyFile = LLVMCodeGenFileType.define('LLVMAssemblyFile', 0)
LLVMObjectFile = LLVMCodeGenFileType.define('LLVMObjectFile', 1)

try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMLinkInMCJIT:=dll.LLVMLinkInMCJIT).restype, LLVMLinkInMCJIT.argtypes = None, []
except AttributeError: pass

try: (LLVMLinkInInterpreter:=dll.LLVMLinkInInterpreter).restype, LLVMLinkInInterpreter.argtypes = None, []
except AttributeError: pass

class struct_LLVMOpaqueGenericValue(Struct): pass
LLVMGenericValueRef = ctypes.POINTER(struct_LLVMOpaqueGenericValue)
class struct_LLVMOpaqueExecutionEngine(Struct): pass
LLVMExecutionEngineRef = ctypes.POINTER(struct_LLVMOpaqueExecutionEngine)
class struct_LLVMOpaqueMCJITMemoryManager(Struct): pass
LLVMMCJITMemoryManagerRef = ctypes.POINTER(struct_LLVMOpaqueMCJITMemoryManager)
class struct_LLVMMCJITCompilerOptions(Struct): pass
struct_LLVMMCJITCompilerOptions._fields_ = [
  ('OptLevel', ctypes.c_uint32),
  ('CodeModel', LLVMCodeModel),
  ('NoFramePointerElim', LLVMBool),
  ('EnableFastISel', LLVMBool),
  ('MCJMM', LLVMMCJITMemoryManagerRef),
]
try: (LLVMCreateGenericValueOfInt:=dll.LLVMCreateGenericValueOfInt).restype, LLVMCreateGenericValueOfInt.argtypes = LLVMGenericValueRef, [LLVMTypeRef, ctypes.c_uint64, LLVMBool]
except AttributeError: pass

try: (LLVMCreateGenericValueOfPointer:=dll.LLVMCreateGenericValueOfPointer).restype, LLVMCreateGenericValueOfPointer.argtypes = LLVMGenericValueRef, [ctypes.c_void_p]
except AttributeError: pass

try: (LLVMCreateGenericValueOfFloat:=dll.LLVMCreateGenericValueOfFloat).restype, LLVMCreateGenericValueOfFloat.argtypes = LLVMGenericValueRef, [LLVMTypeRef, ctypes.c_double]
except AttributeError: pass

try: (LLVMGenericValueIntWidth:=dll.LLVMGenericValueIntWidth).restype, LLVMGenericValueIntWidth.argtypes = ctypes.c_uint32, [LLVMGenericValueRef]
except AttributeError: pass

try: (LLVMGenericValueToInt:=dll.LLVMGenericValueToInt).restype, LLVMGenericValueToInt.argtypes = ctypes.c_uint64, [LLVMGenericValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGenericValueToPointer:=dll.LLVMGenericValueToPointer).restype, LLVMGenericValueToPointer.argtypes = ctypes.c_void_p, [LLVMGenericValueRef]
except AttributeError: pass

try: (LLVMGenericValueToFloat:=dll.LLVMGenericValueToFloat).restype, LLVMGenericValueToFloat.argtypes = ctypes.c_double, [LLVMTypeRef, LLVMGenericValueRef]
except AttributeError: pass

try: (LLVMDisposeGenericValue:=dll.LLVMDisposeGenericValue).restype, LLVMDisposeGenericValue.argtypes = None, [LLVMGenericValueRef]
except AttributeError: pass

try: (LLVMCreateExecutionEngineForModule:=dll.LLVMCreateExecutionEngineForModule).restype, LLVMCreateExecutionEngineForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMCreateInterpreterForModule:=dll.LLVMCreateInterpreterForModule).restype, LLVMCreateInterpreterForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMCreateJITCompilerForModule:=dll.LLVMCreateJITCompilerForModule).restype, LLVMCreateJITCompilerForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMInitializeMCJITCompilerOptions:=dll.LLVMInitializeMCJITCompilerOptions).restype, LLVMInitializeMCJITCompilerOptions.argtypes = None, [ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t]
except AttributeError: pass

try: (LLVMCreateMCJITCompilerForModule:=dll.LLVMCreateMCJITCompilerForModule).restype, LLVMCreateMCJITCompilerForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMDisposeExecutionEngine:=dll.LLVMDisposeExecutionEngine).restype, LLVMDisposeExecutionEngine.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMRunStaticConstructors:=dll.LLVMRunStaticConstructors).restype, LLVMRunStaticConstructors.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMRunStaticDestructors:=dll.LLVMRunStaticDestructors).restype, LLVMRunStaticDestructors.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMRunFunctionAsMain:=dll.LLVMRunFunctionAsMain).restype, LLVMRunFunctionAsMain.argtypes = ctypes.c_int32, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMRunFunction:=dll.LLVMRunFunction).restype, LLVMRunFunction.argtypes = LLVMGenericValueRef, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(LLVMGenericValueRef)]
except AttributeError: pass

try: (LLVMFreeMachineCodeForFunction:=dll.LLVMFreeMachineCodeForFunction).restype, LLVMFreeMachineCodeForFunction.argtypes = None, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMAddModule:=dll.LLVMAddModule).restype, LLVMAddModule.argtypes = None, [LLVMExecutionEngineRef, LLVMModuleRef]
except AttributeError: pass

try: (LLVMRemoveModule:=dll.LLVMRemoveModule).restype, LLVMRemoveModule.argtypes = LLVMBool, [LLVMExecutionEngineRef, LLVMModuleRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMFindFunction:=dll.LLVMFindFunction).restype, LLVMFindFunction.argtypes = LLVMBool, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

try: (LLVMRecompileAndRelinkFunction:=dll.LLVMRecompileAndRelinkFunction).restype, LLVMRecompileAndRelinkFunction.argtypes = ctypes.c_void_p, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetExecutionEngineTargetData:=dll.LLVMGetExecutionEngineTargetData).restype, LLVMGetExecutionEngineTargetData.argtypes = LLVMTargetDataRef, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMGetExecutionEngineTargetMachine:=dll.LLVMGetExecutionEngineTargetMachine).restype, LLVMGetExecutionEngineTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMAddGlobalMapping:=dll.LLVMAddGlobalMapping).restype, LLVMAddGlobalMapping.argtypes = None, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMGetPointerToGlobal:=dll.LLVMGetPointerToGlobal).restype, LLVMGetPointerToGlobal.argtypes = ctypes.c_void_p, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetGlobalValueAddress:=dll.LLVMGetGlobalValueAddress).restype, LLVMGetGlobalValueAddress.argtypes = uint64_t, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetFunctionAddress:=dll.LLVMGetFunctionAddress).restype, LLVMGetFunctionAddress.argtypes = uint64_t, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMExecutionEngineGetErrMsg:=dll.LLVMExecutionEngineGetErrMsg).restype, LLVMExecutionEngineGetErrMsg.argtypes = LLVMBool, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

LLVMMemoryManagerAllocateCodeSectionCallback = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_ubyte), ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char))
LLVMMemoryManagerAllocateDataSectionCallback = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_ubyte), ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32)
LLVMMemoryManagerFinalizeMemoryCallback = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
LLVMMemoryManagerDestroyCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
try: (LLVMCreateSimpleMCJITMemoryManager:=dll.LLVMCreateSimpleMCJITMemoryManager).restype, LLVMCreateSimpleMCJITMemoryManager.argtypes = LLVMMCJITMemoryManagerRef, [ctypes.c_void_p, LLVMMemoryManagerAllocateCodeSectionCallback, LLVMMemoryManagerAllocateDataSectionCallback, LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback]
except AttributeError: pass

try: (LLVMDisposeMCJITMemoryManager:=dll.LLVMDisposeMCJITMemoryManager).restype, LLVMDisposeMCJITMemoryManager.argtypes = None, [LLVMMCJITMemoryManagerRef]
except AttributeError: pass

class struct_LLVMOpaqueJITEventListener(Struct): pass
LLVMJITEventListenerRef = ctypes.POINTER(struct_LLVMOpaqueJITEventListener)
try: (LLVMCreateGDBRegistrationListener:=dll.LLVMCreateGDBRegistrationListener).restype, LLVMCreateGDBRegistrationListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

try: (LLVMCreateIntelJITEventListener:=dll.LLVMCreateIntelJITEventListener).restype, LLVMCreateIntelJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

try: (LLVMCreateOProfileJITEventListener:=dll.LLVMCreateOProfileJITEventListener).restype, LLVMCreateOProfileJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

try: (LLVMCreatePerfJITEventListener:=dll.LLVMCreatePerfJITEventListener).restype, LLVMCreatePerfJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMParseIRInContext:=dll.LLVMParseIRInContext).restype, LLVMParseIRInContext.argtypes = LLVMBool, [LLVMContextRef, LLVMMemoryBufferRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueExecutionSession(Struct): pass
LLVMOrcExecutionSessionRef = ctypes.POINTER(struct_LLVMOrcOpaqueExecutionSession)
LLVMOrcErrorReporterFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(struct_LLVMOpaqueError))
try: (LLVMOrcExecutionSessionSetErrorReporter:=dll.LLVMOrcExecutionSessionSetErrorReporter).restype, LLVMOrcExecutionSessionSetErrorReporter.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcErrorReporterFunction, ctypes.c_void_p]
except AttributeError: pass

class struct_LLVMOrcOpaqueSymbolStringPool(Struct): pass
LLVMOrcSymbolStringPoolRef = ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPool)
try: (LLVMOrcExecutionSessionGetSymbolStringPool:=dll.LLVMOrcExecutionSessionGetSymbolStringPool).restype, LLVMOrcExecutionSessionGetSymbolStringPool.argtypes = LLVMOrcSymbolStringPoolRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

try: (LLVMOrcSymbolStringPoolClearDeadEntries:=dll.LLVMOrcSymbolStringPoolClearDeadEntries).restype, LLVMOrcSymbolStringPoolClearDeadEntries.argtypes = None, [LLVMOrcSymbolStringPoolRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueSymbolStringPoolEntry(Struct): pass
LLVMOrcSymbolStringPoolEntryRef = ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry)
try: (LLVMOrcExecutionSessionIntern:=dll.LLVMOrcExecutionSessionIntern).restype, LLVMOrcExecutionSessionIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

LLVMOrcLookupKind = CEnum(ctypes.c_uint32)
LLVMOrcLookupKindStatic = LLVMOrcLookupKind.define('LLVMOrcLookupKindStatic', 0)
LLVMOrcLookupKindDLSym = LLVMOrcLookupKind.define('LLVMOrcLookupKindDLSym', 1)

class LLVMOrcCJITDylibSearchOrderElement(Struct): pass
class struct_LLVMOrcOpaqueJITDylib(Struct): pass
LLVMOrcJITDylibRef = ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib)
LLVMOrcJITDylibLookupFlags = CEnum(ctypes.c_uint32)
LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly = LLVMOrcJITDylibLookupFlags.define('LLVMOrcJITDylibLookupFlagsMatchExportedSymbolsOnly', 0)
LLVMOrcJITDylibLookupFlagsMatchAllSymbols = LLVMOrcJITDylibLookupFlags.define('LLVMOrcJITDylibLookupFlagsMatchAllSymbols', 1)

LLVMOrcCJITDylibSearchOrderElement._fields_ = [
  ('JD', LLVMOrcJITDylibRef),
  ('JDLookupFlags', LLVMOrcJITDylibLookupFlags),
]
LLVMOrcCJITDylibSearchOrder = ctypes.POINTER(LLVMOrcCJITDylibSearchOrderElement)
class LLVMOrcCLookupSetElement(Struct): pass
LLVMOrcSymbolLookupFlags = CEnum(ctypes.c_uint32)
LLVMOrcSymbolLookupFlagsRequiredSymbol = LLVMOrcSymbolLookupFlags.define('LLVMOrcSymbolLookupFlagsRequiredSymbol', 0)
LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol = LLVMOrcSymbolLookupFlags.define('LLVMOrcSymbolLookupFlagsWeaklyReferencedSymbol', 1)

LLVMOrcCLookupSetElement._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('LookupFlags', LLVMOrcSymbolLookupFlags),
]
LLVMOrcCLookupSet = ctypes.POINTER(LLVMOrcCLookupSetElement)
class LLVMOrcCSymbolMapPair(Struct): pass
class LLVMJITEvaluatedSymbol(Struct): pass
LLVMOrcExecutorAddress = ctypes.c_uint64
class LLVMJITSymbolFlags(Struct): pass
LLVMJITSymbolFlags._fields_ = [
  ('GenericFlags', uint8_t),
  ('TargetFlags', uint8_t),
]
LLVMJITEvaluatedSymbol._fields_ = [
  ('Address', LLVMOrcExecutorAddress),
  ('Flags', LLVMJITSymbolFlags),
]
LLVMOrcCSymbolMapPair._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('Sym', LLVMJITEvaluatedSymbol),
]
LLVMOrcExecutionSessionLookupHandleResultFunction = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_LLVMOpaqueError), ctypes.POINTER(LLVMOrcCSymbolMapPair), ctypes.c_uint64, ctypes.c_void_p)
try: (LLVMOrcExecutionSessionLookup:=dll.LLVMOrcExecutionSessionLookup).restype, LLVMOrcExecutionSessionLookup.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcLookupKind, LLVMOrcCJITDylibSearchOrder, size_t, LLVMOrcCLookupSet, size_t, LLVMOrcExecutionSessionLookupHandleResultFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcRetainSymbolStringPoolEntry:=dll.LLVMOrcRetainSymbolStringPoolEntry).restype, LLVMOrcRetainSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcReleaseSymbolStringPoolEntry:=dll.LLVMOrcReleaseSymbolStringPoolEntry).restype, LLVMOrcReleaseSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcSymbolStringPoolEntryStr:=dll.LLVMOrcSymbolStringPoolEntryStr).restype, LLVMOrcSymbolStringPoolEntryStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueResourceTracker(Struct): pass
LLVMOrcResourceTrackerRef = ctypes.POINTER(struct_LLVMOrcOpaqueResourceTracker)
try: (LLVMOrcReleaseResourceTracker:=dll.LLVMOrcReleaseResourceTracker).restype, LLVMOrcReleaseResourceTracker.argtypes = None, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcResourceTrackerTransferTo:=dll.LLVMOrcResourceTrackerTransferTo).restype, LLVMOrcResourceTrackerTransferTo.argtypes = None, [LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcResourceTrackerRemove:=dll.LLVMOrcResourceTrackerRemove).restype, LLVMOrcResourceTrackerRemove.argtypes = LLVMErrorRef, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueDefinitionGenerator(Struct): pass
LLVMOrcDefinitionGeneratorRef = ctypes.POINTER(struct_LLVMOrcOpaqueDefinitionGenerator)
try: (LLVMOrcDisposeDefinitionGenerator:=dll.LLVMOrcDisposeDefinitionGenerator).restype, LLVMOrcDisposeDefinitionGenerator.argtypes = None, [LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueMaterializationUnit(Struct): pass
LLVMOrcMaterializationUnitRef = ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationUnit)
try: (LLVMOrcDisposeMaterializationUnit:=dll.LLVMOrcDisposeMaterializationUnit).restype, LLVMOrcDisposeMaterializationUnit.argtypes = None, [LLVMOrcMaterializationUnitRef]
except AttributeError: pass

class LLVMOrcCSymbolFlagsMapPair(Struct): pass
LLVMOrcCSymbolFlagsMapPair._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('Flags', LLVMJITSymbolFlags),
]
LLVMOrcCSymbolFlagsMapPairs = ctypes.POINTER(LLVMOrcCSymbolFlagsMapPair)
class struct_LLVMOrcOpaqueMaterializationResponsibility(Struct): pass
LLVMOrcMaterializationUnitMaterializeFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility))
LLVMOrcMaterializationUnitDiscardFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib), ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry))
LLVMOrcMaterializationUnitDestroyFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
try: (LLVMOrcCreateCustomMaterializationUnit:=dll.LLVMOrcCreateCustomMaterializationUnit).restype, LLVMOrcCreateCustomMaterializationUnit.argtypes = LLVMOrcMaterializationUnitRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, LLVMOrcCSymbolFlagsMapPairs, size_t, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcMaterializationUnitMaterializeFunction, LLVMOrcMaterializationUnitDiscardFunction, LLVMOrcMaterializationUnitDestroyFunction]
except AttributeError: pass

LLVMOrcCSymbolMapPairs = ctypes.POINTER(LLVMOrcCSymbolMapPair)
try: (LLVMOrcAbsoluteSymbols:=dll.LLVMOrcAbsoluteSymbols).restype, LLVMOrcAbsoluteSymbols.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

class struct_LLVMOrcOpaqueLazyCallThroughManager(Struct): pass
LLVMOrcLazyCallThroughManagerRef = ctypes.POINTER(struct_LLVMOrcOpaqueLazyCallThroughManager)
class struct_LLVMOrcOpaqueIndirectStubsManager(Struct): pass
LLVMOrcIndirectStubsManagerRef = ctypes.POINTER(struct_LLVMOrcOpaqueIndirectStubsManager)
class LLVMOrcCSymbolAliasMapPair(Struct): pass
class LLVMOrcCSymbolAliasMapEntry(Struct): pass
LLVMOrcCSymbolAliasMapEntry._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('Flags', LLVMJITSymbolFlags),
]
LLVMOrcCSymbolAliasMapPair._fields_ = [
  ('Name', LLVMOrcSymbolStringPoolEntryRef),
  ('Entry', LLVMOrcCSymbolAliasMapEntry),
]
LLVMOrcCSymbolAliasMapPairs = ctypes.POINTER(LLVMOrcCSymbolAliasMapPair)
try: (LLVMOrcLazyReexports:=dll.LLVMOrcLazyReexports).restype, LLVMOrcLazyReexports.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcLazyCallThroughManagerRef, LLVMOrcIndirectStubsManagerRef, LLVMOrcJITDylibRef, LLVMOrcCSymbolAliasMapPairs, size_t]
except AttributeError: pass

LLVMOrcMaterializationResponsibilityRef = ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility)
try: (LLVMOrcDisposeMaterializationResponsibility:=dll.LLVMOrcDisposeMaterializationResponsibility).restype, LLVMOrcDisposeMaterializationResponsibility.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetTargetDylib:=dll.LLVMOrcMaterializationResponsibilityGetTargetDylib).restype, LLVMOrcMaterializationResponsibilityGetTargetDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetExecutionSession:=dll.LLVMOrcMaterializationResponsibilityGetExecutionSession).restype, LLVMOrcMaterializationResponsibilityGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetSymbols:=dll.LLVMOrcMaterializationResponsibilityGetSymbols).restype, LLVMOrcMaterializationResponsibilityGetSymbols.argtypes = LLVMOrcCSymbolFlagsMapPairs, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMOrcDisposeCSymbolFlagsMap:=dll.LLVMOrcDisposeCSymbolFlagsMap).restype, LLVMOrcDisposeCSymbolFlagsMap.argtypes = None, [LLVMOrcCSymbolFlagsMapPairs]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetInitializerSymbol:=dll.LLVMOrcMaterializationResponsibilityGetInitializerSymbol).restype, LLVMOrcMaterializationResponsibilityGetInitializerSymbol.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetRequestedSymbols:=dll.LLVMOrcMaterializationResponsibilityGetRequestedSymbols).restype, LLVMOrcMaterializationResponsibilityGetRequestedSymbols.argtypes = ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMOrcDisposeSymbols:=dll.LLVMOrcDisposeSymbols).restype, LLVMOrcDisposeSymbols.argtypes = None, [ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityNotifyResolved:=dll.LLVMOrcMaterializationResponsibilityNotifyResolved).restype, LLVMOrcMaterializationResponsibilityNotifyResolved.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

class LLVMOrcCSymbolDependenceGroup(Struct): pass
class LLVMOrcCSymbolsList(Struct): pass
LLVMOrcCSymbolsList._fields_ = [
  ('Symbols', ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)),
  ('Length', size_t),
]
class LLVMOrcCDependenceMapPair(Struct): pass
LLVMOrcCDependenceMapPair._fields_ = [
  ('JD', LLVMOrcJITDylibRef),
  ('Names', LLVMOrcCSymbolsList),
]
LLVMOrcCDependenceMapPairs = ctypes.POINTER(LLVMOrcCDependenceMapPair)
LLVMOrcCSymbolDependenceGroup._fields_ = [
  ('Symbols', LLVMOrcCSymbolsList),
  ('Dependencies', LLVMOrcCDependenceMapPairs),
  ('NumDependencies', size_t),
]
try: (LLVMOrcMaterializationResponsibilityNotifyEmitted:=dll.LLVMOrcMaterializationResponsibilityNotifyEmitted).restype, LLVMOrcMaterializationResponsibilityNotifyEmitted.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcCSymbolDependenceGroup), size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityDefineMaterializing:=dll.LLVMOrcMaterializationResponsibilityDefineMaterializing).restype, LLVMOrcMaterializationResponsibilityDefineMaterializing.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolFlagsMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityFailMaterialization:=dll.LLVMOrcMaterializationResponsibilityFailMaterialization).restype, LLVMOrcMaterializationResponsibilityFailMaterialization.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityReplace:=dll.LLVMOrcMaterializationResponsibilityReplace).restype, LLVMOrcMaterializationResponsibilityReplace.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityDelegate:=dll.LLVMOrcMaterializationResponsibilityDelegate).restype, LLVMOrcMaterializationResponsibilityDelegate.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), size_t, ctypes.POINTER(LLVMOrcMaterializationResponsibilityRef)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionCreateBareJITDylib:=dll.LLVMOrcExecutionSessionCreateBareJITDylib).restype, LLVMOrcExecutionSessionCreateBareJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionCreateJITDylib:=dll.LLVMOrcExecutionSessionCreateJITDylib).restype, LLVMOrcExecutionSessionCreateJITDylib.argtypes = LLVMErrorRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(LLVMOrcJITDylibRef), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionGetJITDylibByName:=dll.LLVMOrcExecutionSessionGetJITDylibByName).restype, LLVMOrcExecutionSessionGetJITDylibByName.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcJITDylibCreateResourceTracker:=dll.LLVMOrcJITDylibCreateResourceTracker).restype, LLVMOrcJITDylibCreateResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibGetDefaultResourceTracker:=dll.LLVMOrcJITDylibGetDefaultResourceTracker).restype, LLVMOrcJITDylibGetDefaultResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibDefine:=dll.LLVMOrcJITDylibDefine).restype, LLVMOrcJITDylibDefine.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcJITDylibClear:=dll.LLVMOrcJITDylibClear).restype, LLVMOrcJITDylibClear.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibAddGenerator:=dll.LLVMOrcJITDylibAddGenerator).restype, LLVMOrcJITDylibAddGenerator.argtypes = None, [LLVMOrcJITDylibRef, LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueLookupState(Struct): pass
LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.POINTER(struct_LLVMOrcOpaqueDefinitionGenerator), ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueLookupState)), LLVMOrcLookupKind, ctypes.POINTER(struct_LLVMOrcOpaqueJITDylib), LLVMOrcJITDylibLookupFlags, ctypes.POINTER(LLVMOrcCLookupSetElement), ctypes.c_uint64)
LLVMOrcDisposeCAPIDefinitionGeneratorFunction = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
try: (LLVMOrcCreateCustomCAPIDefinitionGenerator:=dll.LLVMOrcCreateCustomCAPIDefinitionGenerator).restype, LLVMOrcCreateCustomCAPIDefinitionGenerator.argtypes = LLVMOrcDefinitionGeneratorRef, [LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction, ctypes.c_void_p, LLVMOrcDisposeCAPIDefinitionGeneratorFunction]
except AttributeError: pass

LLVMOrcLookupStateRef = ctypes.POINTER(struct_LLVMOrcOpaqueLookupState)
try: (LLVMOrcLookupStateContinueLookup:=dll.LLVMOrcLookupStateContinueLookup).restype, LLVMOrcLookupStateContinueLookup.argtypes = None, [LLVMOrcLookupStateRef, LLVMErrorRef]
except AttributeError: pass

LLVMOrcSymbolPredicate = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(struct_LLVMOrcOpaqueSymbolStringPoolEntry))
try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForPath).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.POINTER(ctypes.c_char), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

class struct_LLVMOrcOpaqueObjectLayer(Struct): pass
LLVMOrcObjectLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueObjectLayer)
try: (LLVMOrcCreateStaticLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateStaticLibrarySearchGeneratorForPath).restype, LLVMOrcCreateStaticLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), LLVMOrcObjectLayerRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_LLVMOrcOpaqueThreadSafeContext(Struct): pass
LLVMOrcThreadSafeContextRef = ctypes.POINTER(struct_LLVMOrcOpaqueThreadSafeContext)
try: (LLVMOrcCreateNewThreadSafeContext:=dll.LLVMOrcCreateNewThreadSafeContext).restype, LLVMOrcCreateNewThreadSafeContext.argtypes = LLVMOrcThreadSafeContextRef, []
except AttributeError: pass

try: (LLVMOrcThreadSafeContextGetContext:=dll.LLVMOrcThreadSafeContextGetContext).restype, LLVMOrcThreadSafeContextGetContext.argtypes = LLVMContextRef, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcDisposeThreadSafeContext:=dll.LLVMOrcDisposeThreadSafeContext).restype, LLVMOrcDisposeThreadSafeContext.argtypes = None, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueThreadSafeModule(Struct): pass
LLVMOrcThreadSafeModuleRef = ctypes.POINTER(struct_LLVMOrcOpaqueThreadSafeModule)
try: (LLVMOrcCreateNewThreadSafeModule:=dll.LLVMOrcCreateNewThreadSafeModule).restype, LLVMOrcCreateNewThreadSafeModule.argtypes = LLVMOrcThreadSafeModuleRef, [LLVMModuleRef, LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcDisposeThreadSafeModule:=dll.LLVMOrcDisposeThreadSafeModule).restype, LLVMOrcDisposeThreadSafeModule.argtypes = None, [LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

LLVMOrcGenericIRModuleOperationFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.c_void_p, ctypes.POINTER(struct_LLVMOpaqueModule))
try: (LLVMOrcThreadSafeModuleWithModuleDo:=dll.LLVMOrcThreadSafeModuleWithModuleDo).restype, LLVMOrcThreadSafeModuleWithModuleDo.argtypes = LLVMErrorRef, [LLVMOrcThreadSafeModuleRef, LLVMOrcGenericIRModuleOperationFunction, ctypes.c_void_p]
except AttributeError: pass

class struct_LLVMOrcOpaqueJITTargetMachineBuilder(Struct): pass
LLVMOrcJITTargetMachineBuilderRef = ctypes.POINTER(struct_LLVMOrcOpaqueJITTargetMachineBuilder)
try: (LLVMOrcJITTargetMachineBuilderDetectHost:=dll.LLVMOrcJITTargetMachineBuilderDetectHost).restype, LLVMOrcJITTargetMachineBuilderDetectHost.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcJITTargetMachineBuilderRef)]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine:=dll.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine).restype, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.argtypes = LLVMOrcJITTargetMachineBuilderRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMOrcDisposeJITTargetMachineBuilder:=dll.LLVMOrcDisposeJITTargetMachineBuilder).restype, LLVMOrcDisposeJITTargetMachineBuilder.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderGetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderGetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderGetTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderSetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderSetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderSetTargetTriple.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcObjectLayerAddObjectFile:=dll.LLVMOrcObjectLayerAddObjectFile).restype, LLVMOrcObjectLayerAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcObjectLayerAddObjectFileWithRT:=dll.LLVMOrcObjectLayerAddObjectFileWithRT).restype, LLVMOrcObjectLayerAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcObjectLayerEmit:=dll.LLVMOrcObjectLayerEmit).restype, LLVMOrcObjectLayerEmit.argtypes = None, [LLVMOrcObjectLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcDisposeObjectLayer:=dll.LLVMOrcDisposeObjectLayer).restype, LLVMOrcDisposeObjectLayer.argtypes = None, [LLVMOrcObjectLayerRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueIRTransformLayer(Struct): pass
LLVMOrcIRTransformLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueIRTransformLayer)
try: (LLVMOrcIRTransformLayerEmit:=dll.LLVMOrcIRTransformLayerEmit).restype, LLVMOrcIRTransformLayerEmit.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

LLVMOrcIRTransformLayerTransformFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(struct_LLVMOrcOpaqueThreadSafeModule)), ctypes.POINTER(struct_LLVMOrcOpaqueMaterializationResponsibility))
try: (LLVMOrcIRTransformLayerSetTransform:=dll.LLVMOrcIRTransformLayerSetTransform).restype, LLVMOrcIRTransformLayerSetTransform.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcIRTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

class struct_LLVMOrcOpaqueObjectTransformLayer(Struct): pass
LLVMOrcObjectTransformLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueObjectTransformLayer)
LLVMOrcObjectTransformLayerTransformFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOpaqueError), ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(struct_LLVMOpaqueMemoryBuffer)))
try: (LLVMOrcObjectTransformLayerSetTransform:=dll.LLVMOrcObjectTransformLayerSetTransform).restype, LLVMOrcObjectTransformLayerSetTransform.argtypes = None, [LLVMOrcObjectTransformLayerRef, LLVMOrcObjectTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateLocalIndirectStubsManager:=dll.LLVMOrcCreateLocalIndirectStubsManager).restype, LLVMOrcCreateLocalIndirectStubsManager.argtypes = LLVMOrcIndirectStubsManagerRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcDisposeIndirectStubsManager:=dll.LLVMOrcDisposeIndirectStubsManager).restype, LLVMOrcDisposeIndirectStubsManager.argtypes = None, [LLVMOrcIndirectStubsManagerRef]
except AttributeError: pass

LLVMOrcJITTargetAddress = ctypes.c_uint64
try: (LLVMOrcCreateLocalLazyCallThroughManager:=dll.LLVMOrcCreateLocalLazyCallThroughManager).restype, LLVMOrcCreateLocalLazyCallThroughManager.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char), LLVMOrcExecutionSessionRef, LLVMOrcJITTargetAddress, ctypes.POINTER(LLVMOrcLazyCallThroughManagerRef)]
except AttributeError: pass

try: (LLVMOrcDisposeLazyCallThroughManager:=dll.LLVMOrcDisposeLazyCallThroughManager).restype, LLVMOrcDisposeLazyCallThroughManager.argtypes = None, [LLVMOrcLazyCallThroughManagerRef]
except AttributeError: pass

class struct_LLVMOrcOpaqueDumpObjects(Struct): pass
LLVMOrcDumpObjectsRef = ctypes.POINTER(struct_LLVMOrcOpaqueDumpObjects)
try: (LLVMOrcCreateDumpObjects:=dll.LLVMOrcCreateDumpObjects).restype, LLVMOrcCreateDumpObjects.argtypes = LLVMOrcDumpObjectsRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcDisposeDumpObjects:=dll.LLVMOrcDisposeDumpObjects).restype, LLVMOrcDisposeDumpObjects.argtypes = None, [LLVMOrcDumpObjectsRef]
except AttributeError: pass

try: (LLVMOrcDumpObjects_CallOperator:=dll.LLVMOrcDumpObjects_CallOperator).restype, LLVMOrcDumpObjects_CallOperator.argtypes = LLVMErrorRef, [LLVMOrcDumpObjectsRef, ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction = ctypes.CFUNCTYPE(ctypes.POINTER(struct_LLVMOrcOpaqueObjectLayer), ctypes.c_void_p, ctypes.POINTER(struct_LLVMOrcOpaqueExecutionSession), ctypes.POINTER(ctypes.c_char))
class struct_LLVMOrcOpaqueLLJITBuilder(Struct): pass
LLVMOrcLLJITBuilderRef = ctypes.POINTER(struct_LLVMOrcOpaqueLLJITBuilder)
class struct_LLVMOrcOpaqueLLJIT(Struct): pass
LLVMOrcLLJITRef = ctypes.POINTER(struct_LLVMOrcOpaqueLLJIT)
try: (LLVMOrcCreateLLJITBuilder:=dll.LLVMOrcCreateLLJITBuilder).restype, LLVMOrcCreateLLJITBuilder.argtypes = LLVMOrcLLJITBuilderRef, []
except AttributeError: pass

try: (LLVMOrcDisposeLLJITBuilder:=dll.LLVMOrcDisposeLLJITBuilder).restype, LLVMOrcDisposeLLJITBuilder.argtypes = None, [LLVMOrcLLJITBuilderRef]
except AttributeError: pass

try: (LLVMOrcLLJITBuilderSetJITTargetMachineBuilder:=dll.LLVMOrcLLJITBuilderSetJITTargetMachineBuilder).restype, LLVMOrcLLJITBuilderSetJITTargetMachineBuilder.argtypes = None, [LLVMOrcLLJITBuilderRef, LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator:=dll.LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator).restype, LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator.argtypes = None, [LLVMOrcLLJITBuilderRef, LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateLLJIT:=dll.LLVMOrcCreateLLJIT).restype, LLVMOrcCreateLLJIT.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcLLJITRef), LLVMOrcLLJITBuilderRef]
except AttributeError: pass

try: (LLVMOrcDisposeLLJIT:=dll.LLVMOrcDisposeLLJIT).restype, LLVMOrcDisposeLLJIT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetExecutionSession:=dll.LLVMOrcLLJITGetExecutionSession).restype, LLVMOrcLLJITGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetMainJITDylib:=dll.LLVMOrcLLJITGetMainJITDylib).restype, LLVMOrcLLJITGetMainJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetTripleString:=dll.LLVMOrcLLJITGetTripleString).restype, LLVMOrcLLJITGetTripleString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetGlobalPrefix:=dll.LLVMOrcLLJITGetGlobalPrefix).restype, LLVMOrcLLJITGetGlobalPrefix.argtypes = ctypes.c_char, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITMangleAndIntern:=dll.LLVMOrcLLJITMangleAndIntern).restype, LLVMOrcLLJITMangleAndIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcLLJITRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcLLJITAddObjectFile:=dll.LLVMOrcLLJITAddObjectFile).restype, LLVMOrcLLJITAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcLLJITAddObjectFileWithRT:=dll.LLVMOrcLLJITAddObjectFileWithRT).restype, LLVMOrcLLJITAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcLLJITAddLLVMIRModule:=dll.LLVMOrcLLJITAddLLVMIRModule).restype, LLVMOrcLLJITAddLLVMIRModule.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcLLJITAddLLVMIRModuleWithRT:=dll.LLVMOrcLLJITAddLLVMIRModuleWithRT).restype, LLVMOrcLLJITAddLLVMIRModuleWithRT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcLLJITLookup:=dll.LLVMOrcLLJITLookup).restype, LLVMOrcLLJITLookup.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, ctypes.POINTER(LLVMOrcExecutorAddress), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcLLJITGetObjLinkingLayer:=dll.LLVMOrcLLJITGetObjLinkingLayer).restype, LLVMOrcLLJITGetObjLinkingLayer.argtypes = LLVMOrcObjectLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetObjTransformLayer:=dll.LLVMOrcLLJITGetObjTransformLayer).restype, LLVMOrcLLJITGetObjTransformLayer.argtypes = LLVMOrcObjectTransformLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetIRTransformLayer:=dll.LLVMOrcLLJITGetIRTransformLayer).restype, LLVMOrcLLJITGetIRTransformLayer.argtypes = LLVMOrcIRTransformLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetDataLayoutStr:=dll.LLVMOrcLLJITGetDataLayoutStr).restype, LLVMOrcLLJITGetDataLayoutStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMOrcExecutionSessionSetErrorReporter:=dll.LLVMOrcExecutionSessionSetErrorReporter).restype, LLVMOrcExecutionSessionSetErrorReporter.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcErrorReporterFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcExecutionSessionGetSymbolStringPool:=dll.LLVMOrcExecutionSessionGetSymbolStringPool).restype, LLVMOrcExecutionSessionGetSymbolStringPool.argtypes = LLVMOrcSymbolStringPoolRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

try: (LLVMOrcSymbolStringPoolClearDeadEntries:=dll.LLVMOrcSymbolStringPoolClearDeadEntries).restype, LLVMOrcSymbolStringPoolClearDeadEntries.argtypes = None, [LLVMOrcSymbolStringPoolRef]
except AttributeError: pass

try: (LLVMOrcExecutionSessionIntern:=dll.LLVMOrcExecutionSessionIntern).restype, LLVMOrcExecutionSessionIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionLookup:=dll.LLVMOrcExecutionSessionLookup).restype, LLVMOrcExecutionSessionLookup.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcLookupKind, LLVMOrcCJITDylibSearchOrder, size_t, LLVMOrcCLookupSet, size_t, LLVMOrcExecutionSessionLookupHandleResultFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcRetainSymbolStringPoolEntry:=dll.LLVMOrcRetainSymbolStringPoolEntry).restype, LLVMOrcRetainSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcReleaseSymbolStringPoolEntry:=dll.LLVMOrcReleaseSymbolStringPoolEntry).restype, LLVMOrcReleaseSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcSymbolStringPoolEntryStr:=dll.LLVMOrcSymbolStringPoolEntryStr).restype, LLVMOrcSymbolStringPoolEntryStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcReleaseResourceTracker:=dll.LLVMOrcReleaseResourceTracker).restype, LLVMOrcReleaseResourceTracker.argtypes = None, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcResourceTrackerTransferTo:=dll.LLVMOrcResourceTrackerTransferTo).restype, LLVMOrcResourceTrackerTransferTo.argtypes = None, [LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcResourceTrackerRemove:=dll.LLVMOrcResourceTrackerRemove).restype, LLVMOrcResourceTrackerRemove.argtypes = LLVMErrorRef, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcDisposeDefinitionGenerator:=dll.LLVMOrcDisposeDefinitionGenerator).restype, LLVMOrcDisposeDefinitionGenerator.argtypes = None, [LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

try: (LLVMOrcDisposeMaterializationUnit:=dll.LLVMOrcDisposeMaterializationUnit).restype, LLVMOrcDisposeMaterializationUnit.argtypes = None, [LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcCreateCustomMaterializationUnit:=dll.LLVMOrcCreateCustomMaterializationUnit).restype, LLVMOrcCreateCustomMaterializationUnit.argtypes = LLVMOrcMaterializationUnitRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, LLVMOrcCSymbolFlagsMapPairs, size_t, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcMaterializationUnitMaterializeFunction, LLVMOrcMaterializationUnitDiscardFunction, LLVMOrcMaterializationUnitDestroyFunction]
except AttributeError: pass

try: (LLVMOrcAbsoluteSymbols:=dll.LLVMOrcAbsoluteSymbols).restype, LLVMOrcAbsoluteSymbols.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcLazyReexports:=dll.LLVMOrcLazyReexports).restype, LLVMOrcLazyReexports.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcLazyCallThroughManagerRef, LLVMOrcIndirectStubsManagerRef, LLVMOrcJITDylibRef, LLVMOrcCSymbolAliasMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcDisposeMaterializationResponsibility:=dll.LLVMOrcDisposeMaterializationResponsibility).restype, LLVMOrcDisposeMaterializationResponsibility.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetTargetDylib:=dll.LLVMOrcMaterializationResponsibilityGetTargetDylib).restype, LLVMOrcMaterializationResponsibilityGetTargetDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetExecutionSession:=dll.LLVMOrcMaterializationResponsibilityGetExecutionSession).restype, LLVMOrcMaterializationResponsibilityGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetSymbols:=dll.LLVMOrcMaterializationResponsibilityGetSymbols).restype, LLVMOrcMaterializationResponsibilityGetSymbols.argtypes = LLVMOrcCSymbolFlagsMapPairs, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMOrcDisposeCSymbolFlagsMap:=dll.LLVMOrcDisposeCSymbolFlagsMap).restype, LLVMOrcDisposeCSymbolFlagsMap.argtypes = None, [LLVMOrcCSymbolFlagsMapPairs]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetInitializerSymbol:=dll.LLVMOrcMaterializationResponsibilityGetInitializerSymbol).restype, LLVMOrcMaterializationResponsibilityGetInitializerSymbol.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetRequestedSymbols:=dll.LLVMOrcMaterializationResponsibilityGetRequestedSymbols).restype, LLVMOrcMaterializationResponsibilityGetRequestedSymbols.argtypes = ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMOrcDisposeSymbols:=dll.LLVMOrcDisposeSymbols).restype, LLVMOrcDisposeSymbols.argtypes = None, [ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityNotifyResolved:=dll.LLVMOrcMaterializationResponsibilityNotifyResolved).restype, LLVMOrcMaterializationResponsibilityNotifyResolved.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityNotifyEmitted:=dll.LLVMOrcMaterializationResponsibilityNotifyEmitted).restype, LLVMOrcMaterializationResponsibilityNotifyEmitted.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcCSymbolDependenceGroup), size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityDefineMaterializing:=dll.LLVMOrcMaterializationResponsibilityDefineMaterializing).restype, LLVMOrcMaterializationResponsibilityDefineMaterializing.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolFlagsMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityFailMaterialization:=dll.LLVMOrcMaterializationResponsibilityFailMaterialization).restype, LLVMOrcMaterializationResponsibilityFailMaterialization.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityReplace:=dll.LLVMOrcMaterializationResponsibilityReplace).restype, LLVMOrcMaterializationResponsibilityReplace.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityDelegate:=dll.LLVMOrcMaterializationResponsibilityDelegate).restype, LLVMOrcMaterializationResponsibilityDelegate.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), size_t, ctypes.POINTER(LLVMOrcMaterializationResponsibilityRef)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionCreateBareJITDylib:=dll.LLVMOrcExecutionSessionCreateBareJITDylib).restype, LLVMOrcExecutionSessionCreateBareJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionCreateJITDylib:=dll.LLVMOrcExecutionSessionCreateJITDylib).restype, LLVMOrcExecutionSessionCreateJITDylib.argtypes = LLVMErrorRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(LLVMOrcJITDylibRef), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionGetJITDylibByName:=dll.LLVMOrcExecutionSessionGetJITDylibByName).restype, LLVMOrcExecutionSessionGetJITDylibByName.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcJITDylibCreateResourceTracker:=dll.LLVMOrcJITDylibCreateResourceTracker).restype, LLVMOrcJITDylibCreateResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibGetDefaultResourceTracker:=dll.LLVMOrcJITDylibGetDefaultResourceTracker).restype, LLVMOrcJITDylibGetDefaultResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibDefine:=dll.LLVMOrcJITDylibDefine).restype, LLVMOrcJITDylibDefine.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcJITDylibClear:=dll.LLVMOrcJITDylibClear).restype, LLVMOrcJITDylibClear.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibAddGenerator:=dll.LLVMOrcJITDylibAddGenerator).restype, LLVMOrcJITDylibAddGenerator.argtypes = None, [LLVMOrcJITDylibRef, LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

try: (LLVMOrcCreateCustomCAPIDefinitionGenerator:=dll.LLVMOrcCreateCustomCAPIDefinitionGenerator).restype, LLVMOrcCreateCustomCAPIDefinitionGenerator.argtypes = LLVMOrcDefinitionGeneratorRef, [LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction, ctypes.c_void_p, LLVMOrcDisposeCAPIDefinitionGeneratorFunction]
except AttributeError: pass

try: (LLVMOrcLookupStateContinueLookup:=dll.LLVMOrcLookupStateContinueLookup).restype, LLVMOrcLookupStateContinueLookup.argtypes = None, [LLVMOrcLookupStateRef, LLVMErrorRef]
except AttributeError: pass

try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForPath).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.POINTER(ctypes.c_char), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateStaticLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateStaticLibrarySearchGeneratorForPath).restype, LLVMOrcCreateStaticLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), LLVMOrcObjectLayerRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcCreateNewThreadSafeContext:=dll.LLVMOrcCreateNewThreadSafeContext).restype, LLVMOrcCreateNewThreadSafeContext.argtypes = LLVMOrcThreadSafeContextRef, []
except AttributeError: pass

try: (LLVMOrcThreadSafeContextGetContext:=dll.LLVMOrcThreadSafeContextGetContext).restype, LLVMOrcThreadSafeContextGetContext.argtypes = LLVMContextRef, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcDisposeThreadSafeContext:=dll.LLVMOrcDisposeThreadSafeContext).restype, LLVMOrcDisposeThreadSafeContext.argtypes = None, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcCreateNewThreadSafeModule:=dll.LLVMOrcCreateNewThreadSafeModule).restype, LLVMOrcCreateNewThreadSafeModule.argtypes = LLVMOrcThreadSafeModuleRef, [LLVMModuleRef, LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcDisposeThreadSafeModule:=dll.LLVMOrcDisposeThreadSafeModule).restype, LLVMOrcDisposeThreadSafeModule.argtypes = None, [LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcThreadSafeModuleWithModuleDo:=dll.LLVMOrcThreadSafeModuleWithModuleDo).restype, LLVMOrcThreadSafeModuleWithModuleDo.argtypes = LLVMErrorRef, [LLVMOrcThreadSafeModuleRef, LLVMOrcGenericIRModuleOperationFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderDetectHost:=dll.LLVMOrcJITTargetMachineBuilderDetectHost).restype, LLVMOrcJITTargetMachineBuilderDetectHost.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcJITTargetMachineBuilderRef)]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine:=dll.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine).restype, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.argtypes = LLVMOrcJITTargetMachineBuilderRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMOrcDisposeJITTargetMachineBuilder:=dll.LLVMOrcDisposeJITTargetMachineBuilder).restype, LLVMOrcDisposeJITTargetMachineBuilder.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderGetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderGetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderGetTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderSetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderSetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderSetTargetTriple.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcObjectLayerAddObjectFile:=dll.LLVMOrcObjectLayerAddObjectFile).restype, LLVMOrcObjectLayerAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcObjectLayerAddObjectFileWithRT:=dll.LLVMOrcObjectLayerAddObjectFileWithRT).restype, LLVMOrcObjectLayerAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcObjectLayerEmit:=dll.LLVMOrcObjectLayerEmit).restype, LLVMOrcObjectLayerEmit.argtypes = None, [LLVMOrcObjectLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcDisposeObjectLayer:=dll.LLVMOrcDisposeObjectLayer).restype, LLVMOrcDisposeObjectLayer.argtypes = None, [LLVMOrcObjectLayerRef]
except AttributeError: pass

try: (LLVMOrcIRTransformLayerEmit:=dll.LLVMOrcIRTransformLayerEmit).restype, LLVMOrcIRTransformLayerEmit.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcIRTransformLayerSetTransform:=dll.LLVMOrcIRTransformLayerSetTransform).restype, LLVMOrcIRTransformLayerSetTransform.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcIRTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcObjectTransformLayerSetTransform:=dll.LLVMOrcObjectTransformLayerSetTransform).restype, LLVMOrcObjectTransformLayerSetTransform.argtypes = None, [LLVMOrcObjectTransformLayerRef, LLVMOrcObjectTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateLocalIndirectStubsManager:=dll.LLVMOrcCreateLocalIndirectStubsManager).restype, LLVMOrcCreateLocalIndirectStubsManager.argtypes = LLVMOrcIndirectStubsManagerRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcDisposeIndirectStubsManager:=dll.LLVMOrcDisposeIndirectStubsManager).restype, LLVMOrcDisposeIndirectStubsManager.argtypes = None, [LLVMOrcIndirectStubsManagerRef]
except AttributeError: pass

try: (LLVMOrcCreateLocalLazyCallThroughManager:=dll.LLVMOrcCreateLocalLazyCallThroughManager).restype, LLVMOrcCreateLocalLazyCallThroughManager.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char), LLVMOrcExecutionSessionRef, LLVMOrcJITTargetAddress, ctypes.POINTER(LLVMOrcLazyCallThroughManagerRef)]
except AttributeError: pass

try: (LLVMOrcDisposeLazyCallThroughManager:=dll.LLVMOrcDisposeLazyCallThroughManager).restype, LLVMOrcDisposeLazyCallThroughManager.argtypes = None, [LLVMOrcLazyCallThroughManagerRef]
except AttributeError: pass

try: (LLVMOrcCreateDumpObjects:=dll.LLVMOrcCreateDumpObjects).restype, LLVMOrcCreateDumpObjects.argtypes = LLVMOrcDumpObjectsRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcDisposeDumpObjects:=dll.LLVMOrcDisposeDumpObjects).restype, LLVMOrcDisposeDumpObjects.argtypes = None, [LLVMOrcDumpObjectsRef]
except AttributeError: pass

try: (LLVMOrcDumpObjects_CallOperator:=dll.LLVMOrcDumpObjects_CallOperator).restype, LLVMOrcDumpObjects_CallOperator.argtypes = LLVMErrorRef, [LLVMOrcDumpObjectsRef, ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

try: (LLVMOrcCreateLLJITBuilder:=dll.LLVMOrcCreateLLJITBuilder).restype, LLVMOrcCreateLLJITBuilder.argtypes = LLVMOrcLLJITBuilderRef, []
except AttributeError: pass

try: (LLVMOrcDisposeLLJITBuilder:=dll.LLVMOrcDisposeLLJITBuilder).restype, LLVMOrcDisposeLLJITBuilder.argtypes = None, [LLVMOrcLLJITBuilderRef]
except AttributeError: pass

try: (LLVMOrcLLJITBuilderSetJITTargetMachineBuilder:=dll.LLVMOrcLLJITBuilderSetJITTargetMachineBuilder).restype, LLVMOrcLLJITBuilderSetJITTargetMachineBuilder.argtypes = None, [LLVMOrcLLJITBuilderRef, LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator:=dll.LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator).restype, LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator.argtypes = None, [LLVMOrcLLJITBuilderRef, LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateLLJIT:=dll.LLVMOrcCreateLLJIT).restype, LLVMOrcCreateLLJIT.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcLLJITRef), LLVMOrcLLJITBuilderRef]
except AttributeError: pass

try: (LLVMOrcDisposeLLJIT:=dll.LLVMOrcDisposeLLJIT).restype, LLVMOrcDisposeLLJIT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetExecutionSession:=dll.LLVMOrcLLJITGetExecutionSession).restype, LLVMOrcLLJITGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetMainJITDylib:=dll.LLVMOrcLLJITGetMainJITDylib).restype, LLVMOrcLLJITGetMainJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetTripleString:=dll.LLVMOrcLLJITGetTripleString).restype, LLVMOrcLLJITGetTripleString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetGlobalPrefix:=dll.LLVMOrcLLJITGetGlobalPrefix).restype, LLVMOrcLLJITGetGlobalPrefix.argtypes = ctypes.c_char, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITMangleAndIntern:=dll.LLVMOrcLLJITMangleAndIntern).restype, LLVMOrcLLJITMangleAndIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcLLJITRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcLLJITAddObjectFile:=dll.LLVMOrcLLJITAddObjectFile).restype, LLVMOrcLLJITAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcLLJITAddObjectFileWithRT:=dll.LLVMOrcLLJITAddObjectFileWithRT).restype, LLVMOrcLLJITAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcLLJITAddLLVMIRModule:=dll.LLVMOrcLLJITAddLLVMIRModule).restype, LLVMOrcLLJITAddLLVMIRModule.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcJITDylibRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcLLJITAddLLVMIRModuleWithRT:=dll.LLVMOrcLLJITAddLLVMIRModuleWithRT).restype, LLVMOrcLLJITAddLLVMIRModuleWithRT.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, LLVMOrcResourceTrackerRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcLLJITLookup:=dll.LLVMOrcLLJITLookup).restype, LLVMOrcLLJITLookup.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef, ctypes.POINTER(LLVMOrcExecutorAddress), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcLLJITGetObjLinkingLayer:=dll.LLVMOrcLLJITGetObjLinkingLayer).restype, LLVMOrcLLJITGetObjLinkingLayer.argtypes = LLVMOrcObjectLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetObjTransformLayer:=dll.LLVMOrcLLJITGetObjTransformLayer).restype, LLVMOrcLLJITGetObjTransformLayer.argtypes = LLVMOrcObjectTransformLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetIRTransformLayer:=dll.LLVMOrcLLJITGetIRTransformLayer).restype, LLVMOrcLLJITGetIRTransformLayer.argtypes = LLVMOrcIRTransformLayerRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITGetDataLayoutStr:=dll.LLVMOrcLLJITGetDataLayoutStr).restype, LLVMOrcLLJITGetDataLayoutStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcLLJITRef]
except AttributeError: pass

try: (LLVMOrcLLJITEnableDebugSupport:=dll.LLVMOrcLLJITEnableDebugSupport).restype, LLVMOrcLLJITEnableDebugSupport.argtypes = LLVMErrorRef, [LLVMOrcLLJITRef]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

LLVMLinkerMode = CEnum(ctypes.c_uint32)
LLVMLinkerDestroySource = LLVMLinkerMode.define('LLVMLinkerDestroySource', 0)
LLVMLinkerPreserveSource_Removed = LLVMLinkerMode.define('LLVMLinkerPreserveSource_Removed', 1)

try: (LLVMLinkModules2:=dll.LLVMLinkModules2).restype, LLVMLinkModules2.argtypes = LLVMBool, [LLVMModuleRef, LLVMModuleRef]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

class struct_LLVMOpaqueSectionIterator(Struct): pass
LLVMSectionIteratorRef = ctypes.POINTER(struct_LLVMOpaqueSectionIterator)
class struct_LLVMOpaqueSymbolIterator(Struct): pass
LLVMSymbolIteratorRef = ctypes.POINTER(struct_LLVMOpaqueSymbolIterator)
class struct_LLVMOpaqueRelocationIterator(Struct): pass
LLVMRelocationIteratorRef = ctypes.POINTER(struct_LLVMOpaqueRelocationIterator)
LLVMBinaryType = CEnum(ctypes.c_uint32)
LLVMBinaryTypeArchive = LLVMBinaryType.define('LLVMBinaryTypeArchive', 0)
LLVMBinaryTypeMachOUniversalBinary = LLVMBinaryType.define('LLVMBinaryTypeMachOUniversalBinary', 1)
LLVMBinaryTypeCOFFImportFile = LLVMBinaryType.define('LLVMBinaryTypeCOFFImportFile', 2)
LLVMBinaryTypeIR = LLVMBinaryType.define('LLVMBinaryTypeIR', 3)
LLVMBinaryTypeWinRes = LLVMBinaryType.define('LLVMBinaryTypeWinRes', 4)
LLVMBinaryTypeCOFF = LLVMBinaryType.define('LLVMBinaryTypeCOFF', 5)
LLVMBinaryTypeELF32L = LLVMBinaryType.define('LLVMBinaryTypeELF32L', 6)
LLVMBinaryTypeELF32B = LLVMBinaryType.define('LLVMBinaryTypeELF32B', 7)
LLVMBinaryTypeELF64L = LLVMBinaryType.define('LLVMBinaryTypeELF64L', 8)
LLVMBinaryTypeELF64B = LLVMBinaryType.define('LLVMBinaryTypeELF64B', 9)
LLVMBinaryTypeMachO32L = LLVMBinaryType.define('LLVMBinaryTypeMachO32L', 10)
LLVMBinaryTypeMachO32B = LLVMBinaryType.define('LLVMBinaryTypeMachO32B', 11)
LLVMBinaryTypeMachO64L = LLVMBinaryType.define('LLVMBinaryTypeMachO64L', 12)
LLVMBinaryTypeMachO64B = LLVMBinaryType.define('LLVMBinaryTypeMachO64B', 13)
LLVMBinaryTypeWasm = LLVMBinaryType.define('LLVMBinaryTypeWasm', 14)
LLVMBinaryTypeOffload = LLVMBinaryType.define('LLVMBinaryTypeOffload', 15)

class struct_LLVMOpaqueBinary(Struct): pass
LLVMBinaryRef = ctypes.POINTER(struct_LLVMOpaqueBinary)
try: (LLVMCreateBinary:=dll.LLVMCreateBinary).restype, LLVMCreateBinary.argtypes = LLVMBinaryRef, [LLVMMemoryBufferRef, LLVMContextRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMDisposeBinary:=dll.LLVMDisposeBinary).restype, LLVMDisposeBinary.argtypes = None, [LLVMBinaryRef]
except AttributeError: pass

try: (LLVMBinaryCopyMemoryBuffer:=dll.LLVMBinaryCopyMemoryBuffer).restype, LLVMBinaryCopyMemoryBuffer.argtypes = LLVMMemoryBufferRef, [LLVMBinaryRef]
except AttributeError: pass

try: (LLVMBinaryGetType:=dll.LLVMBinaryGetType).restype, LLVMBinaryGetType.argtypes = LLVMBinaryType, [LLVMBinaryRef]
except AttributeError: pass

try: (LLVMMachOUniversalBinaryCopyObjectForArch:=dll.LLVMMachOUniversalBinaryCopyObjectForArch).restype, LLVMMachOUniversalBinaryCopyObjectForArch.argtypes = LLVMBinaryRef, [LLVMBinaryRef, ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMObjectFileCopySectionIterator:=dll.LLVMObjectFileCopySectionIterator).restype, LLVMObjectFileCopySectionIterator.argtypes = LLVMSectionIteratorRef, [LLVMBinaryRef]
except AttributeError: pass

try: (LLVMObjectFileIsSectionIteratorAtEnd:=dll.LLVMObjectFileIsSectionIteratorAtEnd).restype, LLVMObjectFileIsSectionIteratorAtEnd.argtypes = LLVMBool, [LLVMBinaryRef, LLVMSectionIteratorRef]
except AttributeError: pass

try: (LLVMObjectFileCopySymbolIterator:=dll.LLVMObjectFileCopySymbolIterator).restype, LLVMObjectFileCopySymbolIterator.argtypes = LLVMSymbolIteratorRef, [LLVMBinaryRef]
except AttributeError: pass

try: (LLVMObjectFileIsSymbolIteratorAtEnd:=dll.LLVMObjectFileIsSymbolIteratorAtEnd).restype, LLVMObjectFileIsSymbolIteratorAtEnd.argtypes = LLVMBool, [LLVMBinaryRef, LLVMSymbolIteratorRef]
except AttributeError: pass

try: (LLVMDisposeSectionIterator:=dll.LLVMDisposeSectionIterator).restype, LLVMDisposeSectionIterator.argtypes = None, [LLVMSectionIteratorRef]
except AttributeError: pass

try: (LLVMMoveToNextSection:=dll.LLVMMoveToNextSection).restype, LLVMMoveToNextSection.argtypes = None, [LLVMSectionIteratorRef]
except AttributeError: pass

try: (LLVMMoveToContainingSection:=dll.LLVMMoveToContainingSection).restype, LLVMMoveToContainingSection.argtypes = None, [LLVMSectionIteratorRef, LLVMSymbolIteratorRef]
except AttributeError: pass

try: (LLVMDisposeSymbolIterator:=dll.LLVMDisposeSymbolIterator).restype, LLVMDisposeSymbolIterator.argtypes = None, [LLVMSymbolIteratorRef]
except AttributeError: pass

try: (LLVMMoveToNextSymbol:=dll.LLVMMoveToNextSymbol).restype, LLVMMoveToNextSymbol.argtypes = None, [LLVMSymbolIteratorRef]
except AttributeError: pass

try: (LLVMGetSectionName:=dll.LLVMGetSectionName).restype, LLVMGetSectionName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMSectionIteratorRef]
except AttributeError: pass

try: (LLVMGetSectionSize:=dll.LLVMGetSectionSize).restype, LLVMGetSectionSize.argtypes = uint64_t, [LLVMSectionIteratorRef]
except AttributeError: pass

try: (LLVMGetSectionContents:=dll.LLVMGetSectionContents).restype, LLVMGetSectionContents.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMSectionIteratorRef]
except AttributeError: pass

try: (LLVMGetSectionAddress:=dll.LLVMGetSectionAddress).restype, LLVMGetSectionAddress.argtypes = uint64_t, [LLVMSectionIteratorRef]
except AttributeError: pass

try: (LLVMGetSectionContainsSymbol:=dll.LLVMGetSectionContainsSymbol).restype, LLVMGetSectionContainsSymbol.argtypes = LLVMBool, [LLVMSectionIteratorRef, LLVMSymbolIteratorRef]
except AttributeError: pass

try: (LLVMGetRelocations:=dll.LLVMGetRelocations).restype, LLVMGetRelocations.argtypes = LLVMRelocationIteratorRef, [LLVMSectionIteratorRef]
except AttributeError: pass

try: (LLVMDisposeRelocationIterator:=dll.LLVMDisposeRelocationIterator).restype, LLVMDisposeRelocationIterator.argtypes = None, [LLVMRelocationIteratorRef]
except AttributeError: pass

try: (LLVMIsRelocationIteratorAtEnd:=dll.LLVMIsRelocationIteratorAtEnd).restype, LLVMIsRelocationIteratorAtEnd.argtypes = LLVMBool, [LLVMSectionIteratorRef, LLVMRelocationIteratorRef]
except AttributeError: pass

try: (LLVMMoveToNextRelocation:=dll.LLVMMoveToNextRelocation).restype, LLVMMoveToNextRelocation.argtypes = None, [LLVMRelocationIteratorRef]
except AttributeError: pass

try: (LLVMGetSymbolName:=dll.LLVMGetSymbolName).restype, LLVMGetSymbolName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMSymbolIteratorRef]
except AttributeError: pass

try: (LLVMGetSymbolAddress:=dll.LLVMGetSymbolAddress).restype, LLVMGetSymbolAddress.argtypes = uint64_t, [LLVMSymbolIteratorRef]
except AttributeError: pass

try: (LLVMGetSymbolSize:=dll.LLVMGetSymbolSize).restype, LLVMGetSymbolSize.argtypes = uint64_t, [LLVMSymbolIteratorRef]
except AttributeError: pass

try: (LLVMGetRelocationOffset:=dll.LLVMGetRelocationOffset).restype, LLVMGetRelocationOffset.argtypes = uint64_t, [LLVMRelocationIteratorRef]
except AttributeError: pass

try: (LLVMGetRelocationSymbol:=dll.LLVMGetRelocationSymbol).restype, LLVMGetRelocationSymbol.argtypes = LLVMSymbolIteratorRef, [LLVMRelocationIteratorRef]
except AttributeError: pass

try: (LLVMGetRelocationType:=dll.LLVMGetRelocationType).restype, LLVMGetRelocationType.argtypes = uint64_t, [LLVMRelocationIteratorRef]
except AttributeError: pass

try: (LLVMGetRelocationTypeName:=dll.LLVMGetRelocationTypeName).restype, LLVMGetRelocationTypeName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMRelocationIteratorRef]
except AttributeError: pass

try: (LLVMGetRelocationValueString:=dll.LLVMGetRelocationValueString).restype, LLVMGetRelocationValueString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMRelocationIteratorRef]
except AttributeError: pass

class struct_LLVMOpaqueObjectFile(Struct): pass
LLVMObjectFileRef = ctypes.POINTER(struct_LLVMOpaqueObjectFile)
try: (LLVMCreateObjectFile:=dll.LLVMCreateObjectFile).restype, LLVMCreateObjectFile.argtypes = LLVMObjectFileRef, [LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMDisposeObjectFile:=dll.LLVMDisposeObjectFile).restype, LLVMDisposeObjectFile.argtypes = None, [LLVMObjectFileRef]
except AttributeError: pass

try: (LLVMGetSections:=dll.LLVMGetSections).restype, LLVMGetSections.argtypes = LLVMSectionIteratorRef, [LLVMObjectFileRef]
except AttributeError: pass

try: (LLVMIsSectionIteratorAtEnd:=dll.LLVMIsSectionIteratorAtEnd).restype, LLVMIsSectionIteratorAtEnd.argtypes = LLVMBool, [LLVMObjectFileRef, LLVMSectionIteratorRef]
except AttributeError: pass

try: (LLVMGetSymbols:=dll.LLVMGetSymbols).restype, LLVMGetSymbols.argtypes = LLVMSymbolIteratorRef, [LLVMObjectFileRef]
except AttributeError: pass

try: (LLVMIsSymbolIteratorAtEnd:=dll.LLVMIsSymbolIteratorAtEnd).restype, LLVMIsSymbolIteratorAtEnd.argtypes = LLVMBool, [LLVMObjectFileRef, LLVMSymbolIteratorRef]
except AttributeError: pass

try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

LLVMJITSymbolGenericFlags = CEnum(ctypes.c_uint32)
LLVMJITSymbolGenericFlagsNone = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsNone', 0)
LLVMJITSymbolGenericFlagsExported = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsExported', 1)
LLVMJITSymbolGenericFlagsWeak = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsWeak', 2)
LLVMJITSymbolGenericFlagsCallable = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsCallable', 4)
LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly = LLVMJITSymbolGenericFlags.define('LLVMJITSymbolGenericFlagsMaterializationSideEffectsOnly', 8)

LLVMJITSymbolTargetFlags = ctypes.c_ubyte
class struct_LLVMOrcOpaqueObjectLinkingLayer(Struct): pass
LLVMOrcObjectLinkingLayerRef = ctypes.POINTER(struct_LLVMOrcOpaqueObjectLinkingLayer)
try: (LLVMOrcExecutionSessionSetErrorReporter:=dll.LLVMOrcExecutionSessionSetErrorReporter).restype, LLVMOrcExecutionSessionSetErrorReporter.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcErrorReporterFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcExecutionSessionGetSymbolStringPool:=dll.LLVMOrcExecutionSessionGetSymbolStringPool).restype, LLVMOrcExecutionSessionGetSymbolStringPool.argtypes = LLVMOrcSymbolStringPoolRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

try: (LLVMOrcSymbolStringPoolClearDeadEntries:=dll.LLVMOrcSymbolStringPoolClearDeadEntries).restype, LLVMOrcSymbolStringPoolClearDeadEntries.argtypes = None, [LLVMOrcSymbolStringPoolRef]
except AttributeError: pass

try: (LLVMOrcExecutionSessionIntern:=dll.LLVMOrcExecutionSessionIntern).restype, LLVMOrcExecutionSessionIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionLookup:=dll.LLVMOrcExecutionSessionLookup).restype, LLVMOrcExecutionSessionLookup.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcLookupKind, LLVMOrcCJITDylibSearchOrder, size_t, LLVMOrcCLookupSet, size_t, LLVMOrcExecutionSessionLookupHandleResultFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcRetainSymbolStringPoolEntry:=dll.LLVMOrcRetainSymbolStringPoolEntry).restype, LLVMOrcRetainSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcReleaseSymbolStringPoolEntry:=dll.LLVMOrcReleaseSymbolStringPoolEntry).restype, LLVMOrcReleaseSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcSymbolStringPoolEntryStr:=dll.LLVMOrcSymbolStringPoolEntryStr).restype, LLVMOrcSymbolStringPoolEntryStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcReleaseResourceTracker:=dll.LLVMOrcReleaseResourceTracker).restype, LLVMOrcReleaseResourceTracker.argtypes = None, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcResourceTrackerTransferTo:=dll.LLVMOrcResourceTrackerTransferTo).restype, LLVMOrcResourceTrackerTransferTo.argtypes = None, [LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcResourceTrackerRemove:=dll.LLVMOrcResourceTrackerRemove).restype, LLVMOrcResourceTrackerRemove.argtypes = LLVMErrorRef, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcDisposeDefinitionGenerator:=dll.LLVMOrcDisposeDefinitionGenerator).restype, LLVMOrcDisposeDefinitionGenerator.argtypes = None, [LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

try: (LLVMOrcDisposeMaterializationUnit:=dll.LLVMOrcDisposeMaterializationUnit).restype, LLVMOrcDisposeMaterializationUnit.argtypes = None, [LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcCreateCustomMaterializationUnit:=dll.LLVMOrcCreateCustomMaterializationUnit).restype, LLVMOrcCreateCustomMaterializationUnit.argtypes = LLVMOrcMaterializationUnitRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, LLVMOrcCSymbolFlagsMapPairs, size_t, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcMaterializationUnitMaterializeFunction, LLVMOrcMaterializationUnitDiscardFunction, LLVMOrcMaterializationUnitDestroyFunction]
except AttributeError: pass

try: (LLVMOrcAbsoluteSymbols:=dll.LLVMOrcAbsoluteSymbols).restype, LLVMOrcAbsoluteSymbols.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcLazyReexports:=dll.LLVMOrcLazyReexports).restype, LLVMOrcLazyReexports.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcLazyCallThroughManagerRef, LLVMOrcIndirectStubsManagerRef, LLVMOrcJITDylibRef, LLVMOrcCSymbolAliasMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcDisposeMaterializationResponsibility:=dll.LLVMOrcDisposeMaterializationResponsibility).restype, LLVMOrcDisposeMaterializationResponsibility.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetTargetDylib:=dll.LLVMOrcMaterializationResponsibilityGetTargetDylib).restype, LLVMOrcMaterializationResponsibilityGetTargetDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetExecutionSession:=dll.LLVMOrcMaterializationResponsibilityGetExecutionSession).restype, LLVMOrcMaterializationResponsibilityGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetSymbols:=dll.LLVMOrcMaterializationResponsibilityGetSymbols).restype, LLVMOrcMaterializationResponsibilityGetSymbols.argtypes = LLVMOrcCSymbolFlagsMapPairs, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMOrcDisposeCSymbolFlagsMap:=dll.LLVMOrcDisposeCSymbolFlagsMap).restype, LLVMOrcDisposeCSymbolFlagsMap.argtypes = None, [LLVMOrcCSymbolFlagsMapPairs]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetInitializerSymbol:=dll.LLVMOrcMaterializationResponsibilityGetInitializerSymbol).restype, LLVMOrcMaterializationResponsibilityGetInitializerSymbol.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetRequestedSymbols:=dll.LLVMOrcMaterializationResponsibilityGetRequestedSymbols).restype, LLVMOrcMaterializationResponsibilityGetRequestedSymbols.argtypes = ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMOrcDisposeSymbols:=dll.LLVMOrcDisposeSymbols).restype, LLVMOrcDisposeSymbols.argtypes = None, [ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityNotifyResolved:=dll.LLVMOrcMaterializationResponsibilityNotifyResolved).restype, LLVMOrcMaterializationResponsibilityNotifyResolved.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityNotifyEmitted:=dll.LLVMOrcMaterializationResponsibilityNotifyEmitted).restype, LLVMOrcMaterializationResponsibilityNotifyEmitted.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcCSymbolDependenceGroup), size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityDefineMaterializing:=dll.LLVMOrcMaterializationResponsibilityDefineMaterializing).restype, LLVMOrcMaterializationResponsibilityDefineMaterializing.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolFlagsMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityFailMaterialization:=dll.LLVMOrcMaterializationResponsibilityFailMaterialization).restype, LLVMOrcMaterializationResponsibilityFailMaterialization.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityReplace:=dll.LLVMOrcMaterializationResponsibilityReplace).restype, LLVMOrcMaterializationResponsibilityReplace.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityDelegate:=dll.LLVMOrcMaterializationResponsibilityDelegate).restype, LLVMOrcMaterializationResponsibilityDelegate.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), size_t, ctypes.POINTER(LLVMOrcMaterializationResponsibilityRef)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionCreateBareJITDylib:=dll.LLVMOrcExecutionSessionCreateBareJITDylib).restype, LLVMOrcExecutionSessionCreateBareJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionCreateJITDylib:=dll.LLVMOrcExecutionSessionCreateJITDylib).restype, LLVMOrcExecutionSessionCreateJITDylib.argtypes = LLVMErrorRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(LLVMOrcJITDylibRef), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionGetJITDylibByName:=dll.LLVMOrcExecutionSessionGetJITDylibByName).restype, LLVMOrcExecutionSessionGetJITDylibByName.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcJITDylibCreateResourceTracker:=dll.LLVMOrcJITDylibCreateResourceTracker).restype, LLVMOrcJITDylibCreateResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibGetDefaultResourceTracker:=dll.LLVMOrcJITDylibGetDefaultResourceTracker).restype, LLVMOrcJITDylibGetDefaultResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibDefine:=dll.LLVMOrcJITDylibDefine).restype, LLVMOrcJITDylibDefine.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcJITDylibClear:=dll.LLVMOrcJITDylibClear).restype, LLVMOrcJITDylibClear.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibAddGenerator:=dll.LLVMOrcJITDylibAddGenerator).restype, LLVMOrcJITDylibAddGenerator.argtypes = None, [LLVMOrcJITDylibRef, LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

try: (LLVMOrcCreateCustomCAPIDefinitionGenerator:=dll.LLVMOrcCreateCustomCAPIDefinitionGenerator).restype, LLVMOrcCreateCustomCAPIDefinitionGenerator.argtypes = LLVMOrcDefinitionGeneratorRef, [LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction, ctypes.c_void_p, LLVMOrcDisposeCAPIDefinitionGeneratorFunction]
except AttributeError: pass

try: (LLVMOrcLookupStateContinueLookup:=dll.LLVMOrcLookupStateContinueLookup).restype, LLVMOrcLookupStateContinueLookup.argtypes = None, [LLVMOrcLookupStateRef, LLVMErrorRef]
except AttributeError: pass

try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForPath).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.POINTER(ctypes.c_char), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateStaticLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateStaticLibrarySearchGeneratorForPath).restype, LLVMOrcCreateStaticLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), LLVMOrcObjectLayerRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcCreateNewThreadSafeContext:=dll.LLVMOrcCreateNewThreadSafeContext).restype, LLVMOrcCreateNewThreadSafeContext.argtypes = LLVMOrcThreadSafeContextRef, []
except AttributeError: pass

try: (LLVMOrcThreadSafeContextGetContext:=dll.LLVMOrcThreadSafeContextGetContext).restype, LLVMOrcThreadSafeContextGetContext.argtypes = LLVMContextRef, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcDisposeThreadSafeContext:=dll.LLVMOrcDisposeThreadSafeContext).restype, LLVMOrcDisposeThreadSafeContext.argtypes = None, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcCreateNewThreadSafeModule:=dll.LLVMOrcCreateNewThreadSafeModule).restype, LLVMOrcCreateNewThreadSafeModule.argtypes = LLVMOrcThreadSafeModuleRef, [LLVMModuleRef, LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcDisposeThreadSafeModule:=dll.LLVMOrcDisposeThreadSafeModule).restype, LLVMOrcDisposeThreadSafeModule.argtypes = None, [LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcThreadSafeModuleWithModuleDo:=dll.LLVMOrcThreadSafeModuleWithModuleDo).restype, LLVMOrcThreadSafeModuleWithModuleDo.argtypes = LLVMErrorRef, [LLVMOrcThreadSafeModuleRef, LLVMOrcGenericIRModuleOperationFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderDetectHost:=dll.LLVMOrcJITTargetMachineBuilderDetectHost).restype, LLVMOrcJITTargetMachineBuilderDetectHost.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcJITTargetMachineBuilderRef)]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine:=dll.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine).restype, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.argtypes = LLVMOrcJITTargetMachineBuilderRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMOrcDisposeJITTargetMachineBuilder:=dll.LLVMOrcDisposeJITTargetMachineBuilder).restype, LLVMOrcDisposeJITTargetMachineBuilder.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderGetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderGetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderGetTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderSetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderSetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderSetTargetTriple.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcObjectLayerAddObjectFile:=dll.LLVMOrcObjectLayerAddObjectFile).restype, LLVMOrcObjectLayerAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcObjectLayerAddObjectFileWithRT:=dll.LLVMOrcObjectLayerAddObjectFileWithRT).restype, LLVMOrcObjectLayerAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcObjectLayerEmit:=dll.LLVMOrcObjectLayerEmit).restype, LLVMOrcObjectLayerEmit.argtypes = None, [LLVMOrcObjectLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcDisposeObjectLayer:=dll.LLVMOrcDisposeObjectLayer).restype, LLVMOrcDisposeObjectLayer.argtypes = None, [LLVMOrcObjectLayerRef]
except AttributeError: pass

try: (LLVMOrcIRTransformLayerEmit:=dll.LLVMOrcIRTransformLayerEmit).restype, LLVMOrcIRTransformLayerEmit.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcIRTransformLayerSetTransform:=dll.LLVMOrcIRTransformLayerSetTransform).restype, LLVMOrcIRTransformLayerSetTransform.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcIRTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcObjectTransformLayerSetTransform:=dll.LLVMOrcObjectTransformLayerSetTransform).restype, LLVMOrcObjectTransformLayerSetTransform.argtypes = None, [LLVMOrcObjectTransformLayerRef, LLVMOrcObjectTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateLocalIndirectStubsManager:=dll.LLVMOrcCreateLocalIndirectStubsManager).restype, LLVMOrcCreateLocalIndirectStubsManager.argtypes = LLVMOrcIndirectStubsManagerRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcDisposeIndirectStubsManager:=dll.LLVMOrcDisposeIndirectStubsManager).restype, LLVMOrcDisposeIndirectStubsManager.argtypes = None, [LLVMOrcIndirectStubsManagerRef]
except AttributeError: pass

try: (LLVMOrcCreateLocalLazyCallThroughManager:=dll.LLVMOrcCreateLocalLazyCallThroughManager).restype, LLVMOrcCreateLocalLazyCallThroughManager.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char), LLVMOrcExecutionSessionRef, LLVMOrcJITTargetAddress, ctypes.POINTER(LLVMOrcLazyCallThroughManagerRef)]
except AttributeError: pass

try: (LLVMOrcDisposeLazyCallThroughManager:=dll.LLVMOrcDisposeLazyCallThroughManager).restype, LLVMOrcDisposeLazyCallThroughManager.argtypes = None, [LLVMOrcLazyCallThroughManagerRef]
except AttributeError: pass

try: (LLVMOrcCreateDumpObjects:=dll.LLVMOrcCreateDumpObjects).restype, LLVMOrcCreateDumpObjects.argtypes = LLVMOrcDumpObjectsRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcDisposeDumpObjects:=dll.LLVMOrcDisposeDumpObjects).restype, LLVMOrcDisposeDumpObjects.argtypes = None, [LLVMOrcDumpObjectsRef]
except AttributeError: pass

try: (LLVMOrcDumpObjects_CallOperator:=dll.LLVMOrcDumpObjects_CallOperator).restype, LLVMOrcDumpObjects_CallOperator.argtypes = LLVMErrorRef, [LLVMOrcDumpObjectsRef, ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMLinkInMCJIT:=dll.LLVMLinkInMCJIT).restype, LLVMLinkInMCJIT.argtypes = None, []
except AttributeError: pass

try: (LLVMLinkInInterpreter:=dll.LLVMLinkInInterpreter).restype, LLVMLinkInInterpreter.argtypes = None, []
except AttributeError: pass

try: (LLVMCreateGenericValueOfInt:=dll.LLVMCreateGenericValueOfInt).restype, LLVMCreateGenericValueOfInt.argtypes = LLVMGenericValueRef, [LLVMTypeRef, ctypes.c_uint64, LLVMBool]
except AttributeError: pass

try: (LLVMCreateGenericValueOfPointer:=dll.LLVMCreateGenericValueOfPointer).restype, LLVMCreateGenericValueOfPointer.argtypes = LLVMGenericValueRef, [ctypes.c_void_p]
except AttributeError: pass

try: (LLVMCreateGenericValueOfFloat:=dll.LLVMCreateGenericValueOfFloat).restype, LLVMCreateGenericValueOfFloat.argtypes = LLVMGenericValueRef, [LLVMTypeRef, ctypes.c_double]
except AttributeError: pass

try: (LLVMGenericValueIntWidth:=dll.LLVMGenericValueIntWidth).restype, LLVMGenericValueIntWidth.argtypes = ctypes.c_uint32, [LLVMGenericValueRef]
except AttributeError: pass

try: (LLVMGenericValueToInt:=dll.LLVMGenericValueToInt).restype, LLVMGenericValueToInt.argtypes = ctypes.c_uint64, [LLVMGenericValueRef, LLVMBool]
except AttributeError: pass

try: (LLVMGenericValueToPointer:=dll.LLVMGenericValueToPointer).restype, LLVMGenericValueToPointer.argtypes = ctypes.c_void_p, [LLVMGenericValueRef]
except AttributeError: pass

try: (LLVMGenericValueToFloat:=dll.LLVMGenericValueToFloat).restype, LLVMGenericValueToFloat.argtypes = ctypes.c_double, [LLVMTypeRef, LLVMGenericValueRef]
except AttributeError: pass

try: (LLVMDisposeGenericValue:=dll.LLVMDisposeGenericValue).restype, LLVMDisposeGenericValue.argtypes = None, [LLVMGenericValueRef]
except AttributeError: pass

try: (LLVMCreateExecutionEngineForModule:=dll.LLVMCreateExecutionEngineForModule).restype, LLVMCreateExecutionEngineForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMCreateInterpreterForModule:=dll.LLVMCreateInterpreterForModule).restype, LLVMCreateInterpreterForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMCreateJITCompilerForModule:=dll.LLVMCreateJITCompilerForModule).restype, LLVMCreateJITCompilerForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMInitializeMCJITCompilerOptions:=dll.LLVMInitializeMCJITCompilerOptions).restype, LLVMInitializeMCJITCompilerOptions.argtypes = None, [ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t]
except AttributeError: pass

try: (LLVMCreateMCJITCompilerForModule:=dll.LLVMCreateMCJITCompilerForModule).restype, LLVMCreateMCJITCompilerForModule.argtypes = LLVMBool, [ctypes.POINTER(LLVMExecutionEngineRef), LLVMModuleRef, ctypes.POINTER(struct_LLVMMCJITCompilerOptions), size_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMDisposeExecutionEngine:=dll.LLVMDisposeExecutionEngine).restype, LLVMDisposeExecutionEngine.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMRunStaticConstructors:=dll.LLVMRunStaticConstructors).restype, LLVMRunStaticConstructors.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMRunStaticDestructors:=dll.LLVMRunStaticDestructors).restype, LLVMRunStaticDestructors.argtypes = None, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMRunFunctionAsMain:=dll.LLVMRunFunctionAsMain).restype, LLVMRunFunctionAsMain.argtypes = ctypes.c_int32, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMRunFunction:=dll.LLVMRunFunction).restype, LLVMRunFunction.argtypes = LLVMGenericValueRef, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_uint32, ctypes.POINTER(LLVMGenericValueRef)]
except AttributeError: pass

try: (LLVMFreeMachineCodeForFunction:=dll.LLVMFreeMachineCodeForFunction).restype, LLVMFreeMachineCodeForFunction.argtypes = None, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMAddModule:=dll.LLVMAddModule).restype, LLVMAddModule.argtypes = None, [LLVMExecutionEngineRef, LLVMModuleRef]
except AttributeError: pass

try: (LLVMRemoveModule:=dll.LLVMRemoveModule).restype, LLVMRemoveModule.argtypes = LLVMBool, [LLVMExecutionEngineRef, LLVMModuleRef, ctypes.POINTER(LLVMModuleRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMFindFunction:=dll.LLVMFindFunction).restype, LLVMFindFunction.argtypes = LLVMBool, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMValueRef)]
except AttributeError: pass

try: (LLVMRecompileAndRelinkFunction:=dll.LLVMRecompileAndRelinkFunction).restype, LLVMRecompileAndRelinkFunction.argtypes = ctypes.c_void_p, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetExecutionEngineTargetData:=dll.LLVMGetExecutionEngineTargetData).restype, LLVMGetExecutionEngineTargetData.argtypes = LLVMTargetDataRef, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMGetExecutionEngineTargetMachine:=dll.LLVMGetExecutionEngineTargetMachine).restype, LLVMGetExecutionEngineTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMExecutionEngineRef]
except AttributeError: pass

try: (LLVMAddGlobalMapping:=dll.LLVMAddGlobalMapping).restype, LLVMAddGlobalMapping.argtypes = None, [LLVMExecutionEngineRef, LLVMValueRef, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMGetPointerToGlobal:=dll.LLVMGetPointerToGlobal).restype, LLVMGetPointerToGlobal.argtypes = ctypes.c_void_p, [LLVMExecutionEngineRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMGetGlobalValueAddress:=dll.LLVMGetGlobalValueAddress).restype, LLVMGetGlobalValueAddress.argtypes = uint64_t, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetFunctionAddress:=dll.LLVMGetFunctionAddress).restype, LLVMGetFunctionAddress.argtypes = uint64_t, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMExecutionEngineGetErrMsg:=dll.LLVMExecutionEngineGetErrMsg).restype, LLVMExecutionEngineGetErrMsg.argtypes = LLVMBool, [LLVMExecutionEngineRef, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMCreateSimpleMCJITMemoryManager:=dll.LLVMCreateSimpleMCJITMemoryManager).restype, LLVMCreateSimpleMCJITMemoryManager.argtypes = LLVMMCJITMemoryManagerRef, [ctypes.c_void_p, LLVMMemoryManagerAllocateCodeSectionCallback, LLVMMemoryManagerAllocateDataSectionCallback, LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback]
except AttributeError: pass

try: (LLVMDisposeMCJITMemoryManager:=dll.LLVMDisposeMCJITMemoryManager).restype, LLVMDisposeMCJITMemoryManager.argtypes = None, [LLVMMCJITMemoryManagerRef]
except AttributeError: pass

try: (LLVMCreateGDBRegistrationListener:=dll.LLVMCreateGDBRegistrationListener).restype, LLVMCreateGDBRegistrationListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

try: (LLVMCreateIntelJITEventListener:=dll.LLVMCreateIntelJITEventListener).restype, LLVMCreateIntelJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

try: (LLVMCreateOProfileJITEventListener:=dll.LLVMCreateOProfileJITEventListener).restype, LLVMCreateOProfileJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

try: (LLVMCreatePerfJITEventListener:=dll.LLVMCreatePerfJITEventListener).restype, LLVMCreatePerfJITEventListener.argtypes = LLVMJITEventListenerRef, []
except AttributeError: pass

try: (LLVMOrcExecutionSessionSetErrorReporter:=dll.LLVMOrcExecutionSessionSetErrorReporter).restype, LLVMOrcExecutionSessionSetErrorReporter.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcErrorReporterFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcExecutionSessionGetSymbolStringPool:=dll.LLVMOrcExecutionSessionGetSymbolStringPool).restype, LLVMOrcExecutionSessionGetSymbolStringPool.argtypes = LLVMOrcSymbolStringPoolRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

try: (LLVMOrcSymbolStringPoolClearDeadEntries:=dll.LLVMOrcSymbolStringPoolClearDeadEntries).restype, LLVMOrcSymbolStringPoolClearDeadEntries.argtypes = None, [LLVMOrcSymbolStringPoolRef]
except AttributeError: pass

try: (LLVMOrcExecutionSessionIntern:=dll.LLVMOrcExecutionSessionIntern).restype, LLVMOrcExecutionSessionIntern.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionLookup:=dll.LLVMOrcExecutionSessionLookup).restype, LLVMOrcExecutionSessionLookup.argtypes = None, [LLVMOrcExecutionSessionRef, LLVMOrcLookupKind, LLVMOrcCJITDylibSearchOrder, size_t, LLVMOrcCLookupSet, size_t, LLVMOrcExecutionSessionLookupHandleResultFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcRetainSymbolStringPoolEntry:=dll.LLVMOrcRetainSymbolStringPoolEntry).restype, LLVMOrcRetainSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcReleaseSymbolStringPoolEntry:=dll.LLVMOrcReleaseSymbolStringPoolEntry).restype, LLVMOrcReleaseSymbolStringPoolEntry.argtypes = None, [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcSymbolStringPoolEntryStr:=dll.LLVMOrcSymbolStringPoolEntryStr).restype, LLVMOrcSymbolStringPoolEntryStr.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcSymbolStringPoolEntryRef]
except AttributeError: pass

try: (LLVMOrcReleaseResourceTracker:=dll.LLVMOrcReleaseResourceTracker).restype, LLVMOrcReleaseResourceTracker.argtypes = None, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcResourceTrackerTransferTo:=dll.LLVMOrcResourceTrackerTransferTo).restype, LLVMOrcResourceTrackerTransferTo.argtypes = None, [LLVMOrcResourceTrackerRef, LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcResourceTrackerRemove:=dll.LLVMOrcResourceTrackerRemove).restype, LLVMOrcResourceTrackerRemove.argtypes = LLVMErrorRef, [LLVMOrcResourceTrackerRef]
except AttributeError: pass

try: (LLVMOrcDisposeDefinitionGenerator:=dll.LLVMOrcDisposeDefinitionGenerator).restype, LLVMOrcDisposeDefinitionGenerator.argtypes = None, [LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

try: (LLVMOrcDisposeMaterializationUnit:=dll.LLVMOrcDisposeMaterializationUnit).restype, LLVMOrcDisposeMaterializationUnit.argtypes = None, [LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcCreateCustomMaterializationUnit:=dll.LLVMOrcCreateCustomMaterializationUnit).restype, LLVMOrcCreateCustomMaterializationUnit.argtypes = LLVMOrcMaterializationUnitRef, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, LLVMOrcCSymbolFlagsMapPairs, size_t, LLVMOrcSymbolStringPoolEntryRef, LLVMOrcMaterializationUnitMaterializeFunction, LLVMOrcMaterializationUnitDiscardFunction, LLVMOrcMaterializationUnitDestroyFunction]
except AttributeError: pass

try: (LLVMOrcAbsoluteSymbols:=dll.LLVMOrcAbsoluteSymbols).restype, LLVMOrcAbsoluteSymbols.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcLazyReexports:=dll.LLVMOrcLazyReexports).restype, LLVMOrcLazyReexports.argtypes = LLVMOrcMaterializationUnitRef, [LLVMOrcLazyCallThroughManagerRef, LLVMOrcIndirectStubsManagerRef, LLVMOrcJITDylibRef, LLVMOrcCSymbolAliasMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcDisposeMaterializationResponsibility:=dll.LLVMOrcDisposeMaterializationResponsibility).restype, LLVMOrcDisposeMaterializationResponsibility.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetTargetDylib:=dll.LLVMOrcMaterializationResponsibilityGetTargetDylib).restype, LLVMOrcMaterializationResponsibilityGetTargetDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetExecutionSession:=dll.LLVMOrcMaterializationResponsibilityGetExecutionSession).restype, LLVMOrcMaterializationResponsibilityGetExecutionSession.argtypes = LLVMOrcExecutionSessionRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetSymbols:=dll.LLVMOrcMaterializationResponsibilityGetSymbols).restype, LLVMOrcMaterializationResponsibilityGetSymbols.argtypes = LLVMOrcCSymbolFlagsMapPairs, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMOrcDisposeCSymbolFlagsMap:=dll.LLVMOrcDisposeCSymbolFlagsMap).restype, LLVMOrcDisposeCSymbolFlagsMap.argtypes = None, [LLVMOrcCSymbolFlagsMapPairs]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetInitializerSymbol:=dll.LLVMOrcMaterializationResponsibilityGetInitializerSymbol).restype, LLVMOrcMaterializationResponsibilityGetInitializerSymbol.argtypes = LLVMOrcSymbolStringPoolEntryRef, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityGetRequestedSymbols:=dll.LLVMOrcMaterializationResponsibilityGetRequestedSymbols).restype, LLVMOrcMaterializationResponsibilityGetRequestedSymbols.argtypes = ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (LLVMOrcDisposeSymbols:=dll.LLVMOrcDisposeSymbols).restype, LLVMOrcDisposeSymbols.argtypes = None, [ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef)]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityNotifyResolved:=dll.LLVMOrcMaterializationResponsibilityNotifyResolved).restype, LLVMOrcMaterializationResponsibilityNotifyResolved.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityNotifyEmitted:=dll.LLVMOrcMaterializationResponsibilityNotifyEmitted).restype, LLVMOrcMaterializationResponsibilityNotifyEmitted.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcCSymbolDependenceGroup), size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityDefineMaterializing:=dll.LLVMOrcMaterializationResponsibilityDefineMaterializing).restype, LLVMOrcMaterializationResponsibilityDefineMaterializing.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcCSymbolFlagsMapPairs, size_t]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityFailMaterialization:=dll.LLVMOrcMaterializationResponsibilityFailMaterialization).restype, LLVMOrcMaterializationResponsibilityFailMaterialization.argtypes = None, [LLVMOrcMaterializationResponsibilityRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityReplace:=dll.LLVMOrcMaterializationResponsibilityReplace).restype, LLVMOrcMaterializationResponsibilityReplace.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcMaterializationResponsibilityDelegate:=dll.LLVMOrcMaterializationResponsibilityDelegate).restype, LLVMOrcMaterializationResponsibilityDelegate.argtypes = LLVMErrorRef, [LLVMOrcMaterializationResponsibilityRef, ctypes.POINTER(LLVMOrcSymbolStringPoolEntryRef), size_t, ctypes.POINTER(LLVMOrcMaterializationResponsibilityRef)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionCreateBareJITDylib:=dll.LLVMOrcExecutionSessionCreateBareJITDylib).restype, LLVMOrcExecutionSessionCreateBareJITDylib.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionCreateJITDylib:=dll.LLVMOrcExecutionSessionCreateJITDylib).restype, LLVMOrcExecutionSessionCreateJITDylib.argtypes = LLVMErrorRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(LLVMOrcJITDylibRef), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcExecutionSessionGetJITDylibByName:=dll.LLVMOrcExecutionSessionGetJITDylibByName).restype, LLVMOrcExecutionSessionGetJITDylibByName.argtypes = LLVMOrcJITDylibRef, [LLVMOrcExecutionSessionRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcJITDylibCreateResourceTracker:=dll.LLVMOrcJITDylibCreateResourceTracker).restype, LLVMOrcJITDylibCreateResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibGetDefaultResourceTracker:=dll.LLVMOrcJITDylibGetDefaultResourceTracker).restype, LLVMOrcJITDylibGetDefaultResourceTracker.argtypes = LLVMOrcResourceTrackerRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibDefine:=dll.LLVMOrcJITDylibDefine).restype, LLVMOrcJITDylibDefine.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef, LLVMOrcMaterializationUnitRef]
except AttributeError: pass

try: (LLVMOrcJITDylibClear:=dll.LLVMOrcJITDylibClear).restype, LLVMOrcJITDylibClear.argtypes = LLVMErrorRef, [LLVMOrcJITDylibRef]
except AttributeError: pass

try: (LLVMOrcJITDylibAddGenerator:=dll.LLVMOrcJITDylibAddGenerator).restype, LLVMOrcJITDylibAddGenerator.argtypes = None, [LLVMOrcJITDylibRef, LLVMOrcDefinitionGeneratorRef]
except AttributeError: pass

try: (LLVMOrcCreateCustomCAPIDefinitionGenerator:=dll.LLVMOrcCreateCustomCAPIDefinitionGenerator).restype, LLVMOrcCreateCustomCAPIDefinitionGenerator.argtypes = LLVMOrcDefinitionGeneratorRef, [LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction, ctypes.c_void_p, LLVMOrcDisposeCAPIDefinitionGeneratorFunction]
except AttributeError: pass

try: (LLVMOrcLookupStateContinueLookup:=dll.LLVMOrcLookupStateContinueLookup).restype, LLVMOrcLookupStateContinueLookup.argtypes = None, [LLVMOrcLookupStateRef, LLVMErrorRef]
except AttributeError: pass

try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateDynamicLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateDynamicLibrarySearchGeneratorForPath).restype, LLVMOrcCreateDynamicLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), ctypes.POINTER(ctypes.c_char), ctypes.c_char, LLVMOrcSymbolPredicate, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateStaticLibrarySearchGeneratorForPath:=dll.LLVMOrcCreateStaticLibrarySearchGeneratorForPath).restype, LLVMOrcCreateStaticLibrarySearchGeneratorForPath.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcDefinitionGeneratorRef), LLVMOrcObjectLayerRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcCreateNewThreadSafeContext:=dll.LLVMOrcCreateNewThreadSafeContext).restype, LLVMOrcCreateNewThreadSafeContext.argtypes = LLVMOrcThreadSafeContextRef, []
except AttributeError: pass

try: (LLVMOrcThreadSafeContextGetContext:=dll.LLVMOrcThreadSafeContextGetContext).restype, LLVMOrcThreadSafeContextGetContext.argtypes = LLVMContextRef, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcDisposeThreadSafeContext:=dll.LLVMOrcDisposeThreadSafeContext).restype, LLVMOrcDisposeThreadSafeContext.argtypes = None, [LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcCreateNewThreadSafeModule:=dll.LLVMOrcCreateNewThreadSafeModule).restype, LLVMOrcCreateNewThreadSafeModule.argtypes = LLVMOrcThreadSafeModuleRef, [LLVMModuleRef, LLVMOrcThreadSafeContextRef]
except AttributeError: pass

try: (LLVMOrcDisposeThreadSafeModule:=dll.LLVMOrcDisposeThreadSafeModule).restype, LLVMOrcDisposeThreadSafeModule.argtypes = None, [LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcThreadSafeModuleWithModuleDo:=dll.LLVMOrcThreadSafeModuleWithModuleDo).restype, LLVMOrcThreadSafeModuleWithModuleDo.argtypes = LLVMErrorRef, [LLVMOrcThreadSafeModuleRef, LLVMOrcGenericIRModuleOperationFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderDetectHost:=dll.LLVMOrcJITTargetMachineBuilderDetectHost).restype, LLVMOrcJITTargetMachineBuilderDetectHost.argtypes = LLVMErrorRef, [ctypes.POINTER(LLVMOrcJITTargetMachineBuilderRef)]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine:=dll.LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine).restype, LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine.argtypes = LLVMOrcJITTargetMachineBuilderRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMOrcDisposeJITTargetMachineBuilder:=dll.LLVMOrcDisposeJITTargetMachineBuilder).restype, LLVMOrcDisposeJITTargetMachineBuilder.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderGetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderGetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderGetTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMOrcJITTargetMachineBuilderRef]
except AttributeError: pass

try: (LLVMOrcJITTargetMachineBuilderSetTargetTriple:=dll.LLVMOrcJITTargetMachineBuilderSetTargetTriple).restype, LLVMOrcJITTargetMachineBuilderSetTargetTriple.argtypes = None, [LLVMOrcJITTargetMachineBuilderRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcObjectLayerAddObjectFile:=dll.LLVMOrcObjectLayerAddObjectFile).restype, LLVMOrcObjectLayerAddObjectFile.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcJITDylibRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcObjectLayerAddObjectFileWithRT:=dll.LLVMOrcObjectLayerAddObjectFileWithRT).restype, LLVMOrcObjectLayerAddObjectFileWithRT.argtypes = LLVMErrorRef, [LLVMOrcObjectLayerRef, LLVMOrcResourceTrackerRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcObjectLayerEmit:=dll.LLVMOrcObjectLayerEmit).restype, LLVMOrcObjectLayerEmit.argtypes = None, [LLVMOrcObjectLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMMemoryBufferRef]
except AttributeError: pass

try: (LLVMOrcDisposeObjectLayer:=dll.LLVMOrcDisposeObjectLayer).restype, LLVMOrcDisposeObjectLayer.argtypes = None, [LLVMOrcObjectLayerRef]
except AttributeError: pass

try: (LLVMOrcIRTransformLayerEmit:=dll.LLVMOrcIRTransformLayerEmit).restype, LLVMOrcIRTransformLayerEmit.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcMaterializationResponsibilityRef, LLVMOrcThreadSafeModuleRef]
except AttributeError: pass

try: (LLVMOrcIRTransformLayerSetTransform:=dll.LLVMOrcIRTransformLayerSetTransform).restype, LLVMOrcIRTransformLayerSetTransform.argtypes = None, [LLVMOrcIRTransformLayerRef, LLVMOrcIRTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcObjectTransformLayerSetTransform:=dll.LLVMOrcObjectTransformLayerSetTransform).restype, LLVMOrcObjectTransformLayerSetTransform.argtypes = None, [LLVMOrcObjectTransformLayerRef, LLVMOrcObjectTransformLayerTransformFunction, ctypes.c_void_p]
except AttributeError: pass

try: (LLVMOrcCreateLocalIndirectStubsManager:=dll.LLVMOrcCreateLocalIndirectStubsManager).restype, LLVMOrcCreateLocalIndirectStubsManager.argtypes = LLVMOrcIndirectStubsManagerRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcDisposeIndirectStubsManager:=dll.LLVMOrcDisposeIndirectStubsManager).restype, LLVMOrcDisposeIndirectStubsManager.argtypes = None, [LLVMOrcIndirectStubsManagerRef]
except AttributeError: pass

try: (LLVMOrcCreateLocalLazyCallThroughManager:=dll.LLVMOrcCreateLocalLazyCallThroughManager).restype, LLVMOrcCreateLocalLazyCallThroughManager.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char), LLVMOrcExecutionSessionRef, LLVMOrcJITTargetAddress, ctypes.POINTER(LLVMOrcLazyCallThroughManagerRef)]
except AttributeError: pass

try: (LLVMOrcDisposeLazyCallThroughManager:=dll.LLVMOrcDisposeLazyCallThroughManager).restype, LLVMOrcDisposeLazyCallThroughManager.argtypes = None, [LLVMOrcLazyCallThroughManagerRef]
except AttributeError: pass

try: (LLVMOrcCreateDumpObjects:=dll.LLVMOrcCreateDumpObjects).restype, LLVMOrcCreateDumpObjects.argtypes = LLVMOrcDumpObjectsRef, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMOrcDisposeDumpObjects:=dll.LLVMOrcDisposeDumpObjects).restype, LLVMOrcDisposeDumpObjects.argtypes = None, [LLVMOrcDumpObjectsRef]
except AttributeError: pass

try: (LLVMOrcDumpObjects_CallOperator:=dll.LLVMOrcDumpObjects_CallOperator).restype, LLVMOrcDumpObjects_CallOperator.argtypes = LLVMErrorRef, [LLVMOrcDumpObjectsRef, ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

LLVMMemoryManagerCreateContextCallback = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)
LLVMMemoryManagerNotifyTerminatingCallback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
try: (LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager:=dll.LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager).restype, LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager.argtypes = LLVMOrcObjectLayerRef, [LLVMOrcExecutionSessionRef]
except AttributeError: pass

try: (LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks:=dll.LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks).restype, LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks.argtypes = LLVMOrcObjectLayerRef, [LLVMOrcExecutionSessionRef, ctypes.c_void_p, LLVMMemoryManagerCreateContextCallback, LLVMMemoryManagerNotifyTerminatingCallback, LLVMMemoryManagerAllocateCodeSectionCallback, LLVMMemoryManagerAllocateDataSectionCallback, LLVMMemoryManagerFinalizeMemoryCallback, LLVMMemoryManagerDestroyCallback]
except AttributeError: pass

try: (LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener:=dll.LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener).restype, LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener.argtypes = None, [LLVMOrcObjectLayerRef, LLVMJITEventListenerRef]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

enum_LLVMRemarkType = CEnum(ctypes.c_uint32)
LLVMRemarkTypeUnknown = enum_LLVMRemarkType.define('LLVMRemarkTypeUnknown', 0)
LLVMRemarkTypePassed = enum_LLVMRemarkType.define('LLVMRemarkTypePassed', 1)
LLVMRemarkTypeMissed = enum_LLVMRemarkType.define('LLVMRemarkTypeMissed', 2)
LLVMRemarkTypeAnalysis = enum_LLVMRemarkType.define('LLVMRemarkTypeAnalysis', 3)
LLVMRemarkTypeAnalysisFPCommute = enum_LLVMRemarkType.define('LLVMRemarkTypeAnalysisFPCommute', 4)
LLVMRemarkTypeAnalysisAliasing = enum_LLVMRemarkType.define('LLVMRemarkTypeAnalysisAliasing', 5)
LLVMRemarkTypeFailure = enum_LLVMRemarkType.define('LLVMRemarkTypeFailure', 6)

class struct_LLVMRemarkOpaqueString(Struct): pass
LLVMRemarkStringRef = ctypes.POINTER(struct_LLVMRemarkOpaqueString)
try: (LLVMRemarkStringGetData:=dll.LLVMRemarkStringGetData).restype, LLVMRemarkStringGetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMRemarkStringRef]
except AttributeError: pass

try: (LLVMRemarkStringGetLen:=dll.LLVMRemarkStringGetLen).restype, LLVMRemarkStringGetLen.argtypes = uint32_t, [LLVMRemarkStringRef]
except AttributeError: pass

class struct_LLVMRemarkOpaqueDebugLoc(Struct): pass
LLVMRemarkDebugLocRef = ctypes.POINTER(struct_LLVMRemarkOpaqueDebugLoc)
try: (LLVMRemarkDebugLocGetSourceFilePath:=dll.LLVMRemarkDebugLocGetSourceFilePath).restype, LLVMRemarkDebugLocGetSourceFilePath.argtypes = LLVMRemarkStringRef, [LLVMRemarkDebugLocRef]
except AttributeError: pass

try: (LLVMRemarkDebugLocGetSourceLine:=dll.LLVMRemarkDebugLocGetSourceLine).restype, LLVMRemarkDebugLocGetSourceLine.argtypes = uint32_t, [LLVMRemarkDebugLocRef]
except AttributeError: pass

try: (LLVMRemarkDebugLocGetSourceColumn:=dll.LLVMRemarkDebugLocGetSourceColumn).restype, LLVMRemarkDebugLocGetSourceColumn.argtypes = uint32_t, [LLVMRemarkDebugLocRef]
except AttributeError: pass

class struct_LLVMRemarkOpaqueArg(Struct): pass
LLVMRemarkArgRef = ctypes.POINTER(struct_LLVMRemarkOpaqueArg)
try: (LLVMRemarkArgGetKey:=dll.LLVMRemarkArgGetKey).restype, LLVMRemarkArgGetKey.argtypes = LLVMRemarkStringRef, [LLVMRemarkArgRef]
except AttributeError: pass

try: (LLVMRemarkArgGetValue:=dll.LLVMRemarkArgGetValue).restype, LLVMRemarkArgGetValue.argtypes = LLVMRemarkStringRef, [LLVMRemarkArgRef]
except AttributeError: pass

try: (LLVMRemarkArgGetDebugLoc:=dll.LLVMRemarkArgGetDebugLoc).restype, LLVMRemarkArgGetDebugLoc.argtypes = LLVMRemarkDebugLocRef, [LLVMRemarkArgRef]
except AttributeError: pass

class struct_LLVMRemarkOpaqueEntry(Struct): pass
LLVMRemarkEntryRef = ctypes.POINTER(struct_LLVMRemarkOpaqueEntry)
try: (LLVMRemarkEntryDispose:=dll.LLVMRemarkEntryDispose).restype, LLVMRemarkEntryDispose.argtypes = None, [LLVMRemarkEntryRef]
except AttributeError: pass

try: (LLVMRemarkEntryGetType:=dll.LLVMRemarkEntryGetType).restype, LLVMRemarkEntryGetType.argtypes = enum_LLVMRemarkType, [LLVMRemarkEntryRef]
except AttributeError: pass

try: (LLVMRemarkEntryGetPassName:=dll.LLVMRemarkEntryGetPassName).restype, LLVMRemarkEntryGetPassName.argtypes = LLVMRemarkStringRef, [LLVMRemarkEntryRef]
except AttributeError: pass

try: (LLVMRemarkEntryGetRemarkName:=dll.LLVMRemarkEntryGetRemarkName).restype, LLVMRemarkEntryGetRemarkName.argtypes = LLVMRemarkStringRef, [LLVMRemarkEntryRef]
except AttributeError: pass

try: (LLVMRemarkEntryGetFunctionName:=dll.LLVMRemarkEntryGetFunctionName).restype, LLVMRemarkEntryGetFunctionName.argtypes = LLVMRemarkStringRef, [LLVMRemarkEntryRef]
except AttributeError: pass

try: (LLVMRemarkEntryGetDebugLoc:=dll.LLVMRemarkEntryGetDebugLoc).restype, LLVMRemarkEntryGetDebugLoc.argtypes = LLVMRemarkDebugLocRef, [LLVMRemarkEntryRef]
except AttributeError: pass

try: (LLVMRemarkEntryGetHotness:=dll.LLVMRemarkEntryGetHotness).restype, LLVMRemarkEntryGetHotness.argtypes = uint64_t, [LLVMRemarkEntryRef]
except AttributeError: pass

try: (LLVMRemarkEntryGetNumArgs:=dll.LLVMRemarkEntryGetNumArgs).restype, LLVMRemarkEntryGetNumArgs.argtypes = uint32_t, [LLVMRemarkEntryRef]
except AttributeError: pass

try: (LLVMRemarkEntryGetFirstArg:=dll.LLVMRemarkEntryGetFirstArg).restype, LLVMRemarkEntryGetFirstArg.argtypes = LLVMRemarkArgRef, [LLVMRemarkEntryRef]
except AttributeError: pass

try: (LLVMRemarkEntryGetNextArg:=dll.LLVMRemarkEntryGetNextArg).restype, LLVMRemarkEntryGetNextArg.argtypes = LLVMRemarkArgRef, [LLVMRemarkArgRef, LLVMRemarkEntryRef]
except AttributeError: pass

class struct_LLVMRemarkOpaqueParser(Struct): pass
LLVMRemarkParserRef = ctypes.POINTER(struct_LLVMRemarkOpaqueParser)
try: (LLVMRemarkParserCreateYAML:=dll.LLVMRemarkParserCreateYAML).restype, LLVMRemarkParserCreateYAML.argtypes = LLVMRemarkParserRef, [ctypes.c_void_p, uint64_t]
except AttributeError: pass

try: (LLVMRemarkParserCreateBitstream:=dll.LLVMRemarkParserCreateBitstream).restype, LLVMRemarkParserCreateBitstream.argtypes = LLVMRemarkParserRef, [ctypes.c_void_p, uint64_t]
except AttributeError: pass

try: (LLVMRemarkParserGetNext:=dll.LLVMRemarkParserGetNext).restype, LLVMRemarkParserGetNext.argtypes = LLVMRemarkEntryRef, [LLVMRemarkParserRef]
except AttributeError: pass

try: (LLVMRemarkParserHasError:=dll.LLVMRemarkParserHasError).restype, LLVMRemarkParserHasError.argtypes = LLVMBool, [LLVMRemarkParserRef]
except AttributeError: pass

try: (LLVMRemarkParserGetErrorMessage:=dll.LLVMRemarkParserGetErrorMessage).restype, LLVMRemarkParserGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMRemarkParserRef]
except AttributeError: pass

try: (LLVMRemarkParserDispose:=dll.LLVMRemarkParserDispose).restype, LLVMRemarkParserDispose.argtypes = None, [LLVMRemarkParserRef]
except AttributeError: pass

try: (LLVMRemarkVersion:=dll.LLVMRemarkVersion).restype, LLVMRemarkVersion.argtypes = uint32_t, []
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMLoadLibraryPermanently:=dll.LLVMLoadLibraryPermanently).restype, LLVMLoadLibraryPermanently.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMParseCommandLineOptions:=dll.LLVMParseCommandLineOptions).restype, LLVMParseCommandLineOptions.argtypes = None, [ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMSearchForAddressOfSymbol:=dll.LLVMSearchForAddressOfSymbol).restype, LLVMSearchForAddressOfSymbol.argtypes = ctypes.c_void_p, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMAddSymbol:=dll.LLVMAddSymbol).restype, LLVMAddSymbol.argtypes = None, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMGetErrorTypeId:=dll.LLVMGetErrorTypeId).restype, LLVMGetErrorTypeId.argtypes = LLVMErrorTypeId, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMConsumeError:=dll.LLVMConsumeError).restype, LLVMConsumeError.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMCantFail:=dll.LLVMCantFail).restype, LLVMCantFail.argtypes = None, [LLVMErrorRef]
except AttributeError: pass

try: (LLVMGetErrorMessage:=dll.LLVMGetErrorMessage).restype, LLVMGetErrorMessage.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMErrorRef]
except AttributeError: pass

try: (LLVMDisposeErrorMessage:=dll.LLVMDisposeErrorMessage).restype, LLVMDisposeErrorMessage.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetStringErrorTypeId:=dll.LLVMGetStringErrorTypeId).restype, LLVMGetStringErrorTypeId.argtypes = LLVMErrorTypeId, []
except AttributeError: pass

try: (LLVMCreateStringError:=dll.LLVMCreateStringError).restype, LLVMCreateStringError.argtypes = LLVMErrorRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

try: (LLVMInitializeAArch64TargetInfo:=dll.LLVMInitializeAArch64TargetInfo).restype, LLVMInitializeAArch64TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetInfo:=dll.LLVMInitializeAMDGPUTargetInfo).restype, LLVMInitializeAMDGPUTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetInfo:=dll.LLVMInitializeARMTargetInfo).restype, LLVMInitializeARMTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetInfo:=dll.LLVMInitializeAVRTargetInfo).restype, LLVMInitializeAVRTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetInfo:=dll.LLVMInitializeBPFTargetInfo).restype, LLVMInitializeBPFTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetInfo:=dll.LLVMInitializeHexagonTargetInfo).restype, LLVMInitializeHexagonTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetInfo:=dll.LLVMInitializeLanaiTargetInfo).restype, LLVMInitializeLanaiTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetInfo:=dll.LLVMInitializeLoongArchTargetInfo).restype, LLVMInitializeLoongArchTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetInfo:=dll.LLVMInitializeMipsTargetInfo).restype, LLVMInitializeMipsTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetInfo:=dll.LLVMInitializeMSP430TargetInfo).restype, LLVMInitializeMSP430TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetInfo:=dll.LLVMInitializeNVPTXTargetInfo).restype, LLVMInitializeNVPTXTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetInfo:=dll.LLVMInitializePowerPCTargetInfo).restype, LLVMInitializePowerPCTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetInfo:=dll.LLVMInitializeRISCVTargetInfo).restype, LLVMInitializeRISCVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetInfo:=dll.LLVMInitializeSparcTargetInfo).restype, LLVMInitializeSparcTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetInfo:=dll.LLVMInitializeSPIRVTargetInfo).restype, LLVMInitializeSPIRVTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetInfo:=dll.LLVMInitializeSystemZTargetInfo).restype, LLVMInitializeSystemZTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetInfo:=dll.LLVMInitializeVETargetInfo).restype, LLVMInitializeVETargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetInfo:=dll.LLVMInitializeWebAssemblyTargetInfo).restype, LLVMInitializeWebAssemblyTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetInfo:=dll.LLVMInitializeX86TargetInfo).restype, LLVMInitializeX86TargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetInfo:=dll.LLVMInitializeXCoreTargetInfo).restype, LLVMInitializeXCoreTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetInfo:=dll.LLVMInitializeM68kTargetInfo).restype, LLVMInitializeM68kTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetInfo:=dll.LLVMInitializeXtensaTargetInfo).restype, LLVMInitializeXtensaTargetInfo.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Target:=dll.LLVMInitializeAArch64Target).restype, LLVMInitializeAArch64Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTarget:=dll.LLVMInitializeAMDGPUTarget).restype, LLVMInitializeAMDGPUTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTarget:=dll.LLVMInitializeARMTarget).restype, LLVMInitializeARMTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTarget:=dll.LLVMInitializeAVRTarget).restype, LLVMInitializeAVRTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTarget:=dll.LLVMInitializeBPFTarget).restype, LLVMInitializeBPFTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTarget:=dll.LLVMInitializeHexagonTarget).restype, LLVMInitializeHexagonTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTarget:=dll.LLVMInitializeLanaiTarget).restype, LLVMInitializeLanaiTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTarget:=dll.LLVMInitializeLoongArchTarget).restype, LLVMInitializeLoongArchTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTarget:=dll.LLVMInitializeMipsTarget).restype, LLVMInitializeMipsTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Target:=dll.LLVMInitializeMSP430Target).restype, LLVMInitializeMSP430Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTarget:=dll.LLVMInitializeNVPTXTarget).restype, LLVMInitializeNVPTXTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTarget:=dll.LLVMInitializePowerPCTarget).restype, LLVMInitializePowerPCTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTarget:=dll.LLVMInitializeRISCVTarget).restype, LLVMInitializeRISCVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTarget:=dll.LLVMInitializeSparcTarget).restype, LLVMInitializeSparcTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTarget:=dll.LLVMInitializeSPIRVTarget).restype, LLVMInitializeSPIRVTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTarget:=dll.LLVMInitializeSystemZTarget).restype, LLVMInitializeSystemZTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETarget:=dll.LLVMInitializeVETarget).restype, LLVMInitializeVETarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTarget:=dll.LLVMInitializeWebAssemblyTarget).restype, LLVMInitializeWebAssemblyTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Target:=dll.LLVMInitializeX86Target).restype, LLVMInitializeX86Target.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTarget:=dll.LLVMInitializeXCoreTarget).restype, LLVMInitializeXCoreTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTarget:=dll.LLVMInitializeM68kTarget).restype, LLVMInitializeM68kTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTarget:=dll.LLVMInitializeXtensaTarget).restype, LLVMInitializeXtensaTarget.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64TargetMC:=dll.LLVMInitializeAArch64TargetMC).restype, LLVMInitializeAArch64TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUTargetMC:=dll.LLVMInitializeAMDGPUTargetMC).restype, LLVMInitializeAMDGPUTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMTargetMC:=dll.LLVMInitializeARMTargetMC).restype, LLVMInitializeARMTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRTargetMC:=dll.LLVMInitializeAVRTargetMC).restype, LLVMInitializeAVRTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFTargetMC:=dll.LLVMInitializeBPFTargetMC).restype, LLVMInitializeBPFTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonTargetMC:=dll.LLVMInitializeHexagonTargetMC).restype, LLVMInitializeHexagonTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiTargetMC:=dll.LLVMInitializeLanaiTargetMC).restype, LLVMInitializeLanaiTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchTargetMC:=dll.LLVMInitializeLoongArchTargetMC).restype, LLVMInitializeLoongArchTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsTargetMC:=dll.LLVMInitializeMipsTargetMC).restype, LLVMInitializeMipsTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430TargetMC:=dll.LLVMInitializeMSP430TargetMC).restype, LLVMInitializeMSP430TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXTargetMC:=dll.LLVMInitializeNVPTXTargetMC).restype, LLVMInitializeNVPTXTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCTargetMC:=dll.LLVMInitializePowerPCTargetMC).restype, LLVMInitializePowerPCTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVTargetMC:=dll.LLVMInitializeRISCVTargetMC).restype, LLVMInitializeRISCVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcTargetMC:=dll.LLVMInitializeSparcTargetMC).restype, LLVMInitializeSparcTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVTargetMC:=dll.LLVMInitializeSPIRVTargetMC).restype, LLVMInitializeSPIRVTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZTargetMC:=dll.LLVMInitializeSystemZTargetMC).restype, LLVMInitializeSystemZTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVETargetMC:=dll.LLVMInitializeVETargetMC).restype, LLVMInitializeVETargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyTargetMC:=dll.LLVMInitializeWebAssemblyTargetMC).restype, LLVMInitializeWebAssemblyTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86TargetMC:=dll.LLVMInitializeX86TargetMC).restype, LLVMInitializeX86TargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreTargetMC:=dll.LLVMInitializeXCoreTargetMC).restype, LLVMInitializeXCoreTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kTargetMC:=dll.LLVMInitializeM68kTargetMC).restype, LLVMInitializeM68kTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaTargetMC:=dll.LLVMInitializeXtensaTargetMC).restype, LLVMInitializeXtensaTargetMC.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmPrinter:=dll.LLVMInitializeAArch64AsmPrinter).restype, LLVMInitializeAArch64AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmPrinter:=dll.LLVMInitializeAMDGPUAsmPrinter).restype, LLVMInitializeAMDGPUAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmPrinter:=dll.LLVMInitializeARMAsmPrinter).restype, LLVMInitializeARMAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmPrinter:=dll.LLVMInitializeAVRAsmPrinter).restype, LLVMInitializeAVRAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmPrinter:=dll.LLVMInitializeBPFAsmPrinter).restype, LLVMInitializeBPFAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmPrinter:=dll.LLVMInitializeHexagonAsmPrinter).restype, LLVMInitializeHexagonAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmPrinter:=dll.LLVMInitializeLanaiAsmPrinter).restype, LLVMInitializeLanaiAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmPrinter:=dll.LLVMInitializeLoongArchAsmPrinter).restype, LLVMInitializeLoongArchAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmPrinter:=dll.LLVMInitializeMipsAsmPrinter).restype, LLVMInitializeMipsAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmPrinter:=dll.LLVMInitializeMSP430AsmPrinter).restype, LLVMInitializeMSP430AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeNVPTXAsmPrinter:=dll.LLVMInitializeNVPTXAsmPrinter).restype, LLVMInitializeNVPTXAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmPrinter:=dll.LLVMInitializePowerPCAsmPrinter).restype, LLVMInitializePowerPCAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmPrinter:=dll.LLVMInitializeRISCVAsmPrinter).restype, LLVMInitializeRISCVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmPrinter:=dll.LLVMInitializeSparcAsmPrinter).restype, LLVMInitializeSparcAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSPIRVAsmPrinter:=dll.LLVMInitializeSPIRVAsmPrinter).restype, LLVMInitializeSPIRVAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmPrinter:=dll.LLVMInitializeSystemZAsmPrinter).restype, LLVMInitializeSystemZAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmPrinter:=dll.LLVMInitializeVEAsmPrinter).restype, LLVMInitializeVEAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmPrinter:=dll.LLVMInitializeWebAssemblyAsmPrinter).restype, LLVMInitializeWebAssemblyAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmPrinter:=dll.LLVMInitializeX86AsmPrinter).restype, LLVMInitializeX86AsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreAsmPrinter:=dll.LLVMInitializeXCoreAsmPrinter).restype, LLVMInitializeXCoreAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmPrinter:=dll.LLVMInitializeM68kAsmPrinter).restype, LLVMInitializeM68kAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmPrinter:=dll.LLVMInitializeXtensaAsmPrinter).restype, LLVMInitializeXtensaAsmPrinter.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64AsmParser:=dll.LLVMInitializeAArch64AsmParser).restype, LLVMInitializeAArch64AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUAsmParser:=dll.LLVMInitializeAMDGPUAsmParser).restype, LLVMInitializeAMDGPUAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMAsmParser:=dll.LLVMInitializeARMAsmParser).restype, LLVMInitializeARMAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRAsmParser:=dll.LLVMInitializeAVRAsmParser).restype, LLVMInitializeAVRAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFAsmParser:=dll.LLVMInitializeBPFAsmParser).restype, LLVMInitializeBPFAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonAsmParser:=dll.LLVMInitializeHexagonAsmParser).restype, LLVMInitializeHexagonAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiAsmParser:=dll.LLVMInitializeLanaiAsmParser).restype, LLVMInitializeLanaiAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchAsmParser:=dll.LLVMInitializeLoongArchAsmParser).restype, LLVMInitializeLoongArchAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsAsmParser:=dll.LLVMInitializeMipsAsmParser).restype, LLVMInitializeMipsAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430AsmParser:=dll.LLVMInitializeMSP430AsmParser).restype, LLVMInitializeMSP430AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCAsmParser:=dll.LLVMInitializePowerPCAsmParser).restype, LLVMInitializePowerPCAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVAsmParser:=dll.LLVMInitializeRISCVAsmParser).restype, LLVMInitializeRISCVAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcAsmParser:=dll.LLVMInitializeSparcAsmParser).restype, LLVMInitializeSparcAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZAsmParser:=dll.LLVMInitializeSystemZAsmParser).restype, LLVMInitializeSystemZAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEAsmParser:=dll.LLVMInitializeVEAsmParser).restype, LLVMInitializeVEAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyAsmParser:=dll.LLVMInitializeWebAssemblyAsmParser).restype, LLVMInitializeWebAssemblyAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86AsmParser:=dll.LLVMInitializeX86AsmParser).restype, LLVMInitializeX86AsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kAsmParser:=dll.LLVMInitializeM68kAsmParser).restype, LLVMInitializeM68kAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaAsmParser:=dll.LLVMInitializeXtensaAsmParser).restype, LLVMInitializeXtensaAsmParser.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAArch64Disassembler:=dll.LLVMInitializeAArch64Disassembler).restype, LLVMInitializeAArch64Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAMDGPUDisassembler:=dll.LLVMInitializeAMDGPUDisassembler).restype, LLVMInitializeAMDGPUDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeARMDisassembler:=dll.LLVMInitializeARMDisassembler).restype, LLVMInitializeARMDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeAVRDisassembler:=dll.LLVMInitializeAVRDisassembler).restype, LLVMInitializeAVRDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeBPFDisassembler:=dll.LLVMInitializeBPFDisassembler).restype, LLVMInitializeBPFDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeHexagonDisassembler:=dll.LLVMInitializeHexagonDisassembler).restype, LLVMInitializeHexagonDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLanaiDisassembler:=dll.LLVMInitializeLanaiDisassembler).restype, LLVMInitializeLanaiDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeLoongArchDisassembler:=dll.LLVMInitializeLoongArchDisassembler).restype, LLVMInitializeLoongArchDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMipsDisassembler:=dll.LLVMInitializeMipsDisassembler).restype, LLVMInitializeMipsDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeMSP430Disassembler:=dll.LLVMInitializeMSP430Disassembler).restype, LLVMInitializeMSP430Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializePowerPCDisassembler:=dll.LLVMInitializePowerPCDisassembler).restype, LLVMInitializePowerPCDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeRISCVDisassembler:=dll.LLVMInitializeRISCVDisassembler).restype, LLVMInitializeRISCVDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSparcDisassembler:=dll.LLVMInitializeSparcDisassembler).restype, LLVMInitializeSparcDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeSystemZDisassembler:=dll.LLVMInitializeSystemZDisassembler).restype, LLVMInitializeSystemZDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeVEDisassembler:=dll.LLVMInitializeVEDisassembler).restype, LLVMInitializeVEDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeWebAssemblyDisassembler:=dll.LLVMInitializeWebAssemblyDisassembler).restype, LLVMInitializeWebAssemblyDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeX86Disassembler:=dll.LLVMInitializeX86Disassembler).restype, LLVMInitializeX86Disassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXCoreDisassembler:=dll.LLVMInitializeXCoreDisassembler).restype, LLVMInitializeXCoreDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeM68kDisassembler:=dll.LLVMInitializeM68kDisassembler).restype, LLVMInitializeM68kDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMInitializeXtensaDisassembler:=dll.LLVMInitializeXtensaDisassembler).restype, LLVMInitializeXtensaDisassembler.argtypes = None, []
except AttributeError: pass

try: (LLVMGetModuleDataLayout:=dll.LLVMGetModuleDataLayout).restype, LLVMGetModuleDataLayout.argtypes = LLVMTargetDataRef, [LLVMModuleRef]
except AttributeError: pass

try: (LLVMSetModuleDataLayout:=dll.LLVMSetModuleDataLayout).restype, LLVMSetModuleDataLayout.argtypes = None, [LLVMModuleRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMCreateTargetData:=dll.LLVMCreateTargetData).restype, LLVMCreateTargetData.argtypes = LLVMTargetDataRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMDisposeTargetData:=dll.LLVMDisposeTargetData).restype, LLVMDisposeTargetData.argtypes = None, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMAddTargetLibraryInfo:=dll.LLVMAddTargetLibraryInfo).restype, LLVMAddTargetLibraryInfo.argtypes = None, [LLVMTargetLibraryInfoRef, LLVMPassManagerRef]
except AttributeError: pass

try: (LLVMCopyStringRepOfTargetData:=dll.LLVMCopyStringRepOfTargetData).restype, LLVMCopyStringRepOfTargetData.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMByteOrder:=dll.LLVMByteOrder).restype, LLVMByteOrder.argtypes = enum_LLVMByteOrdering, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSize:=dll.LLVMPointerSize).restype, LLVMPointerSize.argtypes = ctypes.c_uint32, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMPointerSizeForAS:=dll.LLVMPointerSizeForAS).restype, LLVMPointerSizeForAS.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrType:=dll.LLVMIntPtrType).restype, LLVMIntPtrType.argtypes = LLVMTypeRef, [LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForAS:=dll.LLVMIntPtrTypeForAS).restype, LLVMIntPtrTypeForAS.argtypes = LLVMTypeRef, [LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMIntPtrTypeInContext:=dll.LLVMIntPtrTypeInContext).restype, LLVMIntPtrTypeInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef]
except AttributeError: pass

try: (LLVMIntPtrTypeForASInContext:=dll.LLVMIntPtrTypeForASInContext).restype, LLVMIntPtrTypeForASInContext.argtypes = LLVMTypeRef, [LLVMContextRef, LLVMTargetDataRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMSizeOfTypeInBits:=dll.LLVMSizeOfTypeInBits).restype, LLVMSizeOfTypeInBits.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMStoreSizeOfType:=dll.LLVMStoreSizeOfType).restype, LLVMStoreSizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABISizeOfType:=dll.LLVMABISizeOfType).restype, LLVMABISizeOfType.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMABIAlignmentOfType:=dll.LLVMABIAlignmentOfType).restype, LLVMABIAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMCallFrameAlignmentOfType:=dll.LLVMCallFrameAlignmentOfType).restype, LLVMCallFrameAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfType:=dll.LLVMPreferredAlignmentOfType).restype, LLVMPreferredAlignmentOfType.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef]
except AttributeError: pass

try: (LLVMPreferredAlignmentOfGlobal:=dll.LLVMPreferredAlignmentOfGlobal).restype, LLVMPreferredAlignmentOfGlobal.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMValueRef]
except AttributeError: pass

try: (LLVMElementAtOffset:=dll.LLVMElementAtOffset).restype, LLVMElementAtOffset.argtypes = ctypes.c_uint32, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint64]
except AttributeError: pass

try: (LLVMOffsetOfElement:=dll.LLVMOffsetOfElement).restype, LLVMOffsetOfElement.argtypes = ctypes.c_uint64, [LLVMTargetDataRef, LLVMTypeRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMGetFirstTarget:=dll.LLVMGetFirstTarget).restype, LLVMGetFirstTarget.argtypes = LLVMTargetRef, []
except AttributeError: pass

try: (LLVMGetNextTarget:=dll.LLVMGetNextTarget).restype, LLVMGetNextTarget.argtypes = LLVMTargetRef, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetFromName:=dll.LLVMGetTargetFromName).restype, LLVMGetTargetFromName.argtypes = LLVMTargetRef, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetTargetFromTriple:=dll.LLVMGetTargetFromTriple).restype, LLVMGetTargetFromTriple.argtypes = LLVMBool, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(LLVMTargetRef), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMGetTargetName:=dll.LLVMGetTargetName).restype, LLVMGetTargetName.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMGetTargetDescription:=dll.LLVMGetTargetDescription).restype, LLVMGetTargetDescription.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasJIT:=dll.LLVMTargetHasJIT).restype, LLVMTargetHasJIT.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasTargetMachine:=dll.LLVMTargetHasTargetMachine).restype, LLVMTargetHasTargetMachine.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMTargetHasAsmBackend:=dll.LLVMTargetHasAsmBackend).restype, LLVMTargetHasAsmBackend.argtypes = LLVMBool, [LLVMTargetRef]
except AttributeError: pass

try: (LLVMCreateTargetMachineOptions:=dll.LLVMCreateTargetMachineOptions).restype, LLVMCreateTargetMachineOptions.argtypes = LLVMTargetMachineOptionsRef, []
except AttributeError: pass

try: (LLVMDisposeTargetMachineOptions:=dll.LLVMDisposeTargetMachineOptions).restype, LLVMDisposeTargetMachineOptions.argtypes = None, [LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCPU:=dll.LLVMTargetMachineOptionsSetCPU).restype, LLVMTargetMachineOptionsSetCPU.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetFeatures:=dll.LLVMTargetMachineOptionsSetFeatures).restype, LLVMTargetMachineOptionsSetFeatures.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetABI:=dll.LLVMTargetMachineOptionsSetABI).restype, LLVMTargetMachineOptionsSetABI.argtypes = None, [LLVMTargetMachineOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeGenOptLevel:=dll.LLVMTargetMachineOptionsSetCodeGenOptLevel).restype, LLVMTargetMachineOptionsSetCodeGenOptLevel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeGenOptLevel]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetRelocMode:=dll.LLVMTargetMachineOptionsSetRelocMode).restype, LLVMTargetMachineOptionsSetRelocMode.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMRelocMode]
except AttributeError: pass

try: (LLVMTargetMachineOptionsSetCodeModel:=dll.LLVMTargetMachineOptionsSetCodeModel).restype, LLVMTargetMachineOptionsSetCodeModel.argtypes = None, [LLVMTargetMachineOptionsRef, LLVMCodeModel]
except AttributeError: pass

try: (LLVMCreateTargetMachineWithOptions:=dll.LLVMCreateTargetMachineWithOptions).restype, LLVMCreateTargetMachineWithOptions.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineOptionsRef]
except AttributeError: pass

try: (LLVMCreateTargetMachine:=dll.LLVMCreateTargetMachine).restype, LLVMCreateTargetMachine.argtypes = LLVMTargetMachineRef, [LLVMTargetRef, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), LLVMCodeGenOptLevel, LLVMRelocMode, LLVMCodeModel]
except AttributeError: pass

try: (LLVMDisposeTargetMachine:=dll.LLVMDisposeTargetMachine).restype, LLVMDisposeTargetMachine.argtypes = None, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTarget:=dll.LLVMGetTargetMachineTarget).restype, LLVMGetTargetMachineTarget.argtypes = LLVMTargetRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineTriple:=dll.LLVMGetTargetMachineTriple).restype, LLVMGetTargetMachineTriple.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineCPU:=dll.LLVMGetTargetMachineCPU).restype, LLVMGetTargetMachineCPU.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMGetTargetMachineFeatureString:=dll.LLVMGetTargetMachineFeatureString).restype, LLVMGetTargetMachineFeatureString.argtypes = ctypes.POINTER(ctypes.c_char), [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMCreateTargetDataLayout:=dll.LLVMCreateTargetDataLayout).restype, LLVMCreateTargetDataLayout.argtypes = LLVMTargetDataRef, [LLVMTargetMachineRef]
except AttributeError: pass

try: (LLVMSetTargetMachineAsmVerbosity:=dll.LLVMSetTargetMachineAsmVerbosity).restype, LLVMSetTargetMachineAsmVerbosity.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineFastISel:=dll.LLVMSetTargetMachineFastISel).restype, LLVMSetTargetMachineFastISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISel:=dll.LLVMSetTargetMachineGlobalISel).restype, LLVMSetTargetMachineGlobalISel.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMSetTargetMachineGlobalISelAbort:=dll.LLVMSetTargetMachineGlobalISelAbort).restype, LLVMSetTargetMachineGlobalISelAbort.argtypes = None, [LLVMTargetMachineRef, LLVMGlobalISelAbortMode]
except AttributeError: pass

try: (LLVMSetTargetMachineMachineOutliner:=dll.LLVMSetTargetMachineMachineOutliner).restype, LLVMSetTargetMachineMachineOutliner.argtypes = None, [LLVMTargetMachineRef, LLVMBool]
except AttributeError: pass

try: (LLVMTargetMachineEmitToFile:=dll.LLVMTargetMachineEmitToFile).restype, LLVMTargetMachineEmitToFile.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (LLVMTargetMachineEmitToMemoryBuffer:=dll.LLVMTargetMachineEmitToMemoryBuffer).restype, LLVMTargetMachineEmitToMemoryBuffer.argtypes = LLVMBool, [LLVMTargetMachineRef, LLVMModuleRef, LLVMCodeGenFileType, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(LLVMMemoryBufferRef)]
except AttributeError: pass

try: (LLVMGetDefaultTargetTriple:=dll.LLVMGetDefaultTargetTriple).restype, LLVMGetDefaultTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMNormalizeTargetTriple:=dll.LLVMNormalizeTargetTriple).restype, LLVMNormalizeTargetTriple.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMGetHostCPUName:=dll.LLVMGetHostCPUName).restype, LLVMGetHostCPUName.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMGetHostCPUFeatures:=dll.LLVMGetHostCPUFeatures).restype, LLVMGetHostCPUFeatures.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (LLVMAddAnalysisPasses:=dll.LLVMAddAnalysisPasses).restype, LLVMAddAnalysisPasses.argtypes = None, [LLVMTargetMachineRef, LLVMPassManagerRef]
except AttributeError: pass

class struct_LLVMOpaquePassBuilderOptions(Struct): pass
LLVMPassBuilderOptionsRef = ctypes.POINTER(struct_LLVMOpaquePassBuilderOptions)
try: (LLVMRunPasses:=dll.LLVMRunPasses).restype, LLVMRunPasses.argtypes = LLVMErrorRef, [LLVMModuleRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineRef, LLVMPassBuilderOptionsRef]
except AttributeError: pass

try: (LLVMRunPassesOnFunction:=dll.LLVMRunPassesOnFunction).restype, LLVMRunPassesOnFunction.argtypes = LLVMErrorRef, [LLVMValueRef, ctypes.POINTER(ctypes.c_char), LLVMTargetMachineRef, LLVMPassBuilderOptionsRef]
except AttributeError: pass

try: (LLVMCreatePassBuilderOptions:=dll.LLVMCreatePassBuilderOptions).restype, LLVMCreatePassBuilderOptions.argtypes = LLVMPassBuilderOptionsRef, []
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetVerifyEach:=dll.LLVMPassBuilderOptionsSetVerifyEach).restype, LLVMPassBuilderOptionsSetVerifyEach.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetDebugLogging:=dll.LLVMPassBuilderOptionsSetDebugLogging).restype, LLVMPassBuilderOptionsSetDebugLogging.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetAAPipeline:=dll.LLVMPassBuilderOptionsSetAAPipeline).restype, LLVMPassBuilderOptionsSetAAPipeline.argtypes = None, [LLVMPassBuilderOptionsRef, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetLoopInterleaving:=dll.LLVMPassBuilderOptionsSetLoopInterleaving).restype, LLVMPassBuilderOptionsSetLoopInterleaving.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetLoopVectorization:=dll.LLVMPassBuilderOptionsSetLoopVectorization).restype, LLVMPassBuilderOptionsSetLoopVectorization.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetSLPVectorization:=dll.LLVMPassBuilderOptionsSetSLPVectorization).restype, LLVMPassBuilderOptionsSetSLPVectorization.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetLoopUnrolling:=dll.LLVMPassBuilderOptionsSetLoopUnrolling).restype, LLVMPassBuilderOptionsSetLoopUnrolling.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll:=dll.LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll).restype, LLVMPassBuilderOptionsSetForgetAllSCEVInLoopUnroll.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetLicmMssaOptCap:=dll.LLVMPassBuilderOptionsSetLicmMssaOptCap).restype, LLVMPassBuilderOptionsSetLicmMssaOptCap.argtypes = None, [LLVMPassBuilderOptionsRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap:=dll.LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap).restype, LLVMPassBuilderOptionsSetLicmMssaNoAccForPromotionCap.argtypes = None, [LLVMPassBuilderOptionsRef, ctypes.c_uint32]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetCallGraphProfile:=dll.LLVMPassBuilderOptionsSetCallGraphProfile).restype, LLVMPassBuilderOptionsSetCallGraphProfile.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetMergeFunctions:=dll.LLVMPassBuilderOptionsSetMergeFunctions).restype, LLVMPassBuilderOptionsSetMergeFunctions.argtypes = None, [LLVMPassBuilderOptionsRef, LLVMBool]
except AttributeError: pass

try: (LLVMPassBuilderOptionsSetInlinerThreshold:=dll.LLVMPassBuilderOptionsSetInlinerThreshold).restype, LLVMPassBuilderOptionsSetInlinerThreshold.argtypes = None, [LLVMPassBuilderOptionsRef, ctypes.c_int32]
except AttributeError: pass

try: (LLVMDisposePassBuilderOptions:=dll.LLVMDisposePassBuilderOptions).restype, LLVMDisposePassBuilderOptions.argtypes = None, [LLVMPassBuilderOptionsRef]
except AttributeError: pass

try: (imaxabs:=dll.imaxabs).restype, imaxabs.argtypes = intmax_t, [intmax_t]
except AttributeError: pass

try: (imaxdiv:=dll.imaxdiv).restype, imaxdiv.argtypes = imaxdiv_t, [intmax_t, intmax_t]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (strtoimax:=dll.strtoimax).restype, strtoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (strtoumax:=dll.strtoumax).restype, strtoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoimax:=dll.wcstoimax).restype, wcstoimax.argtypes = intmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (wcstoumax:=dll.wcstoumax).restype, wcstoumax.argtypes = uintmax_t, [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

class llvm_blake3_chunk_state(Struct): pass
llvm_blake3_chunk_state._fields_ = [
  ('cv', (uint32_t * 8)),
  ('chunk_counter', uint64_t),
  ('buf', (uint8_t * 64)),
  ('buf_len', uint8_t),
  ('blocks_compressed', uint8_t),
  ('flags', uint8_t),
]
class llvm_blake3_hasher(Struct): pass
llvm_blake3_hasher._fields_ = [
  ('key', (uint32_t * 8)),
  ('chunk', llvm_blake3_chunk_state),
  ('cv_stack_len', uint8_t),
  ('cv_stack', (uint8_t * 1760)),
]
try: (llvm_blake3_version:=dll.llvm_blake3_version).restype, llvm_blake3_version.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (llvm_blake3_hasher_init:=dll.llvm_blake3_hasher_init).restype, llvm_blake3_hasher_init.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher)]
except AttributeError: pass

try: (llvm_blake3_hasher_init_keyed:=dll.llvm_blake3_hasher_init_keyed).restype, llvm_blake3_hasher_init_keyed.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), (uint8_t * 32)]
except AttributeError: pass

try: (llvm_blake3_hasher_init_derive_key:=dll.llvm_blake3_hasher_init_derive_key).restype, llvm_blake3_hasher_init_derive_key.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (llvm_blake3_hasher_init_derive_key_raw:=dll.llvm_blake3_hasher_init_derive_key_raw).restype, llvm_blake3_hasher_init_derive_key_raw.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), ctypes.c_void_p, size_t]
except AttributeError: pass

try: (llvm_blake3_hasher_update:=dll.llvm_blake3_hasher_update).restype, llvm_blake3_hasher_update.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), ctypes.c_void_p, size_t]
except AttributeError: pass

try: (llvm_blake3_hasher_finalize:=dll.llvm_blake3_hasher_finalize).restype, llvm_blake3_hasher_finalize.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), ctypes.POINTER(uint8_t), size_t]
except AttributeError: pass

try: (llvm_blake3_hasher_finalize_seek:=dll.llvm_blake3_hasher_finalize_seek).restype, llvm_blake3_hasher_finalize_seek.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher), uint64_t, ctypes.POINTER(uint8_t), size_t]
except AttributeError: pass

try: (llvm_blake3_hasher_reset:=dll.llvm_blake3_hasher_reset).restype, llvm_blake3_hasher_reset.argtypes = None, [ctypes.POINTER(llvm_blake3_hasher)]
except AttributeError: pass

try: (select:=dll.select).restype, select.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int32, [ctypes.c_int32, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(__sigset_t)]
except AttributeError: pass

lto_bool_t = ctypes.c_bool
lto_symbol_attributes = CEnum(ctypes.c_uint32)
LTO_SYMBOL_ALIGNMENT_MASK = lto_symbol_attributes.define('LTO_SYMBOL_ALIGNMENT_MASK', 31)
LTO_SYMBOL_PERMISSIONS_MASK = lto_symbol_attributes.define('LTO_SYMBOL_PERMISSIONS_MASK', 224)
LTO_SYMBOL_PERMISSIONS_CODE = lto_symbol_attributes.define('LTO_SYMBOL_PERMISSIONS_CODE', 160)
LTO_SYMBOL_PERMISSIONS_DATA = lto_symbol_attributes.define('LTO_SYMBOL_PERMISSIONS_DATA', 192)
LTO_SYMBOL_PERMISSIONS_RODATA = lto_symbol_attributes.define('LTO_SYMBOL_PERMISSIONS_RODATA', 128)
LTO_SYMBOL_DEFINITION_MASK = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_MASK', 1792)
LTO_SYMBOL_DEFINITION_REGULAR = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_REGULAR', 256)
LTO_SYMBOL_DEFINITION_TENTATIVE = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_TENTATIVE', 512)
LTO_SYMBOL_DEFINITION_WEAK = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_WEAK', 768)
LTO_SYMBOL_DEFINITION_UNDEFINED = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_UNDEFINED', 1024)
LTO_SYMBOL_DEFINITION_WEAKUNDEF = lto_symbol_attributes.define('LTO_SYMBOL_DEFINITION_WEAKUNDEF', 1280)
LTO_SYMBOL_SCOPE_MASK = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_MASK', 14336)
LTO_SYMBOL_SCOPE_INTERNAL = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_INTERNAL', 2048)
LTO_SYMBOL_SCOPE_HIDDEN = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_HIDDEN', 4096)
LTO_SYMBOL_SCOPE_PROTECTED = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_PROTECTED', 8192)
LTO_SYMBOL_SCOPE_DEFAULT = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_DEFAULT', 6144)
LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN = lto_symbol_attributes.define('LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN', 10240)
LTO_SYMBOL_COMDAT = lto_symbol_attributes.define('LTO_SYMBOL_COMDAT', 16384)
LTO_SYMBOL_ALIAS = lto_symbol_attributes.define('LTO_SYMBOL_ALIAS', 32768)

lto_debug_model = CEnum(ctypes.c_uint32)
LTO_DEBUG_MODEL_NONE = lto_debug_model.define('LTO_DEBUG_MODEL_NONE', 0)
LTO_DEBUG_MODEL_DWARF = lto_debug_model.define('LTO_DEBUG_MODEL_DWARF', 1)

lto_codegen_model = CEnum(ctypes.c_uint32)
LTO_CODEGEN_PIC_MODEL_STATIC = lto_codegen_model.define('LTO_CODEGEN_PIC_MODEL_STATIC', 0)
LTO_CODEGEN_PIC_MODEL_DYNAMIC = lto_codegen_model.define('LTO_CODEGEN_PIC_MODEL_DYNAMIC', 1)
LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC = lto_codegen_model.define('LTO_CODEGEN_PIC_MODEL_DYNAMIC_NO_PIC', 2)
LTO_CODEGEN_PIC_MODEL_DEFAULT = lto_codegen_model.define('LTO_CODEGEN_PIC_MODEL_DEFAULT', 3)

class struct_LLVMOpaqueLTOModule(Struct): pass
lto_module_t = ctypes.POINTER(struct_LLVMOpaqueLTOModule)
class struct_LLVMOpaqueLTOCodeGenerator(Struct): pass
lto_code_gen_t = ctypes.POINTER(struct_LLVMOpaqueLTOCodeGenerator)
class struct_LLVMOpaqueThinLTOCodeGenerator(Struct): pass
thinlto_code_gen_t = ctypes.POINTER(struct_LLVMOpaqueThinLTOCodeGenerator)
try: (lto_get_version:=dll.lto_get_version).restype, lto_get_version.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (lto_get_error_message:=dll.lto_get_error_message).restype, lto_get_error_message.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

try: (lto_module_is_object_file:=dll.lto_module_is_object_file).restype, lto_module_is_object_file.argtypes = lto_bool_t, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_module_is_object_file_for_target:=dll.lto_module_is_object_file_for_target).restype, lto_module_is_object_file_for_target.argtypes = lto_bool_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_module_has_objc_category:=dll.lto_module_has_objc_category).restype, lto_module_has_objc_category.argtypes = lto_bool_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (lto_module_is_object_file_in_memory:=dll.lto_module_is_object_file_in_memory).restype, lto_module_is_object_file_in_memory.argtypes = lto_bool_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (lto_module_is_object_file_in_memory_for_target:=dll.lto_module_is_object_file_in_memory_for_target).restype, lto_module_is_object_file_in_memory_for_target.argtypes = lto_bool_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_module_create:=dll.lto_module_create).restype, lto_module_create.argtypes = lto_module_t, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_module_create_from_memory:=dll.lto_module_create_from_memory).restype, lto_module_create_from_memory.argtypes = lto_module_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (lto_module_create_from_memory_with_path:=dll.lto_module_create_from_memory_with_path).restype, lto_module_create_from_memory_with_path.argtypes = lto_module_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_module_create_in_local_context:=dll.lto_module_create_in_local_context).restype, lto_module_create_in_local_context.argtypes = lto_module_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_module_create_in_codegen_context:=dll.lto_module_create_in_codegen_context).restype, lto_module_create_in_codegen_context.argtypes = lto_module_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char), lto_code_gen_t]
except AttributeError: pass

try: (lto_module_create_from_fd:=dll.lto_module_create_from_fd).restype, lto_module_create_from_fd.argtypes = lto_module_t, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

off_t = ctypes.c_int64
try: (lto_module_create_from_fd_at_offset:=dll.lto_module_create_from_fd_at_offset).restype, lto_module_create_from_fd_at_offset.argtypes = lto_module_t, [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t, size_t, off_t]
except AttributeError: pass

try: (lto_module_dispose:=dll.lto_module_dispose).restype, lto_module_dispose.argtypes = None, [lto_module_t]
except AttributeError: pass

try: (lto_module_get_target_triple:=dll.lto_module_get_target_triple).restype, lto_module_get_target_triple.argtypes = ctypes.POINTER(ctypes.c_char), [lto_module_t]
except AttributeError: pass

try: (lto_module_set_target_triple:=dll.lto_module_set_target_triple).restype, lto_module_set_target_triple.argtypes = None, [lto_module_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_module_get_num_symbols:=dll.lto_module_get_num_symbols).restype, lto_module_get_num_symbols.argtypes = ctypes.c_uint32, [lto_module_t]
except AttributeError: pass

try: (lto_module_get_symbol_name:=dll.lto_module_get_symbol_name).restype, lto_module_get_symbol_name.argtypes = ctypes.POINTER(ctypes.c_char), [lto_module_t, ctypes.c_uint32]
except AttributeError: pass

try: (lto_module_get_symbol_attribute:=dll.lto_module_get_symbol_attribute).restype, lto_module_get_symbol_attribute.argtypes = lto_symbol_attributes, [lto_module_t, ctypes.c_uint32]
except AttributeError: pass

try: (lto_module_get_linkeropts:=dll.lto_module_get_linkeropts).restype, lto_module_get_linkeropts.argtypes = ctypes.POINTER(ctypes.c_char), [lto_module_t]
except AttributeError: pass

try: (lto_module_get_macho_cputype:=dll.lto_module_get_macho_cputype).restype, lto_module_get_macho_cputype.argtypes = lto_bool_t, [lto_module_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError: pass

try: (lto_module_has_ctor_dtor:=dll.lto_module_has_ctor_dtor).restype, lto_module_has_ctor_dtor.argtypes = lto_bool_t, [lto_module_t]
except AttributeError: pass

lto_codegen_diagnostic_severity_t = CEnum(ctypes.c_uint32)
LTO_DS_ERROR = lto_codegen_diagnostic_severity_t.define('LTO_DS_ERROR', 0)
LTO_DS_WARNING = lto_codegen_diagnostic_severity_t.define('LTO_DS_WARNING', 1)
LTO_DS_REMARK = lto_codegen_diagnostic_severity_t.define('LTO_DS_REMARK', 3)
LTO_DS_NOTE = lto_codegen_diagnostic_severity_t.define('LTO_DS_NOTE', 2)

lto_diagnostic_handler_t = ctypes.CFUNCTYPE(None, lto_codegen_diagnostic_severity_t, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p)
try: (lto_codegen_set_diagnostic_handler:=dll.lto_codegen_set_diagnostic_handler).restype, lto_codegen_set_diagnostic_handler.argtypes = None, [lto_code_gen_t, lto_diagnostic_handler_t, ctypes.c_void_p]
except AttributeError: pass

try: (lto_codegen_create:=dll.lto_codegen_create).restype, lto_codegen_create.argtypes = lto_code_gen_t, []
except AttributeError: pass

try: (lto_codegen_create_in_local_context:=dll.lto_codegen_create_in_local_context).restype, lto_codegen_create_in_local_context.argtypes = lto_code_gen_t, []
except AttributeError: pass

try: (lto_codegen_dispose:=dll.lto_codegen_dispose).restype, lto_codegen_dispose.argtypes = None, [lto_code_gen_t]
except AttributeError: pass

try: (lto_codegen_add_module:=dll.lto_codegen_add_module).restype, lto_codegen_add_module.argtypes = lto_bool_t, [lto_code_gen_t, lto_module_t]
except AttributeError: pass

try: (lto_codegen_set_module:=dll.lto_codegen_set_module).restype, lto_codegen_set_module.argtypes = None, [lto_code_gen_t, lto_module_t]
except AttributeError: pass

try: (lto_codegen_set_debug_model:=dll.lto_codegen_set_debug_model).restype, lto_codegen_set_debug_model.argtypes = lto_bool_t, [lto_code_gen_t, lto_debug_model]
except AttributeError: pass

try: (lto_codegen_set_pic_model:=dll.lto_codegen_set_pic_model).restype, lto_codegen_set_pic_model.argtypes = lto_bool_t, [lto_code_gen_t, lto_codegen_model]
except AttributeError: pass

try: (lto_codegen_set_cpu:=dll.lto_codegen_set_cpu).restype, lto_codegen_set_cpu.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_codegen_set_assembler_path:=dll.lto_codegen_set_assembler_path).restype, lto_codegen_set_assembler_path.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_codegen_set_assembler_args:=dll.lto_codegen_set_assembler_args).restype, lto_codegen_set_assembler_args.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (lto_codegen_add_must_preserve_symbol:=dll.lto_codegen_add_must_preserve_symbol).restype, lto_codegen_add_must_preserve_symbol.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_codegen_write_merged_modules:=dll.lto_codegen_write_merged_modules).restype, lto_codegen_write_merged_modules.argtypes = lto_bool_t, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_codegen_compile:=dll.lto_codegen_compile).restype, lto_codegen_compile.argtypes = ctypes.c_void_p, [lto_code_gen_t, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (lto_codegen_compile_to_file:=dll.lto_codegen_compile_to_file).restype, lto_codegen_compile_to_file.argtypes = lto_bool_t, [lto_code_gen_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (lto_codegen_optimize:=dll.lto_codegen_optimize).restype, lto_codegen_optimize.argtypes = lto_bool_t, [lto_code_gen_t]
except AttributeError: pass

try: (lto_codegen_compile_optimized:=dll.lto_codegen_compile_optimized).restype, lto_codegen_compile_optimized.argtypes = ctypes.c_void_p, [lto_code_gen_t, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (lto_api_version:=dll.lto_api_version).restype, lto_api_version.argtypes = ctypes.c_uint32, []
except AttributeError: pass

try: (lto_set_debug_options:=dll.lto_set_debug_options).restype, lto_set_debug_options.argtypes = None, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (lto_codegen_debug_options:=dll.lto_codegen_debug_options).restype, lto_codegen_debug_options.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_codegen_debug_options_array:=dll.lto_codegen_debug_options_array).restype, lto_codegen_debug_options_array.argtypes = None, [lto_code_gen_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (lto_initialize_disassembler:=dll.lto_initialize_disassembler).restype, lto_initialize_disassembler.argtypes = None, []
except AttributeError: pass

try: (lto_codegen_set_should_internalize:=dll.lto_codegen_set_should_internalize).restype, lto_codegen_set_should_internalize.argtypes = None, [lto_code_gen_t, lto_bool_t]
except AttributeError: pass

try: (lto_codegen_set_should_embed_uselists:=dll.lto_codegen_set_should_embed_uselists).restype, lto_codegen_set_should_embed_uselists.argtypes = None, [lto_code_gen_t, lto_bool_t]
except AttributeError: pass

class struct_LLVMOpaqueLTOInput(Struct): pass
lto_input_t = ctypes.POINTER(struct_LLVMOpaqueLTOInput)
try: (lto_input_create:=dll.lto_input_create).restype, lto_input_create.argtypes = lto_input_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (lto_input_dispose:=dll.lto_input_dispose).restype, lto_input_dispose.argtypes = None, [lto_input_t]
except AttributeError: pass

try: (lto_input_get_num_dependent_libraries:=dll.lto_input_get_num_dependent_libraries).restype, lto_input_get_num_dependent_libraries.argtypes = ctypes.c_uint32, [lto_input_t]
except AttributeError: pass

try: (lto_input_get_dependent_library:=dll.lto_input_get_dependent_library).restype, lto_input_get_dependent_library.argtypes = ctypes.POINTER(ctypes.c_char), [lto_input_t, size_t, ctypes.POINTER(size_t)]
except AttributeError: pass

try: (lto_runtime_lib_symbols_list:=dll.lto_runtime_lib_symbols_list).restype, lto_runtime_lib_symbols_list.argtypes = ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), [ctypes.POINTER(size_t)]
except AttributeError: pass

class LTOObjectBuffer(Struct): pass
LTOObjectBuffer._fields_ = [
  ('Buffer', ctypes.POINTER(ctypes.c_char)),
  ('Size', size_t),
]
try: (thinlto_create_codegen:=dll.thinlto_create_codegen).restype, thinlto_create_codegen.argtypes = thinlto_code_gen_t, []
except AttributeError: pass

try: (thinlto_codegen_dispose:=dll.thinlto_codegen_dispose).restype, thinlto_codegen_dispose.argtypes = None, [thinlto_code_gen_t]
except AttributeError: pass

try: (thinlto_codegen_add_module:=dll.thinlto_codegen_add_module).restype, thinlto_codegen_add_module.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

try: (thinlto_codegen_process:=dll.thinlto_codegen_process).restype, thinlto_codegen_process.argtypes = None, [thinlto_code_gen_t]
except AttributeError: pass

try: (thinlto_module_get_num_objects:=dll.thinlto_module_get_num_objects).restype, thinlto_module_get_num_objects.argtypes = ctypes.c_uint32, [thinlto_code_gen_t]
except AttributeError: pass

try: (thinlto_module_get_object:=dll.thinlto_module_get_object).restype, thinlto_module_get_object.argtypes = LTOObjectBuffer, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

try: (thinlto_module_get_num_object_files:=dll.thinlto_module_get_num_object_files).restype, thinlto_module_get_num_object_files.argtypes = ctypes.c_uint32, [thinlto_code_gen_t]
except AttributeError: pass

try: (thinlto_module_get_object_file:=dll.thinlto_module_get_object_file).restype, thinlto_module_get_object_file.argtypes = ctypes.POINTER(ctypes.c_char), [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

try: (thinlto_codegen_set_pic_model:=dll.thinlto_codegen_set_pic_model).restype, thinlto_codegen_set_pic_model.argtypes = lto_bool_t, [thinlto_code_gen_t, lto_codegen_model]
except AttributeError: pass

try: (thinlto_codegen_set_savetemps_dir:=dll.thinlto_codegen_set_savetemps_dir).restype, thinlto_codegen_set_savetemps_dir.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (thinlto_set_generated_objects_dir:=dll.thinlto_set_generated_objects_dir).restype, thinlto_set_generated_objects_dir.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (thinlto_codegen_set_cpu:=dll.thinlto_codegen_set_cpu).restype, thinlto_codegen_set_cpu.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (thinlto_codegen_disable_codegen:=dll.thinlto_codegen_disable_codegen).restype, thinlto_codegen_disable_codegen.argtypes = None, [thinlto_code_gen_t, lto_bool_t]
except AttributeError: pass

try: (thinlto_codegen_set_codegen_only:=dll.thinlto_codegen_set_codegen_only).restype, thinlto_codegen_set_codegen_only.argtypes = None, [thinlto_code_gen_t, lto_bool_t]
except AttributeError: pass

try: (thinlto_debug_options:=dll.thinlto_debug_options).restype, thinlto_debug_options.argtypes = None, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int32]
except AttributeError: pass

try: (lto_module_is_thinlto:=dll.lto_module_is_thinlto).restype, lto_module_is_thinlto.argtypes = lto_bool_t, [lto_module_t]
except AttributeError: pass

try: (thinlto_codegen_add_must_preserve_symbol:=dll.thinlto_codegen_add_must_preserve_symbol).restype, thinlto_codegen_add_must_preserve_symbol.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

try: (thinlto_codegen_add_cross_referenced_symbol:=dll.thinlto_codegen_add_cross_referenced_symbol).restype, thinlto_codegen_add_cross_referenced_symbol.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError: pass

try: (thinlto_codegen_set_cache_dir:=dll.thinlto_codegen_set_cache_dir).restype, thinlto_codegen_set_cache_dir.argtypes = None, [thinlto_code_gen_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

try: (thinlto_codegen_set_cache_pruning_interval:=dll.thinlto_codegen_set_cache_pruning_interval).restype, thinlto_codegen_set_cache_pruning_interval.argtypes = None, [thinlto_code_gen_t, ctypes.c_int32]
except AttributeError: pass

try: (thinlto_codegen_set_final_cache_size_relative_to_available_space:=dll.thinlto_codegen_set_final_cache_size_relative_to_available_space).restype, thinlto_codegen_set_final_cache_size_relative_to_available_space.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

try: (thinlto_codegen_set_cache_entry_expiration:=dll.thinlto_codegen_set_cache_entry_expiration).restype, thinlto_codegen_set_cache_entry_expiration.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

try: (thinlto_codegen_set_cache_size_bytes:=dll.thinlto_codegen_set_cache_size_bytes).restype, thinlto_codegen_set_cache_size_bytes.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

try: (thinlto_codegen_set_cache_size_megabytes:=dll.thinlto_codegen_set_cache_size_megabytes).restype, thinlto_codegen_set_cache_size_megabytes.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

try: (thinlto_codegen_set_cache_size_files:=dll.thinlto_codegen_set_cache_size_files).restype, thinlto_codegen_set_cache_size_files.argtypes = None, [thinlto_code_gen_t, ctypes.c_uint32]
except AttributeError: pass

LLVMDisassembler_Option_UseMarkup = 1
LLVMDisassembler_Option_PrintImmHex = 2
LLVMDisassembler_Option_AsmPrinterVariant = 4
LLVMDisassembler_Option_SetInstrComments = 8
LLVMDisassembler_Option_PrintLatency = 16
LLVMDisassembler_Option_Color = 32
LLVMDisassembler_VariantKind_None = 0
LLVMDisassembler_VariantKind_ARM_HI16 = 1
LLVMDisassembler_VariantKind_ARM_LO16 = 2
LLVMDisassembler_VariantKind_ARM64_PAGE = 1
LLVMDisassembler_VariantKind_ARM64_PAGEOFF = 2
LLVMDisassembler_VariantKind_ARM64_GOTPAGE = 3
LLVMDisassembler_VariantKind_ARM64_GOTPAGEOFF = 4
LLVMDisassembler_VariantKind_ARM64_TLVP = 5
LLVMDisassembler_VariantKind_ARM64_TLVOFF = 6
LLVMDisassembler_ReferenceType_InOut_None = 0
LLVMDisassembler_ReferenceType_In_Branch = 1
LLVMDisassembler_ReferenceType_In_PCrel_Load = 2
LLVMDisassembler_ReferenceType_In_ARM64_ADRP = 0x100000001
LLVMDisassembler_ReferenceType_In_ARM64_ADDXri = 0x100000002
LLVMDisassembler_ReferenceType_In_ARM64_LDRXui = 0x100000003
LLVMDisassembler_ReferenceType_In_ARM64_LDRXl = 0x100000004
LLVMDisassembler_ReferenceType_In_ARM64_ADR = 0x100000005
LLVMDisassembler_ReferenceType_Out_SymbolStub = 1
LLVMDisassembler_ReferenceType_Out_LitPool_SymAddr = 2
LLVMDisassembler_ReferenceType_Out_LitPool_CstrAddr = 3
LLVMDisassembler_ReferenceType_Out_Objc_CFString_Ref = 4
LLVMDisassembler_ReferenceType_Out_Objc_Message = 5
LLVMDisassembler_ReferenceType_Out_Objc_Message_Ref = 6
LLVMDisassembler_ReferenceType_Out_Objc_Selector_Ref = 7
LLVMDisassembler_ReferenceType_Out_Objc_Class_Ref = 8
LLVMDisassembler_ReferenceType_DeMangled_Name = 9
LLVMErrorSuccess = 0
REMARKS_API_VERSION = 1
LLVM_BLAKE3_VERSION_STRING = "1.3.1"
LLVM_BLAKE3_KEY_LEN = 32
LLVM_BLAKE3_OUT_LEN = 32
LLVM_BLAKE3_BLOCK_LEN = 64
LLVM_BLAKE3_CHUNK_LEN = 1024
LLVM_BLAKE3_MAX_DEPTH = 54
LTO_API_VERSION = 29