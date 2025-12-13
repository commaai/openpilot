import ctypes, ctypes.util, sys

cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
libsys = ctypes.CDLL(ctypes.util.find_library("System"))

kern_return_t = ctypes.c_int
mach_port_t = ctypes.c_uint
io_object_t = mach_port_t
io_service_t = io_object_t
io_connect_t = mach_port_t
CFMutableDictionaryRef = ctypes.c_void_p
CFStringRef = ctypes.c_void_p

kIOMasterPortDefault = mach_port_t(0)

libsys.mach_task_self_.restype = mach_port_t

iokit.IOServiceNameMatching.argtypes = [ctypes.c_char_p]
iokit.IOServiceNameMatching.restype  = CFMutableDictionaryRef

iokit.IOServiceGetMatchingService.argtypes = [mach_port_t, CFMutableDictionaryRef]
iokit.IOServiceGetMatchingService.restype  = io_service_t

iokit.IOObjectRelease.argtypes = [io_object_t]
iokit.IOObjectRelease.restype  = kern_return_t

iokit.IOServiceOpen.argtypes = [io_service_t, mach_port_t, ctypes.c_uint32, ctypes.POINTER(io_connect_t)]
iokit.IOServiceOpen.restype  = kern_return_t

iokit.IOConnectCallMethod.argtypes = [io_connect_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32, ctypes.c_void_p,
  ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
iokit.IOConnectCallMethod.restype = kern_return_t

def open_userclient_by_name(name: str, uc_type: int = 0) -> io_connect_t:
  mdict = iokit.IOServiceNameMatching(name.encode("utf-8"))
  if not mdict: raise RuntimeError("IOServiceNameMatching returned NULL")

  # Grab the first matching service
  service = iokit.IOServiceGetMatchingService(kIOMasterPortDefault, mdict)
  if not service: raise RuntimeError(f'service "{name}" not found')

  # print("lol", service)
  # print(libsys.mach_task_self_)
  # cast libsys.mach_task_self_ to uint and print
  # print("lol", ctypes.cast(libsys.mach_task_self_, ctypes.POINTER(ctypes.c_uint)).contents.value)

  try:
    # Open user client (type -> passed to NewUserClient_Impl)
    conn = io_connect_t(0)
    # print("lol", libsys.mach_task_self_)
    kr = iokit.IOServiceOpen(service, ctypes.cast(libsys.mach_task_self_, ctypes.POINTER(ctypes.c_uint)).contents.value,
                             ctypes.c_uint32(uc_type), ctypes.byref(conn))
    if kr != 0: raise OSError(kr, f"IOServiceOpen failed (0x{kr:08x})")
    return conn
  finally: iokit.IOObjectRelease(service)

def external_method(conn: io_connect_t, selector: int = 0) -> int:
  # no scalars in/out, no struct in/out — just ping selector 0
  in_scalars = ctypes.POINTER(ctypes.c_uint64)()  # NULL
  out_scalars = (ctypes.c_uint64 * 1)()       # space if driver returns something
  out_scalars_cnt = ctypes.c_uint32(0)      # driver can set this

  return iokit.IOConnectCallMethod(conn, ctypes.c_uint32(selector), in_scalars, ctypes.c_uint32(0), None, ctypes.c_size_t(0),
    out_scalars, ctypes.byref(out_scalars_cnt), None, ctypes.byref(ctypes.c_size_t(0)))

def close_userclient(conn: io_connect_t) -> None:
  # IOServiceClose is a macro; exported symbol is IOServiceClose in IOKit
  iokit.IOServiceClose.argtypes = [io_connect_t]
  iokit.IOServiceClose.restype  = kern_return_t
  iokit.IOServiceClose(conn)

if __name__ == "__main__":
  try:
    conn = open_userclient_by_name("tinygpu", uc_type=0)
    kr = external_method(conn, selector=0)
    print(f"ExternalMethod(0) -> 0x{kr:08x}")
  except Exception as e:
    print(e)
    sys.exit(1)
  finally:
    if 'conn' in locals() and conn.value: close_userclient(conn)
