import ctypes
from subprocess import check_output
from hexdump import hexdump

def get_pid(name):
  try:
    output = check_output(["pgrep", name])
    return int(output)
  except:
    return None

from ctypes.util import find_library
libc = ctypes.CDLL(find_library('c'))

amfid_pid = get_pid("amfid")

task = ctypes.c_uint32()
mytask = libc.mach_task_self()
ret = libc.task_for_pid(mytask, ctypes.c_int(amfid_pid), ctypes.pointer(task))
print(amfid_pid, ret, task, mytask)

#myport = libc.mach_task_self()

class vm_region_submap_short_info_data_64(ctypes.Structure):
  _pack_ = 1
  _fields_ = [
      ("protection", ctypes.c_uint32),
      ("max_protection", ctypes.c_uint32),
      ("inheritance", ctypes.c_uint32),
      ("offset", ctypes.c_ulonglong),
      ("user_tag", ctypes.c_uint32),
      ("ref_count", ctypes.c_uint32),
      ("shadow_depth", ctypes.c_uint16),
      ("external_pager", ctypes.c_byte),
      ("share_mode", ctypes.c_byte),
      ("is_submap", ctypes.c_uint32),
      ("behavior", ctypes.c_uint32),
      ("object_id", ctypes.c_uint32),
      ("user_wired_count", ctypes.c_uint32),
  ]
submap_info_size = ctypes.sizeof(vm_region_submap_short_info_data_64) // 4

address = ctypes.c_ulong(0)
mapsize = ctypes.c_ulong(0)
count = ctypes.c_uint32(submap_info_size)
sub_info = vm_region_submap_short_info_data_64()
depth = 0

c_depth = ctypes.c_uint32(depth)
for i in range(1):
  ret = libc.mach_vm_region_recurse(task,
    ctypes.pointer(address), ctypes.pointer(mapsize),
    ctypes.pointer(c_depth), ctypes.pointer(sub_info),
    ctypes.pointer(count))
  print("aslr", hex(ret), hex(address.value), mapsize, count, sub_info.protection)
  #address.value += mapsize.value
#exit(0)

patch_address = address.value + 0x8e38
patch = b"\x00\x00\x80\xd2"

pdata = ctypes.c_void_p(0)
data_cnt = ctypes.c_uint32(0)

ret = libc.mach_vm_read(task, ctypes.c_ulong(patch_address), 4, ctypes.pointer(pdata), ctypes.pointer(data_cnt))
buf = ctypes.string_at(pdata.value, data_cnt.value)
hexdump(buf)

#ret = libc.mach_vm_wire(mytask, task, patch_address, 4, 3)
#print(ret)
#exit(0)

"""
ret = libc.mach_vm_read(task, address, mapsize, ctypes.pointer(pdata), ctypes.pointer(data_cnt))
buf = ctypes.string_at(pdata.value, data_cnt.value)
hexdump(buf)

ret = libc.mach_vm_deallocate(task, address, mapsize)
print("mach_vm_deallocate", ret)

ret = libc.mach_vm_allocate(task, ctypes.pointer(address), mapsize, 0)
print("mach_vm_allocate", ret)
"""

ret = libc.mach_vm_protect(task, ctypes.c_ulong(patch_address), 4, True, 3)
print("protect", ret)

longptr = ctypes.POINTER(ctypes.c_ulong)
#shellcodePtr = ctypes.cast(buf, longptr)
#ret = libc.mach_vm_write(task, address, shellcodePtr, len(buf))
#print("write", ret)

shellcodePtr = ctypes.cast(patch, longptr)
ret = libc.mach_vm_write(task, ctypes.c_ulong(patch_address), shellcodePtr, len(buf))
print("write", ret)

#libc.mach_vm_write.argtypes = [ctypes.c_uint32, ctypes.c_ulong, longptr, ctypes.c_uint32]
#libc.mach_vm_write.restype = ctypes.c_uint32
#ret = libc.mach_vm_write(task, ctypes.c_ulong(patch_address), shellcodePtr, len(patch))

ret = libc.mach_vm_protect(task, ctypes.c_ulong(patch_address), 4, False, 5)
print("protect", ret)