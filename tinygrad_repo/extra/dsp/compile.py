#!/usr/bin/env python3
import os, ctypes, time, fcntl, mmap
import llvmlite.binding as llvm
from tinygrad.helpers import getenv, to_mv
from tinygrad.runtime.support.elf import elf_loader
from hexdump import hexdump
from tinygrad.runtime.autogen import libc
if getenv("IOCTL"): import run # noqa: F401 # pylint: disable=unused-import

adsp = ctypes.CDLL(ctypes.util.find_library("adsprpc"))
import adsprpc
import ion
import msm_ion
ION_IOC_ALLOC = 0
ION_IOC_MAP = 2
ION_IOC_SHARE = 4
ION_IOC_CUSTOM = 6
ION_ADSP_HEAP_ID = 22
ION_IOMMU_HEAP_ID = 25

def ion_iowr(fd, nr, args):
  ret = fcntl.ioctl(fd, (3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord(ion.ION_IOC_MAGIC) & 0xFF) << 8 | (nr & 0xFF), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

if __name__ == "__main__":
  # TODO: mmap tensors to the DSP
  # call the target function with the mmaped tensors
  ion_fd = os.open("/dev/ion", os.O_RDWR | os.O_CLOEXEC)
  arg3 = ion.struct_ion_allocation_data(len=0x1000, align=0x1000, heap_id_mask=1<<msm_ion.ION_SYSTEM_HEAP_ID, flags=ion.ION_FLAG_CACHED)
  ion_iowr(ion_fd, ION_IOC_ALLOC, arg3)
  print(arg3.handle)
  arg2 = ion.struct_ion_fd_data(handle=arg3.handle)
  ion_iowr(ion_fd, ION_IOC_SHARE, arg2)
  print(arg2.fd)

  res = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, arg2.fd, 0)
  print("mmapped", hex(res))
  to_mv(res, 0x10)[1] = 0xaa

  from tinygrad.runtime.ops_dsp import ClangCompiler
  cc = ClangCompiler(args=["--target=hexagon", "-mcpu=hexagonv65", "-fuse-ld=lld", "-nostdlib"])

  obj = cc.compile("""
    typedef unsigned long long remote_handle64;
    typedef struct { void *pv; unsigned int len; } remote_buf;
    typedef struct { int fd; unsigned int offset; } remote_dma_handle;
    typedef union { remote_buf buf; remote_handle64 h64; remote_dma_handle dma; } remote_arg;
    void* HAP_mmap(void *addr, int len, int prot, int flags, int fd, long offset);
    int HAP_munmap(void *addr, int len);
    #define HAP_MEM_CACHE_WRITETHROUGH 0x40

    int entry(unsigned long long handle, unsigned int sc, remote_arg* pra) {
      if (sc>>24 == 1) {
        //void *mmaped = *((void**)pra[0].buf.pv);
        void *a = HAP_mmap(0, 0x1000, 3, 0, pra[1].dma.fd, 0);
        ((char*)a)[0] = 0x55;
        ((char*)a)[4] = 0x55;
        ((char*)a)[8] = 0x99;
        //((char*)a)[1] = 0x9b;
        //char ret = ((char*)a)[1];
        HAP_munmap(a, 0x1000);
        return 0;

        //return ((int)mmaped)&0xFFFF;
        //return ((char*)mmaped)[1];
        //return sizeof(void*);

        //((char*)mmaped)[0] = 55;
        //return ((int)mmaped)&0xFFFF;

        //void addr = *((void**)pra[1])
        //return sizeof(remote_buf);
        //((char*)pra[1].h64)[0] = 55;

        //return ((char*)mmaped)[1];
        //((char*)mmaped)[0] = 55;
        // NOTE: you have to return 0 for outbufs to work
        //return ((int)pra[1].h64)&0xFFFF;
      }
      return 0;
    }
  """)
  with open("/tmp/swag.so", "wb") as f: f.write(obj)

  handle = ctypes.c_int64(-1)
  adsp.remote_handle64_open(ctypes.create_string_buffer(b"file:////tmp/swag.so?entry&_modver=1.0&_dom=cdsp"), ctypes.byref(handle))
  print("HANDLE", handle.value)
  #print(adsp.remote_handle64_invoke(handle, 0, None))

  #rem = adsp.remote_register_buf(res, 0x1000, arg2.fd, 4)
  #rem = adsp.remote_register_dma_handle(arg2.fd, 0x1000)
  #print("remote_register_buf_attr", rem)

  #out = ctypes.c_uint64(0)
  #ret = adsp.remote_mmap(arg2.fd, 0, 0, 0x1000, ctypes.byref(out))
  #print(ret)
  #print("mapped at", hex(out.value))

  #arg_2 = ctypes.c_int64(out.value)
  arg_2 = ctypes.c_int64(arg2.fd)
  pra = (adsprpc.union_remote_arg64 * 3)()
  pra[0].buf.pv = ctypes.addressof(arg_2)
  pra[0].buf.len = 8
  pra[1].dma.fd = arg2.fd
  pra[1].dma.len = 0x1000
  print("invoke")
  ret = adsp.remote_handle64_invoke(handle, (1<<24) | (1<<16) | (1 << 4), pra)
  print("return value", ret, hex(ret))
  #print(hex(arg_2.value), arg_2.value)
  #time.sleep(0.1)

  # flush the cache
  """
  flush_data = msm_ion.struct_ion_flush_data(handle=arg3.handle, vaddr=res, offset=0, length=0x1000)
  # ION_IOC_CLEAN_INV_CACHES
  cd = ion.struct_ion_custom_data(
    cmd=(3 << 30) | (ctypes.sizeof(flush_data) & 0x1FFF) << 16 | (ord(msm_ion.ION_IOC_MSM_MAGIC) & 0xFF) << 8 | (2 & 0xFF),
    arg=ctypes.addressof(flush_data))
  ret = ion_iowr(ion_fd, ION_IOC_CUSTOM, cd)

  res2 = libc.mmap(0, 0x1000, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED, arg2.fd, 0)
  """
  hexdump(to_mv(res, 0x10))
  os._exit(0)

