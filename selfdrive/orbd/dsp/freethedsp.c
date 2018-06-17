// freethedsp by geohot
//   (because the DSP should be free)
// released under MIT License

// usage instructions:
//   1. Compile an example from the Qualcomm Hexagon SDK 
//   2. Try to run it on your phone
//   3. Be very sad when "adsprpc ... dlopen error: ... signature verify start failed for ..." appears in logcat
// ...here is where people would give up before freethedsp
//   4. Compile freethedsp with 'clang -shared freethedsp.c -o freethedsp.so' (or statically link it to your program)
//   5. Run your program with 'LD_PRELOAD=./freethedsp.so ./<your_prog>'
//   6. OMG THE DSP WORKS
//   7. Be happy.

// *** patch may have to change for your phone ***

// this is patching /dsp/fastrpc_shell_0
// correct if sha hash of fastrpc_shell_0 is "fbadc96848aefad99a95aa4edb560929dcdf78f8"
// patch to return 0xFFFFFFFF from is_test_enabled instead of 0
// your fastrpc_shell_0 may vary
#define PATCH_ADDR 0x5200c
#define PATCH_OLD "\x40\x3f\x20\x50"
#define PATCH_NEW "\x40\x3f\x00\x5a"
#define PATCH_LEN (sizeof(PATCH_OLD)-1)
#define _BITS_IOCTL_H_

// under 100 lines of code begins now
#include <stdio.h>
#include <dlfcn.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>

// ioctl stuff
#define IOC_OUT   0x40000000  /* copy out parameters */
#define IOC_IN    0x80000000  /* copy in parameters */
#define IOC_INOUT (IOC_IN|IOC_OUT)
#define IOCPARM_MASK  0x1fff    /* parameter length, at most 13 bits */

#define _IOC(inout,group,num,len) \
  (inout | ((len & IOCPARM_MASK) << 16) | ((group) << 8) | (num))
#define _IOWR(g,n,t)  _IOC(IOC_INOUT, (g), (n), sizeof(t))

// ion ioctls
#include <linux/ion.h>
#define ION_IOC_MSM_MAGIC 'M'
#define ION_IOC_CLEAN_INV_CACHES  _IOWR(ION_IOC_MSM_MAGIC, 2, \
            struct ion_flush_data)

struct ion_flush_data {
  ion_user_handle_t handle;
  int fd;
  void *vaddr;
  unsigned int offset;
  unsigned int length;
};

// fastrpc ioctls
#define FASTRPC_IOCTL_INIT       _IOWR('R', 6, struct fastrpc_ioctl_init)

struct fastrpc_ioctl_init {
  uint32_t flags;   /* one of FASTRPC_INIT_* macros */
  uintptr_t __user file;  /* pointer to elf file */
  int32_t filelen;  /* elf file length */
  int32_t filefd;   /* ION fd for the file */
  uintptr_t __user mem; /* mem for the PD */
  int32_t memlen;   /* mem length */
  int32_t memfd;    /* ION fd for the mem */
};

int ioctl(int fd, unsigned long request, void *arg) {
  static void *handle = NULL;
  static int (*orig_ioctl)(int, int, void*);

  if (handle == NULL) {
    handle = dlopen("/system/lib64/libc.so", RTLD_LAZY);
    assert(handle != NULL);
    orig_ioctl = dlsym(handle, "ioctl");
  }

  int ret = orig_ioctl(fd, request, arg);

  // carefully modify this one
  if (request == FASTRPC_IOCTL_INIT) {
    struct fastrpc_ioctl_init *init = (struct fastrpc_ioctl_init *)arg;

    // confirm patch is correct and do the patch
    assert(memcmp((void*)(init->mem+PATCH_ADDR), PATCH_OLD, PATCH_LEN) == 0);
    memcpy((void*)(init->mem+PATCH_ADDR), PATCH_NEW, PATCH_LEN);

    // flush cache
    int ionfd = open("/dev/ion", O_RDONLY);
    assert(ionfd > 0);

    struct ion_fd_data fd_data;
    fd_data.fd = init->memfd;
    int ret = ioctl(ionfd, ION_IOC_IMPORT, &fd_data);
    assert(ret == 0);

    struct ion_flush_data flush_data;
    flush_data.handle  = fd_data.handle;
    flush_data.vaddr   = (void*)init->mem;
    flush_data.offset  = 0;
    flush_data.length  = init->memlen;
    ret = ioctl(ionfd, ION_IOC_CLEAN_INV_CACHES, &flush_data);
    assert(ret == 0);

    struct ion_handle_data handle_data;
    handle_data.handle = fd_data.handle;
    ret = ioctl(ionfd, ION_IOC_FREE, &handle_data);
    assert(ret == 0);

    // cleanup
    close(ionfd);
  }

  return ret;
}

