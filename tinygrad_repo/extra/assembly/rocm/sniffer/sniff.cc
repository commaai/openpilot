// template copied from https://github.com/geohot/cuda_ioctl_sniffer/blob/master/sniff.cc

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <signal.h>
#include <ucontext.h>

#include <sys/mman.h>

// includes from the ROCm sources
#include <linux/kfd_ioctl.h>
#include <hsa.h>
#include <amd_hsa_kernel_code.h>
#include <ROCR-Runtime/src/core/inc/sdma_registers.h>
using namespace rocr::AMD;

#include <string>
#include <map>
std::map<int, std::string> files;
std::map<uint64_t, uint64_t> ring_base_addresses;

#define D(args...) fprintf(stderr, args)

uint64_t doorbell_offset = -1;
std::map<uint64_t, int> queue_types;

void hexdump(void *d, int l) {
  for (int i = 0; i < l; i++) {
    if (i%0x10 == 0 && i != 0) printf("\n");
    if (i%0x10 == 8) printf(" ");
    if (i%0x10 == 0) printf("%8X: ", i);
    printf("%2.2X ", ((uint8_t*)d)[i]);
  }
  printf("\n");
}

extern "C" {

// https://defuse.ca/online-x86-assembler.htm#disassembly2
static void handler(int sig, siginfo_t *si, void *unused) {
  ucontext_t *u = (ucontext_t *)unused;
  uint8_t *rip = (uint8_t*)u->uc_mcontext.gregs[REG_RIP];

  int store_size = 0;
  uint64_t value;
  if (rip[0] == 0x48 && rip[1] == 0x89 && rip[2] == 0x30) {
    // 0:  48 89 30                mov    QWORD PTR [rax],rsi
    store_size = 8;
    value = u->uc_mcontext.gregs[REG_RSI];
    u->uc_mcontext.gregs[REG_RIP] += 3;
  } else if (rip[0] == 0x4c && rip[1] == 0x89 && rip[2] == 0x28) {
    // 0:  4c 89 28                mov    QWORD PTR [rax],r13
    store_size = 8;
    value = u->uc_mcontext.gregs[REG_R13];
    u->uc_mcontext.gregs[REG_RIP] += 3;
  } else {
    D("segfault %02X %02X %02X %02X %02X %02X %02X %02X rip: %p addr: %p\n", rip[0], rip[1], rip[2], rip[3], rip[4], rip[5], rip[6], rip[7], rip, si->si_addr);
    D("rax: %llx rcx: %llx rdx: %llx rsi: %llx rbx: %llx\n", u->uc_mcontext.gregs[REG_RAX], u->uc_mcontext.gregs[REG_RCX], u->uc_mcontext.gregs[REG_RDX], u->uc_mcontext.gregs[REG_RSI], u->uc_mcontext.gregs[REG_RBX]);
    exit(-1);
  }

  uint64_t ring_base_address = ring_base_addresses[((uint64_t)si->si_addr)&0xFFF];
  int queue_type = queue_types[((uint64_t)si->si_addr)&0xFFF];
  D("%16p: \u001b[31mDING DONG\u001b[0m (queue_type %d) store(%d): 0x%8lx -> %p ring_base_address:0x%lx\n", rip, queue_type, store_size, value, si->si_addr, ring_base_address);

  if (queue_type == KFD_IOC_QUEUE_TYPE_SDMA) {
    uint8_t *sdma_ptr = (uint8_t*)(ring_base_address);
    while (sdma_ptr < ((uint8_t*)(ring_base_address)+value)) {
      D("0x%3lx: ", sdma_ptr-(uint8_t*)(ring_base_address));
      if (sdma_ptr[0] == SDMA_OP_TIMESTAMP) {
        D("SDMA_PKT_TIMESTAMP\n");
        sdma_ptr += sizeof(SDMA_PKT_TIMESTAMP);
      } else if (sdma_ptr[0] == SDMA_OP_GCR) {
        D("SDMA_PKT_GCR\n");
        sdma_ptr += sizeof(SDMA_PKT_GCR);
      } else if (sdma_ptr[0] == SDMA_OP_ATOMIC) {
        D("SDMA_PKT_ATOMIC\n");
        sdma_ptr += sizeof(SDMA_PKT_ATOMIC);
      } else if (sdma_ptr[0] == SDMA_OP_FENCE) {
        D("SDMA_PKT_FENCE\n");
        sdma_ptr += sizeof(SDMA_PKT_FENCE);
      } else if (sdma_ptr[0] == SDMA_OP_TRAP) {
        D("SDMA_PKT_TRAP\n");
        sdma_ptr += sizeof(SDMA_PKT_TRAP);
      } else if (sdma_ptr[0] == SDMA_OP_COPY && sdma_ptr[1] == SDMA_SUBOP_COPY_LINEAR) {
        SDMA_PKT_COPY_LINEAR *pkt = (SDMA_PKT_COPY_LINEAR *)sdma_ptr;
        D("SDMA_PKT_COPY_LINEAR: count:0x%x src:0x%lx dst:0x%lx\n", pkt->COUNT_UNION.count+1,
          (uint64_t)pkt->SRC_ADDR_LO_UNION.src_addr_31_0 | ((uint64_t)pkt->SRC_ADDR_HI_UNION.src_addr_63_32 << 32),
          (uint64_t)pkt->DST_ADDR_LO_UNION.dst_addr_31_0 | ((uint64_t)pkt->DST_ADDR_HI_UNION.dst_addr_63_32 << 32)
        );
        sdma_ptr += sizeof(SDMA_PKT_COPY_LINEAR);
      } else {
        D("unhandled packet type %d %d, exiting\n", sdma_ptr[0], sdma_ptr[1]);
        break;
      }
    }

    //hexdump((void*)(ring_base_address), 0x100);
  } else if (queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL) {
    hsa_kernel_dispatch_packet_t *pkt = (hsa_kernel_dispatch_packet_t *)(ring_base_address+value*0x40);
    if ((pkt->header&0xFF) == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      D("HSA_PACKET_TYPE_KERNEL_DISPATCH -- setup:%d workgroup[%d, %d, %d] grid[%d, %d, %d] kernel_object:0x%lx kernarg_address:%p\n", pkt->setup, pkt->workgroup_size_x, pkt->workgroup_size_y, pkt->workgroup_size_z, pkt->grid_size_x, pkt->grid_size_y, pkt->grid_size_z, pkt->kernel_object, pkt->kernarg_address);
      amd_kernel_code_t *code = (amd_kernel_code_t *)pkt->kernel_object;
      D("kernel_code_entry_byte_offset:%lx\n", code->kernel_code_entry_byte_offset);
      uint32_t *kernel_code = (uint32_t*)(pkt->kernel_object + code->kernel_code_entry_byte_offset);
      int code_len = 0;
      while (kernel_code[code_len] != 0xbf9f0000 && kernel_code[code_len] != 0) code_len++;
      hexdump(kernel_code, code_len*4);
      /*FILE *f = fopen("/tmp/kernel_code", "wb");
      fwrite(kernel_code, 4, code_len, f);
      fclose(f);
      system("python -c 'print(\" \".join([(\"0x%02X\"%x) for x in open(\"/tmp/kernel_code\", \"rb\").read()]))' | ../build/llvm-project/bin/llvm-mc --disassemble --arch=amdgcn --mcpu=gfx1100 --show-encoding");*/
      D("kernargs (kernarg_segment_byte_size:0x%lx)\n", code->kernarg_segment_byte_size);
      // get length
      int i;
      for (i = 0; i < 0x400; i+=0x10) {
        if (memcmp((void*)((uint64_t)pkt->kernarg_address+i), "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", 0x10) == 0) break;
      }
      hexdump((void*)pkt->kernarg_address, i+0x10);
    } else if ((pkt->header&0xFF) == HSA_PACKET_TYPE_BARRIER_AND) {
      hsa_barrier_and_packet_t *pkt_and = (hsa_barrier_and_packet_t *)(ring_base_address+value*0x40);
      D("HSA_PACKET_TYPE_BARRIER_AND completion_signal:0x%lx\n", pkt_and->completion_signal.handle);
      //hexdump((void*)(ring_base_address+value*0x40), 0x40);
    } else if ((pkt->header&0xFF) == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
      D("HSA_PACKET_TYPE_VENDOR_SPECIFIC\n");
      hexdump((void*)(ring_base_address+value*0x40), 0x40);
    } else {
      hexdump((void*)(ring_base_address+value*0x40), 0x40);
    }
  }

  mprotect((void *)((uint64_t)si->si_addr & ~0xFFF), 0x2000, PROT_READ | PROT_WRITE);
  if (store_size == 8) {
    *(volatile uint64_t*)(si->si_addr) = value;
  } else if (store_size == 4) {
    *(volatile uint32_t*)(si->si_addr) = value;
  } else if (store_size == 2) {
    *(volatile uint16_t*)(si->si_addr) = value;
  } else {
    D("store size not supported\n");
    exit(-1);
  }
  mprotect((void *)((uint64_t)si->si_addr & ~0xFFF), 0x2000, PROT_NONE);
}

void register_sigsegv_handler() {
  struct sigaction sa = {0};
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = handler;
  if (sigaction(SIGSEGV, &sa, NULL) == -1) {
    D("ERROR: failed to register sigsegv handler");
    exit(-1);
  }
  // NOTE: python (or ocl runtime?) blocks the SIGSEGV signal
  sigset_t x;
  sigemptyset(&x);
  sigaddset(&x, SIGSEGV);
  sigprocmask(SIG_UNBLOCK, &x, NULL);
}

int (*my_open)(const char *pathname, int flags, mode_t mode);
#undef open
int open(const char *pathname, int flags, mode_t mode) {
  if (my_open == NULL) my_open = reinterpret_cast<decltype(my_open)>(dlsym(RTLD_NEXT, "open"));
  int ret = my_open(pathname, flags, mode);
  //D("open %s (0o%o) = %d\n", pathname, flags, ret);
  files[ret] = pathname;
  return ret;
}


int (*my_open64)(const char *pathname, int flags, mode_t mode);
#undef open
int open64(const char *pathname, int flags, mode_t mode) {
  if (my_open64 == NULL) my_open64 = reinterpret_cast<decltype(my_open64)>(dlsym(RTLD_NEXT, "open64"));
  int ret = my_open64(pathname, flags, mode);
  //D("open %s (0o%o) = %d\n", pathname, flags, ret);
  files[ret] = pathname;
  return ret;
}

void *(*my_mmap)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
#undef mmap
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  if (my_mmap == NULL) my_mmap = reinterpret_cast<decltype(my_mmap)>(dlsym(RTLD_NEXT, "mmap"));
  void *ret = my_mmap(addr, length, prot, flags, fd, offset);

  if (doorbell_offset != -1 && offset == doorbell_offset) {
    D("HIDDEN DOORBELL %p, handled by %p\n", addr, handler);
    register_sigsegv_handler();
    mprotect(addr, length, PROT_NONE);
  }

  if (fd != -1) D("mmapped %p (target %p) with flags 0x%x length 0x%zx fd %d %s offset 0x%lx\n", ret, addr, flags, length, fd, files[fd].c_str(), offset);
  return ret;
}

void *(*my_mmap64)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
#undef mmap64
void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off_t offset) { return mmap(addr, length, prot, flags, fd, offset); }

int ioctl_num = 1;
int (*my_ioctl)(int filedes, unsigned long request, void *argp) = NULL;
#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  if (my_ioctl == NULL) my_ioctl = reinterpret_cast<decltype(my_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));
  int ret = 0;
  ret = my_ioctl(filedes, request, argp);
  if (!files.count(filedes)) return ret;

  uint8_t type = (request >> 8) & 0xFF;
  uint8_t nr = (request >> 0) & 0xFF;
  uint16_t size = (request >> 16) & 0xFFF;

  D("%3d: %d = %3d(%20s) 0x%3x ", ioctl_num, ret, filedes, files[filedes].c_str(), size);

  if (request == AMDKFD_IOC_SET_EVENT) {
    kfd_ioctl_set_event_args *args = (kfd_ioctl_set_event_args *)argp;
    D("AMDKFD_IOC_SET_EVENT event_id:%d", args->event_id);
  } else if (request == AMDKFD_IOC_ALLOC_MEMORY_OF_GPU) {
    kfd_ioctl_alloc_memory_of_gpu_args *args = (kfd_ioctl_alloc_memory_of_gpu_args *)argp;
    D("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU va_addr:0x%llx size:0x%llx handle:%llX gpu_id:0x%x", args->va_addr, args->size, args->handle, args->gpu_id);
  } else if (request == AMDKFD_IOC_MAP_MEMORY_TO_GPU) {
    kfd_ioctl_map_memory_to_gpu_args *args = (kfd_ioctl_map_memory_to_gpu_args *)argp;
    D("AMDKFD_IOC_MAP_MEMORY_TO_GPU handle:%llX", args->handle);
  } else if (request == AMDKFD_IOC_CREATE_EVENT) {
    kfd_ioctl_create_event_args *args = (kfd_ioctl_create_event_args *)argp;
    D("AMDKFD_IOC_CREATE_EVENT event_page_offset:0x%llx event_type:%d event_id:%d", args->event_page_offset, args->event_type, args->event_id);
  } else if (request == AMDKFD_IOC_WAIT_EVENTS) {
    D("AMDKFD_IOC_WAIT_EVENTS");
  } else if (request == AMDKFD_IOC_SET_XNACK_MODE) {
    D("AMDKFD_IOC_SET_XNACK_MODE");
  } else if (request == AMDKFD_IOC_SVM || (type == 0x4b && nr == 0x20)) {
    // NOTE: this one is variable length
    kfd_ioctl_svm_args *args = (kfd_ioctl_svm_args *)argp;
    D("AMDKFD_IOC_SVM start_addr:0x%llx size:0x%llx op:%d", args->start_addr, args->size, args->op);
  } else if (request == AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU) {
    kfd_ioctl_unmap_memory_from_gpu_args *args = (kfd_ioctl_unmap_memory_from_gpu_args *)argp;
    D("AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU handle:%llX", args->handle);
  } else if (request == AMDKFD_IOC_FREE_MEMORY_OF_GPU) {
    D("AMDKFD_IOC_FREE_MEMORY_OF_GPU");
  } else if (request == AMDKFD_IOC_SET_SCRATCH_BACKING_VA) {
    D("AMDKFD_IOC_SET_SCRATCH_BACKING_VA");
  } else if (request == AMDKFD_IOC_GET_TILE_CONFIG) {
    D("AMDKFD_IOC_GET_TILE_CONFIG");
  } else if (request == AMDKFD_IOC_SET_TRAP_HANDLER) {
    D("AMDKFD_IOC_SET_TRAP_HANDLER");
  } else if (request == AMDKFD_IOC_GET_VERSION) {
    kfd_ioctl_get_version_args *args = (kfd_ioctl_get_version_args *)argp;
    D("AMDKFD_IOC_GET_VERSION major_version:%d minor_version:%d", args->major_version, args->minor_version);
  } else if (request == AMDKFD_IOC_GET_PROCESS_APERTURES_NEW) {
    D("AMDKFD_IOC_GET_PROCESS_APERTURES_NEW");
  } else if (request == AMDKFD_IOC_ACQUIRE_VM) {
    D("AMDKFD_IOC_ACQUIRE_VM");
  } else if (request == AMDKFD_IOC_SET_MEMORY_POLICY) {
    D("AMDKFD_IOC_SET_MEMORY_POLICY");
  } else if (request == AMDKFD_IOC_GET_CLOCK_COUNTERS) {
    D("AMDKFD_IOC_GET_CLOCK_COUNTERS");
  } else if (request == AMDKFD_IOC_CREATE_QUEUE) {
    kfd_ioctl_create_queue_args *args = (kfd_ioctl_create_queue_args *)argp;
    D("AMDKFD_IOC_CREATE_QUEUE\n");
    D("queue_type:%d ring_base_address:0x%llx\n", args->queue_type, args->ring_base_address);
    D("eop_buffer_address:0x%llx ctx_save_restore_address:0x%llx\n", args->eop_buffer_address, args->ctx_save_restore_address);
    D("ring_size:0x%x queue_priority:%d\n", args->ring_size, args->queue_priority);
    D("RETURNS write_pointer_address:0x%llx read_pointer_address:0x%llx doorbell_offset:0x%llx queue_id:%d\n", args->write_pointer_address, args->read_pointer_address, args->doorbell_offset, args->queue_id);
    //D("RETURNS *write_pointer_address:0x%llx *read_pointer_address:0x%llx\n", *(uint64_t*)args->write_pointer_address, *(uint64_t*)args->read_pointer_address);
    ring_base_addresses[args->doorbell_offset&0xFFF] = args->ring_base_address;
    queue_types[args->doorbell_offset&0xFFF] = args->queue_type;
    doorbell_offset = args->doorbell_offset&~0xFFF;
  } else {
    D("type:0x%x nr:0x%x size:0x%x", type, nr, size);
  }

  D("\n");
  ioctl_num++;
  return ret;
}

}
