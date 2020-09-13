#include <sys/types.h>
#include "include/msm_kgsl.h"
#include "common/clutil.h"
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cassert>
#include <sys/mman.h>

int run_num = 0;
int ioctl_num = 0;

void hexdump(uint32_t *d, int len) {
  assert((len%4) == 0);
  printf("  dumping %p len 0x%x\n", d, len);
  for (int i = 0; i < len/4; i++) {
    if (i != 0 && (i%0x10) == 0) printf("\n");
    printf("%8x ", d[i]);
  }
  printf("\n");
}

void hexdump8(uint8_t *d, int len) {
  printf("  dumping %p len 0x%x\n", d, len);
  for (int i = 0; i < len; i++) {
    if (i != 0 && (i%0x10) == 0) printf("\n");
    printf("%02x ", d[i]);
  }
  printf("\n");
}


#include <string>
#include <vector>
#include <map>
using namespace std;

#include "disasm/include/adreno_pm4types.h"

#define REG_A5XX_TPL1_CS_TEX_CONST_LO        0x0000e760
#define REG_A5XX_TPL1_CS_TEX_SAMP_LO         0x0000e75c

class CachedCommand {
  public:
    CachedCommand(struct kgsl_gpu_command *cmd, int lfd);
    void exec(bool wait);
  private:
    string cmd_0, cmd_1;
    int obj_len;
    int fd;

    struct kgsl_gpu_command cache;
    struct kgsl_command_object cmds[2];
    struct kgsl_command_object objs[1];
};

vector<CachedCommand *> queue_cmds;

void disassemble(uint32_t *src, int len) {
  int i = 0;
  while (i < len) {
		int pktsize;
    int pkttype = -1;

		if (pkt_is_type0(src[i])) {
      pkttype = 0;
			pktsize = type0_pkt_size(src[i]);
		} else if (pkt_is_type3(src[i])) {
      pkttype = 3;
			pktsize = type3_pkt_size(src[i]);
		} else if (pkt_is_type4(src[i])) {
      pkttype = 4;
      pktsize = type4_pkt_size(src[i]);
    } else if (pkt_is_type7(src[i])) {
      pkttype = 7;
      pktsize = type7_pkt_size(src[i]);
    }
    printf("%3d: type:%d size:%d ", i, pkttype, pktsize);

    if (pkttype == 7) {
      printf("op:  %4x ", cp_type7_opcode(src[i]));
    }

    if (pkttype == 4) {
      printf("reg: %4x ", cp_type4_base_index_one_reg_wr(src[i]));
    }

    for (int j = 0; j < pktsize+1; j++) {
      printf("%8.8X ", src[i+j]);
    }
    printf("\n");

    if (pkttype == 7 && cp_type7_opcode(src[i]) == CP_LOAD_STATE) {
      // CP_LOAD_STATE4
      int sz = (src[i+1] & 0xffc00000) >> 22;
      uint64_t addr = (uint64_t)(src[i+2] & 0xfffffffc) | ((uint64_t)(src[i+3]) << 32);
      hexdump((uint32_t *)addr, sz*4);
    }

    if (pkttype == 4 && cp_type4_base_index_one_reg_wr(src[i]) == REG_A5XX_TPL1_CS_TEX_CONST_LO) {
      uint64_t addr = (uint64_t)(src[i+1] & 0xffffffff) | ((uint64_t)(src[i+2]) << 32);
      hexdump((uint32_t *)addr, 0x40);
    }

    if (pkttype == 4 && cp_type4_base_index_one_reg_wr(src[i]) == REG_A5XX_TPL1_CS_TEX_SAMP_LO) {
      uint64_t addr = (uint64_t)(src[i+1] & 0xffffffff) | ((uint64_t)(src[i+2]) << 32);
      hexdump((uint32_t *)addr, 0x40);
    }

    if (pkttype == -1) break;
    i += (1+pktsize);
  }
  assert(i == len);

}

int intercept = 1;
int prop_num = 0;

extern "C" {

/*void *gsl_memory_alloc_pure(long param_1, long param_2, long *param_3) {
  void *(*my_gsl_memory_alloc_pure)(long param_1, long param_2, long *param_3);
  my_gsl_memory_alloc_pure = reinterpret_cast<decltype(my_gsl_memory_alloc_pure)>(dlsym(RTLD_NEXT, "gsl_memory_alloc_pure"));

  void *ret = my_gsl_memory_alloc_pure(param_1, param_2, param_3);
  printf("gsl_memory_alloc_pure: 0x%lx 0x%lx %p = %p\n", param_1, param_2, param_3, ret);
  return ret;
}*/

void *mmap64(void *addr, size_t len, int prot, int flags, int fildes, off64_t off) {
  void *(*my_mmap64)(void *addr, size_t len, int prot, int flags, int fildes, off64_t off);
  my_mmap64 = reinterpret_cast<decltype(my_mmap64)>(dlsym(RTLD_NEXT, "mmap64"));

  void *ret = my_mmap64(addr, len, prot, flags, fildes, off);

  if (fildes == 3) {
    printf("mmap64(addr=%p, len=0x%zx, prot=0x%x, flags=0x%x, fildes=%d, off=0x%lx) = %p\n", addr, len, prot, flags, fildes, off, ret);
  }

  return ret;
}


pid_t gettid(void);

#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  int (*my_ioctl)(int filedes, unsigned long request, void *argp);
  my_ioctl = reinterpret_cast<decltype(my_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));
  int skip = 0;

if (intercept) {

  int tid = gettid();

  if (request == IOCTL_KGSL_GPU_COMMAND) {
    struct kgsl_gpu_command *cmd = (struct kgsl_gpu_command *)argp;
    printf("IOCTL_KGSL_GPU_COMMAND(%d): flags: 0x%lx numcmds: %u   numobjs: %u  numsyncs: %u   context_id: %u  timestamp: %u\n",
        tid,
        cmd->flags,
        cmd->numcmds, cmd->numobjs, cmd->numsyncs,
        cmd->context_id, cmd->timestamp);

    assert(cmd->numcmds == 2);
    assert(cmd->numobjs == 1);
    assert(cmd->numsyncs == 0);

    //struct kgsl_command_object *obj = (struct kgsl_command_object *)cmd->cmdlist;
    //assert(obj[0].size == sizeof(queue_init));
    //memcpy(queue_init, (void*)obj[0].gpuaddr, sizeof(queue_init));
    //string qcmd((char*)obj[1].gpuaddr, obj[1].size);
    if (run_num == 3) {
      CachedCommand *ccmd = new CachedCommand(cmd, filedes);
      queue_cmds.push_back(ccmd);

      //ccmd->exec();

      //skip = 0;
      //printf("command 0x%lx\n", obj[1].gpuaddr);
      //disassemble((uint32_t *)qcmd.data(), qcmd.size()/4);
      //queue_cmds.push_back(qcmd);
    }

    #ifdef DUMP
      char tmp[0x100];
      snprintf(tmp, sizeof(tmp), "/tmp/thneed/run_%d_%d", run_num, ioctl_num++);
      FILE *f = fopen(tmp, "wb");
    #endif

    // kgsl_cmdbatch_add_cmdlist
    for (int i = 0; i < cmd->numcmds; i++) {
      struct kgsl_command_object *obj = (struct kgsl_command_object *)cmd->cmdlist;
      printf("  cmd: %lx %5lx %5lx flags:%3x %d\n",
          obj[i].offset, obj[i].gpuaddr, obj[i].size, obj[i].flags, obj[i].id);
      //hexdump((uint32_t *)obj[i].gpuaddr, obj[i].size);
      #ifdef DUMP
        fwrite(&obj[i].size, sizeof(obj[i].size), 1, f);
        fwrite((void*)obj[i].gpuaddr, obj[i].size, 1, f);
      #endif
    }

    // kgsl_cmdbatch_add_memlist
    for (int i = 0; i < cmd->numobjs; i++) {
      struct kgsl_command_object *obj = (struct kgsl_command_object *)cmd->objlist;
      printf("  obj: %lx %5lx %5lx flags:%3x %d\n",
          obj[i].offset, obj[i].gpuaddr, obj[i].size, obj[i].flags, obj[i].id);
      //hexdump((uint32_t *)obj[i].gpuaddr, obj[i].size);

      #ifdef DUMP
        fwrite(&obj[i].size, sizeof(obj[i].size), 1, f);
        fwrite((void*)obj[i].gpuaddr, obj[i].size, 1, f);
      #endif
    }

    #ifdef DUMP
      fclose(f);
    #endif

  } else if (request == IOCTL_KGSL_SETPROPERTY) {
    struct kgsl_device_getproperty *prop = (struct kgsl_device_getproperty *)argp;
    printf("IOCTL_KGSL_SETPROPERTY(%d): 0x%x\n", tid, prop->type);
    hexdump8((uint8_t*)prop->value, prop->sizebytes);
    if (prop_num == 1) { printf("SKIPPING\n"); skip = 1; }
    if (run_num == 3) prop_num++;
    //hexdump((unsigned char*)prop->value, prop->sizebytes);
  } else if (request == IOCTL_KGSL_GPUOBJ_SYNC) {
    struct kgsl_gpuobj_sync *cmd = (struct kgsl_gpuobj_sync *)argp;
    struct kgsl_gpuobj_sync_obj *objs = (struct kgsl_gpuobj_sync_obj *)(cmd->objs);

    printf("IOCTL_KGSL_GPUOBJ_SYNC(%d) count:%d ", tid, cmd->count);
    for (int i = 0; i < cmd->count; i++) {
      printf(" -- offset:0x%lx len:0x%lx id:%d op:%d  ", objs[i].offset, objs[i].length, objs[i].id, objs[i].op);
    }
    printf("\n");
  } else if (request == IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID) {
    struct kgsl_device_waittimestamp_ctxtid *cmd = (struct kgsl_device_waittimestamp_ctxtid *)argp;
    printf("IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(%d): context_id: %d  timestamp: %d  timeout: %d\n",
        tid, cmd->context_id, cmd->timestamp, cmd->timeout);
  } else if (request == IOCTL_KGSL_GPUOBJ_ALLOC) {
    struct kgsl_gpuobj_alloc *cmd = (struct kgsl_gpuobj_alloc *)argp;
    printf("IOCTL_KGSL_GPUOBJ_ALLOC: size:0x%lx flags:0x%lx va_len:0x%lx  ", cmd->size, cmd->flags, cmd->va_len);
  } else if (request == IOCTL_KGSL_GPUOBJ_FREE) {
    //printf("IOCTL_KGSL_GPUOBJ_FREE\n");
  } else if (filedes == 3) {
    printf("ioctl(%d) %lx\n", tid, request);
  }

}

  int ret;
  if (skip) {
    ret = 0;
  } else {
    ret = my_ioctl(filedes, request, argp);
  }

  if (request == IOCTL_KGSL_GPUOBJ_ALLOC) {
    struct kgsl_gpuobj_alloc *cmd = (struct kgsl_gpuobj_alloc *)argp;
    printf("mmapsize:0x%lx id:%d metadata_len:%x metadata:0x%lx = %d\n", cmd->mmapsize, cmd->id, cmd->metadata_len, cmd->metadata, ret);
  }

  return ret;
}

}

#include <CL/cl.h>
#include "../runners/snpemodel.h"
#include <sys/types.h>
#include <time.h>

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

int global_timestamp = -1;
CachedCommand::CachedCommand(struct kgsl_gpu_command *cmd, int lfd) {
  fd = lfd;
  assert(cmd->numcmds == 2);
  assert(cmd->numobjs == 1);
  assert(cmd->numsyncs == 0);

  global_timestamp = cmd->timestamp;

  printf("%p  %p %p\n", cmd, (void*)cmd->cmdlist, (void*)cmd->objlist);

  memcpy(cmds, (void *)cmd->cmdlist, sizeof(struct kgsl_command_object)*2);
  memcpy(objs, (void *)cmd->objlist, sizeof(struct kgsl_command_object)*1);
  cmd_0.assign((char*)cmds[0].gpuaddr, cmds[0].size);
  cmd_1.assign((char*)cmds[1].gpuaddr, cmds[1].size);


  memcpy(&cache, cmd, sizeof(cache));
}

// i think you get these with cl_a5x_ringbuffer_alloc
uint64_t base = 0;

void CachedCommand::exec(bool wait) {
  printf("old addr 0x%lx ", cmds[1].gpuaddr);
  cmds[1].gpuaddr = base;
  printf("using addr 0x%lx with size 0x%4lx ", cmds[1].gpuaddr, cmd_1.size());
  base += (cmd_1.size()+0xff) & (~0xFF);
  memcpy((void*)cmds[1].gpuaddr, cmd_1.data(), cmd_1.size());

  // set up other buffers
  memcpy((void*)cmds[0].gpuaddr, cmd_0.data(), cmd_0.size());
  memset((void*)objs[0].gpuaddr, 0, objs[0].size);

  cache.timestamp = ++global_timestamp;
  cache.cmdlist = (uint64_t)cmds;
  cache.objlist = (uint64_t)objs;

  // run
  int ret = ioctl(fd, IOCTL_KGSL_GPU_COMMAND, &cache);

  if (wait) {
    struct kgsl_device_waittimestamp_ctxtid wait;
    wait.context_id = cache.context_id;
    wait.timestamp = cache.timestamp;
    wait.timeout = -1;

    uint64_t tb = nanos_since_boot();
    int wret = ioctl(fd, IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, &wait);
    uint64_t te = nanos_since_boot();

    printf("exec %d wait %d after %lu us\n", ret, wret, (te-tb)/1000);
  } else {
    printf("CachedCommand::exec got %d\n", ret);
  }
}


int do_print = 0;

#define TEMPORAL_SIZE 512
#define DESIRE_LEN 8
#define TRAFFIC_CONVENTION_LEN 2

FILE *f = NULL;

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) {
  cl_program (*my_clCreateProgramWithSource)(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) = NULL;
  my_clCreateProgramWithSource = reinterpret_cast<decltype(my_clCreateProgramWithSource)>(dlsym(RTLD_NEXT, "REAL_clCreateProgramWithSource"));
  //printf("clCreateProgramWithSource: %d\n", count);

  if (f == NULL) {
    f = fopen("/tmp/kernels.cl", "w");
  }

  fprintf(f, "/* ************************ PROGRAM BREAK ****************************/\n");
  for (int i = 0; i < count; i++) {
    fprintf(f, "%s\n", strings[i]);
    if (i != 0) fprintf(f, "/* ************************ SECTION BREAK ****************************/\n");
  }
  fflush(f);

  return my_clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
}

map<cl_kernel, string> kernels;
map<cl_kernel, cl_mem> kernel_inputs;
map<cl_kernel, cl_mem> kernel_outputs;

cl_kernel clCreateKernel(cl_program program, const char *kernel_name, cl_int *errcode_ret) {
  cl_kernel (*my_clCreateKernel)(cl_program program, const char *kernel_name, cl_int *errcode_ret) = NULL;
  my_clCreateKernel = reinterpret_cast<decltype(my_clCreateKernel)>(dlsym(RTLD_NEXT, "REAL_clCreateKernel"));
  cl_kernel ret = my_clCreateKernel(program, kernel_name, errcode_ret);

  printf("clCreateKernel: %s -> %p\n", kernel_name, ret);
  kernels.insert(make_pair(ret, kernel_name));
  return ret;
}

typedef struct image {
  size_t image_width;
  size_t image_height;
  size_t image_row_pitch;
  cl_mem buffer;
} image;

map<cl_mem, size_t> buffers;
map<cl_mem, image> images;

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  cl_int (*my_clSetKernelArg)(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) = NULL;
  my_clSetKernelArg = reinterpret_cast<decltype(my_clSetKernelArg)>(dlsym(RTLD_NEXT, "REAL_clSetKernelArg"));

  char arg_type[0x100];
  char arg_name[0x100];
  clGetKernelArgInfo(kernel, arg_index, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_type), arg_type, NULL);
  clGetKernelArgInfo(kernel, arg_index, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
  printf("  %s %s", arg_type, arg_name);

  if (arg_size == 1) {
    printf(" = %d", *((char*)arg_value));
  } else if (arg_size == 2) {
    printf(" = %d", *((short*)arg_value));
  } else if (arg_size == 4) {
    if (strcmp(arg_type, "float") == 0) {
      printf(" = %f", *((float*)arg_value));
    } else {
      printf(" = %d", *((int*)arg_value));
    }
  } else if (arg_size == 8) {
    cl_mem val = (cl_mem)(*((uintptr_t*)arg_value));
    printf(" = %p", val);
    if (strcmp(arg_name, "input") == 0) kernel_inputs[kernel] = val;
    if (strcmp(arg_name, "output") == 0) kernel_outputs[kernel] = val;
    if (strcmp(arg_name, "accumulator") == 0) assert(kernel_inputs[kernel] = val);

    if (buffers.find(val) != buffers.end()) {
      printf(" buffer %zu", buffers[val]);
    }

    if (images.find(val) != images.end()) {
      printf(" image %zu x %zu rp %zu @ %p", images[val].image_width, images[val].image_height, images[val].image_row_pitch, images[val].buffer);
    }

  } else {
    printf(" %zu", arg_size);
  }
  printf("\n");
  cl_int ret = my_clSetKernelArg(kernel, arg_index, arg_size, arg_value);
  return ret;
}

uint64_t start_time = 0;
uint64_t tns = 0;

int cnt = 0;

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event) {

  // SNPE doesn't use these
  assert(num_events_in_wait_list == 0);
  assert(global_work_offset == NULL);

  cl_int (*my_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *) = NULL;
  my_clEnqueueNDRangeKernel = reinterpret_cast<decltype(my_clEnqueueNDRangeKernel)>(dlsym(RTLD_NEXT, "REAL_clEnqueueNDRangeKernel"));


  uint64_t tb = nanos_since_boot();
  cl_int ret = my_clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
    global_work_offset, global_work_size, local_work_size,
    num_events_in_wait_list, event_wait_list, event);
  uint64_t te = nanos_since_boot();

  /*ret = clWaitForEvents(1, event);
  assert(ret == CL_SUCCESS);
  uint64_t tq = nanos_since_boot();*/

  if (do_print) {
    tns += te-tb;
  }

  printf("%10lu %10lu running(%3d) -- %p -- %56s -- %p -> %p %s ", (tb-start_time)/1000, (tns/1000), cnt++, kernel, kernels[kernel].c_str(), kernel_inputs[kernel], kernel_outputs[kernel],
    (buffers[kernel_outputs[kernel]] != 0) ? "B" : "I");

  printf("global -- ");
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", global_work_size[i]);
  }
  printf("local -- ");
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", local_work_size[i]);
  }
  printf("\n");

  return ret;
}


cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) {
  cl_mem (*my_clCreateBuffer)(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr, cl_int *errcode_ret) = NULL;
  my_clCreateBuffer = reinterpret_cast<decltype(my_clCreateBuffer)>(dlsym(RTLD_NEXT, "REAL_clCreateBuffer"));

  cl_mem ret = my_clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
  buffers[ret] = size;
  printf("%p = clCreateBuffer %zu\n", ret, size);
  return ret;
}

cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) {
  cl_mem (*my_clCreateImage)(cl_context context, cl_mem_flags flags, const cl_image_format *image_format, const cl_image_desc *image_desc, void *host_ptr, cl_int *errcode_ret) = NULL;
  my_clCreateImage = reinterpret_cast<decltype(my_clCreateImage)>(dlsym(RTLD_NEXT, "REAL_clCreateImage"));

  // SNPE only uses this
  assert(CL_MEM_OBJECT_IMAGE2D == image_desc->image_type);

  // RGBA, HALF FLOAT
  assert(CL_RGBA == image_format->image_channel_order);
  assert(CL_HALF_FLOAT == image_format->image_channel_data_type);

  map<cl_mem_object_type, string> lc = {
    {CL_MEM_OBJECT_BUFFER, "CL_MEM_OBJECT_BUFFER"},
    {CL_MEM_OBJECT_IMAGE2D, "CL_MEM_OBJECT_IMAGE2D"},  // all this one
    {CL_MEM_OBJECT_IMAGE3D, "CL_MEM_OBJECT_IMAGE3D"},
    {CL_MEM_OBJECT_IMAGE2D_ARRAY, "CL_MEM_OBJECT_IMAGE2D_ARRAY"},
    {CL_MEM_OBJECT_IMAGE1D, "CL_MEM_OBJECT_IMAGE1D"},
    {CL_MEM_OBJECT_IMAGE1D_ARRAY, "CL_MEM_OBJECT_IMAGE1D_ARRAY"},
    {CL_MEM_OBJECT_IMAGE1D_BUFFER, "CL_MEM_OBJECT_IMAGE1D_BUFFER"}};

  assert(image_desc->image_depth == 0);
  assert(image_desc->image_array_size == 0);
  assert(image_desc->image_slice_pitch == 0);
  //assert(image_desc->image_width * image_desc->image_height * 2 == image_desc->image_row_pitch);

  image img;
  img.image_width = image_desc->image_width;
  img.image_height = image_desc->image_height;
  img.image_row_pitch = image_desc->image_row_pitch;
  img.buffer = image_desc->buffer;

  cl_mem ret = my_clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret);
  printf("%p = clCreateImage %s -- %p -- %d %d -- %4zu x %4zu x %4zu -- %4zu %4zu %4zu\n", ret, lc[image_desc->image_type].c_str(),
    image_desc->buffer,
    image_format->image_channel_order, image_format->image_channel_data_type,
    image_desc->image_width, image_desc->image_height, image_desc->image_depth,
    image_desc->image_array_size, image_desc->image_row_pitch, image_desc->image_slice_pitch
  );
  images[ret] = img;
  return ret;
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
  cl_int (*my_clWaitForEvents)(cl_uint num_events, const cl_event *event_list);
  my_clWaitForEvents = reinterpret_cast<decltype(my_clWaitForEvents)>(dlsym(RTLD_NEXT, "REAL_clWaitForEvents"));
  printf("clWaitForEvents\n");
  return my_clWaitForEvents(num_events, event_list);
}

cl_int clReleaseEvent(cl_event event) {
  cl_int (*my_clReleaseEvent)(cl_event event);
  my_clReleaseEvent = reinterpret_cast<decltype(my_clReleaseEvent)>(dlsym(RTLD_NEXT, "REAL_clReleaseEvent"));
  printf("clReleaseEvent: %p\n", event);
  return my_clReleaseEvent(event);
}

/*size_t total = 0;

void *calloc(size_t num, size_t size) {
  void *(*my_calloc)(size_t num, size_t size);
  my_calloc = reinterpret_cast<decltype(my_calloc)>(dlsym(RTLD_NEXT, "REAL_calloc"));

  void *ret = my_calloc(num, size);

  if (do_print) {
    total += num*size;
    printf("calloc %p -- total:0x%zx -- num:0x%zx size:0x%zx\n", ret, total, num, size);
  }
  return ret;
}

void free(void *ptr) {
  void (*my_free)(void *ptr);
  my_free = reinterpret_cast<decltype(my_free)>(dlsym(RTLD_NEXT, "REAL_free"));

  if (do_print) {
    //printf("free: %p\n", ptr);
  } else {
    my_free(ptr);
  }
}*/

void *dlsym(void *handle, const char *symbol) {
  void *(*my_dlsym)(void *handle, const char *symbol) = (void *(*)(void *handle, const char *symbol))((uintptr_t)dlopen-0x2d4);
  if (memcmp("REAL_", symbol, 5) == 0) {
    return my_dlsym(handle, symbol+5);
  } else if (strcmp("clCreateProgramWithSource", symbol) == 0) {
    return (void*)clCreateProgramWithSource;
  } else if (strcmp("clCreateKernel", symbol) == 0) {
    return (void*)clCreateKernel;
  } else if (strcmp("clEnqueueNDRangeKernel", symbol) == 0) {
    return (void*)clEnqueueNDRangeKernel;
  } else if (strcmp("clSetKernelArg", symbol) == 0) {
    return (void*)clSetKernelArg;
  } else if (strcmp("clCreateBuffer", symbol) == 0) {
    return (void*)clCreateBuffer;
  } else if (strcmp("clCreateImage", symbol) == 0) {
    return (void*)clCreateImage;
  /*} else if (strcmp("clReleaseEvent", symbol) == 0) {
    return (void*)clReleaseEvent;
  } else if (strcmp("clWaitForEvents", symbol) == 0) {
    return (void*)clWaitForEvents;*/
  } else {
    //printf("dlsym %s\n", symbol);
    return my_dlsym(handle, symbol);
  }
}

int main(int argc, char* argv[]) {
  int err;
  start_time = nanos_since_boot();
  cl_device_id device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  cl_uint tmp;

  // sweet this is 64!
  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(tmp), &tmp, NULL);
  assert(err == 0);
  printf("CL_DEVICE_MAX_WRITE_IMAGE_ARGS: %u\n", tmp);

  err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(tmp), &tmp, NULL);
  assert(err == 0);
  printf("CL_DEVICE_MAX_READ_IMAGE_ARGS: %u\n", tmp);

  float *output = (float*)calloc(0x10000, sizeof(float));
  SNPEModel mdl(argv[1], output, 0, USE_GPU_RUNTIME);

  float state[TEMPORAL_SIZE];
  mdl.addRecurrent(state, TEMPORAL_SIZE);

  float desire[DESIRE_LEN];
  mdl.addDesire(desire, DESIRE_LEN);

  float traffic_convention[TRAFFIC_CONVENTION_LEN];
  mdl.addTrafficConvention(traffic_convention, TRAFFIC_CONVENTION_LEN);

  float *input = (float*)calloc(0x1000000, sizeof(float));;
  printf("************** execute 1 **************\n");
  printf("%p %p %p %p -> %p\n", input, state, desire, traffic_convention, output);
  run_num = 1; ioctl_num = 0;
  do_print = 0;
  start_time = nanos_since_boot();
  mdl.execute(input, 0);
  printf("************** execute 2 **************\n");
  run_num = 2; ioctl_num = 0;
  do_print = 0;
  mdl.execute(input, 0);
  printf("************** execute 3 **************\n");
  run_num = 3; ioctl_num = 0;

  do_print = 1;
  start_time = nanos_since_boot();
  mdl.execute(input, 0);
  do_print = 0;

  struct kgsl_gpuobj_alloc alloc;
  memset(&alloc, 0, sizeof(alloc));
  alloc.size = 0x40000;
  alloc.flags = 0x10000a00;
  int fd = 3;
  int ret = ioctl(fd, IOCTL_KGSL_GPUOBJ_ALLOC, &alloc);
  void *addr = mmap64(NULL, alloc.mmapsize, 0x3, 0x1, fd, alloc.id*0x1000);
  assert(addr != MAP_FAILED);

  intercept = 0;
  while (1) {
    printf("************** execute 4 **************\n");
    run_num = 4;
    base = (uint64_t)addr;

    uint64_t tb = nanos_since_boot();
    int i = 0;
    for (auto it = queue_cmds.begin(); it != queue_cmds.end(); ++it) {
      printf("run %2d: ", i++);
      //(*it)->exec(i == queue_cmds.size());
      (*it)->exec(true);
    }
    uint64_t te = nanos_since_boot();
    printf("model exec in %lu us\n", (te-tb)/1000);

    break;
  }

  /*FILE *f = fopen("/proc/self/maps", "rb");
  char maps[0x100000];
  int len = fread(maps, 1, sizeof(maps), f);
  maps[len] = '\0';
  fclose(f);
  printf("%s\n", maps);*/

  printf("buffers: %lu images: %lu\n", buffers.size(), images.size());
  printf("queues: %lu\n", queue_cmds.size());

  // IOCTL_KGSL_GPU_COMMAND: flags: 0x11 numcmds: 2   numobjs: 1  numsyncs: 0   context_id: 7  timestamp: 77
  /*int ts = 100;
  for (auto it = queue_cmds.begin(); it != queue_cmds.end(); ++it) {
    auto qcmd = *it;
    //disassemble((uint32_t *)qcmd.data(), qcmd.size()/4);

    struct kgsl_command_object cmdlists[2];
    struct kgsl_command_object objlists;
    struct kgsl_gpu_command cmd;
    uint8_t objs[0xc0];
    memset(objs, 0, 0xc0);

    memset(&cmd, 0, sizeof(cmd));
    memset(&cmdlists, 0, sizeof(struct kgsl_command_object)*2);
    memset(&objlists, 0, sizeof(objlists));

    cmd.flags = 0x11;
    cmd.cmdlist = (uint64_t)cmdlists;
    cmd.numcmds = 2;
    cmd.objlist = (uint64_t)objlists;
    cmd.numobjs = 1;
    cmd.numsyncs = 0;
    cmd.context_id = 7;
    cmd.timestamp = ts++;

    cmdlists[0].gpuaddr = (uint64_t)queue_init;
    cmdlists[0].size = 0xbc;
    cmdlists[0].flags = 1;
    cmdlists[1].gpuaddr = (uint64_t)qcmd.data();
    cmdlists[1].size = qcmd.size();
    cmdlists[1].flags = 1;

    objlists.gpuaddr = (uint64_t)objs;
    objlists.size = 0xc0;
    objlists.flags = 0x18;
  }*/
}

