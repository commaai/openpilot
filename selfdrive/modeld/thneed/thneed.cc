#include "selfdrive/modeld/thneed/thneed.h"

#include <dlfcn.h>
#include <sys/mman.h>

#include <cassert>
#include <cerrno>
#include <cstring>
#include <map>
#include <string>

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
//#define RUN_DISASSEMBLER
//#define RUN_OPTIMIZER

Thneed *g_thneed = NULL;
int g_fd = -1;
map<pair<cl_kernel, int>, string> g_args;
map<pair<cl_kernel, int>, int> g_args_size;
map<cl_program, string> g_program_source;

void hexdump(uint8_t *d, int len) {
  assert((len%4) == 0);
  printf("  dumping %p len 0x%x\n", d, len);
  printf("%s\n", util::hexdump(d, len).c_str());
}

// *********** ioctl interceptor ***********

extern "C" {

int (*my_ioctl)(int filedes, unsigned long request, void *argp) = NULL;
#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  request &= 0xFFFFFFFF;  // needed on QCOM2
  if (my_ioctl == NULL) my_ioctl = reinterpret_cast<decltype(my_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));
  Thneed *thneed = g_thneed;

  // save the fd
  if (request == IOCTL_KGSL_GPUOBJ_ALLOC) g_fd = filedes;

  // note that this runs always, even without a thneed object
  if (request == IOCTL_KGSL_DRAWCTXT_CREATE) {
    struct kgsl_drawctxt_create *create = (struct kgsl_drawctxt_create *)argp;
    create->flags &= ~KGSL_CONTEXT_PRIORITY_MASK;
    create->flags |= 1 << KGSL_CONTEXT_PRIORITY_SHIFT;   // priority from 1-15, 1 is max priority
    printf("IOCTL_KGSL_DRAWCTXT_CREATE: creating context with flags 0x%x\n", create->flags);
  }

  if (thneed != NULL) {
    if (request == IOCTL_KGSL_GPU_COMMAND) {
      struct kgsl_gpu_command *cmd = (struct kgsl_gpu_command *)argp;
      if (thneed->record & THNEED_RECORD) {
        thneed->timestamp = cmd->timestamp;
        thneed->context_id = cmd->context_id;
        thneed->cmds.push_back(unique_ptr<CachedCommand>(new CachedCommand(thneed, cmd)));
      }
      if (thneed->record & THNEED_DEBUG) {
        printf("IOCTL_KGSL_GPU_COMMAND(%2zu): flags: 0x%lx    context_id: %u  timestamp: %u  numcmds: %d  numobjs: %d\n",
            thneed->cmds.size(),
            cmd->flags,
            cmd->context_id, cmd->timestamp, cmd->numcmds, cmd->numobjs);
      }
    } else if (request == IOCTL_KGSL_GPUOBJ_SYNC) {
      struct kgsl_gpuobj_sync *cmd = (struct kgsl_gpuobj_sync *)argp;
      struct kgsl_gpuobj_sync_obj *objs = (struct kgsl_gpuobj_sync_obj *)(cmd->objs);

      if (thneed->record & THNEED_DEBUG) {
        printf("IOCTL_KGSL_GPUOBJ_SYNC count:%d ", cmd->count);
        for (int i = 0; i < cmd->count; i++) {
          printf(" -- offset:0x%lx len:0x%lx id:%d op:%d  ", objs[i].offset, objs[i].length, objs[i].id, objs[i].op);
        }
        printf("\n");
      }

      if (thneed->record & THNEED_RECORD) {
        thneed->cmds.push_back(unique_ptr<CachedSync>(new
              CachedSync(thneed, string((char *)objs, sizeof(struct kgsl_gpuobj_sync_obj)*cmd->count))));
      }
    } else if (request == IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID) {
      struct kgsl_device_waittimestamp_ctxtid *cmd = (struct kgsl_device_waittimestamp_ctxtid *)argp;
      if (thneed->record & THNEED_DEBUG) {
        printf("IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID: context_id: %d  timestamp: %d  timeout: %d\n",
            cmd->context_id, cmd->timestamp, cmd->timeout);
      }
    } else if (request == IOCTL_KGSL_SETPROPERTY) {
      if (thneed->record & THNEED_DEBUG) {
        struct kgsl_device_getproperty *prop = (struct kgsl_device_getproperty *)argp;
        printf("IOCTL_KGSL_SETPROPERTY: 0x%x sizebytes:%zu\n", prop->type, prop->sizebytes);
        if (thneed->record & THNEED_VERBOSE_DEBUG) {
          hexdump((uint8_t *)prop->value, prop->sizebytes);
          if (prop->type == KGSL_PROP_PWR_CONSTRAINT) {
            struct kgsl_device_constraint *constraint = (struct kgsl_device_constraint *)prop->value;
            hexdump((uint8_t *)constraint->data, constraint->size);
          }
        }
      }
    } else if (request == IOCTL_KGSL_DRAWCTXT_CREATE || request == IOCTL_KGSL_DRAWCTXT_DESTROY) {
      // this happens
    } else if (request == IOCTL_KGSL_GPUOBJ_ALLOC || request == IOCTL_KGSL_GPUOBJ_FREE) {
      // this happens
    } else {
      if (thneed->record & THNEED_DEBUG) {
        printf("other ioctl %lx\n", request);
      }
    }
  }

  int ret = my_ioctl(filedes, request, argp);
  if (ret != 0) printf("ioctl returned %d with errno %d\n", ret, errno);
  return ret;
}

}

// *********** GPUMalloc ***********

GPUMalloc::GPUMalloc(int size, int fd) {
  struct kgsl_gpuobj_alloc alloc;
  memset(&alloc, 0, sizeof(alloc));
  alloc.size = size;
  alloc.flags = 0x10000a00;
  ioctl(fd, IOCTL_KGSL_GPUOBJ_ALLOC, &alloc);
  void *addr = mmap64(NULL, alloc.mmapsize, 0x3, 0x1, fd, alloc.id*0x1000);
  assert(addr != MAP_FAILED);

  base = (uint64_t)addr;
  remaining = size;
}

GPUMalloc::~GPUMalloc() {
  // TODO: free the GPU malloced area
}

void *GPUMalloc::alloc(int size) {
  void *ret = (void*)base;
  size = (size+0xff) & (~0xFF);
  assert(size <= remaining);
  remaining -= size;
  base += size;
  return ret;
}

// *********** CachedSync, at the ioctl layer ***********

void CachedSync::exec() {
  struct kgsl_gpuobj_sync cmd;

  cmd.objs = (uint64_t)data.data();
  cmd.obj_len = data.length();
  cmd.count = data.length() / sizeof(struct kgsl_gpuobj_sync_obj);

  int ret = ioctl(thneed->fd, IOCTL_KGSL_GPUOBJ_SYNC, &cmd);
  assert(ret == 0);
}

// *********** CachedCommand, at the ioctl layer ***********

CachedCommand::CachedCommand(Thneed *lthneed, struct kgsl_gpu_command *cmd) {
  thneed = lthneed;
  assert(cmd->numsyncs == 0);

  memcpy(&cache, cmd, sizeof(cache));

  if (cmd->numcmds > 0) {
    cmds = make_unique<struct kgsl_command_object[]>(cmd->numcmds);
    memcpy(cmds.get(), (void *)cmd->cmdlist, sizeof(struct kgsl_command_object)*cmd->numcmds);
    cache.cmdlist = (uint64_t)cmds.get();
    for (int i = 0; i < cmd->numcmds; i++) {
      void *nn = thneed->ram->alloc(cmds[i].size);
      memcpy(nn, (void*)cmds[i].gpuaddr, cmds[i].size);
      cmds[i].gpuaddr = (uint64_t)nn;
    }
  }

  if (cmd->numobjs > 0) {
    objs = make_unique<struct kgsl_command_object[]>(cmd->numobjs);
    memcpy(objs.get(), (void *)cmd->objlist, sizeof(struct kgsl_command_object)*cmd->numobjs);
    cache.objlist = (uint64_t)objs.get();
    for (int i = 0; i < cmd->numobjs; i++) {
      void *nn = thneed->ram->alloc(objs[i].size);
      memset(nn, 0, objs[i].size);
      objs[i].gpuaddr = (uint64_t)nn;
    }
  }

  kq = thneed->ckq;
  thneed->ckq.clear();
}

void CachedCommand::exec() {
  cache.timestamp = ++thneed->timestamp;
  int ret = ioctl(thneed->fd, IOCTL_KGSL_GPU_COMMAND, &cache);

  if (thneed->record & THNEED_DEBUG) printf("CachedCommand::exec got %d\n", ret);

  if (thneed->record & THNEED_VERBOSE_DEBUG) {
    for (auto &it : kq) {
      it->debug_print(false);
    }
    #ifdef RUN_DISASSEMBLER
      // assuming 2 commands
      disassemble(0);
      disassemble(1);
    #endif
  }

  assert(ret == 0);
}

// *********** Thneed ***********

Thneed::Thneed(bool do_clinit) {
  if (do_clinit) clinit();
  assert(g_fd != -1);
  fd = g_fd;
  ram = make_unique<GPUMalloc>(0x80000, fd);
  record = THNEED_RECORD;
  timestamp = -1;
  g_thneed = this;
}

void Thneed::stop() {
  find_inputs_outputs();
  printf("Thneed::stop: recorded %lu commands\n", cmds.size());
  record = 0;
}

void Thneed::find_inputs_outputs() {
  cl_int err;
  if (inputs.size() > 0) return;

  // save the global inputs/outputs
  for (auto &k : kq) {
    for (int i = 0; i < k->num_args; i++) {
      if (k->name == "zero_pad_image_float" && k->arg_names[i] == "input") {
        cl_mem aa = *(cl_mem*)(k->args[i].data());
        input_clmem.push_back(aa);

        size_t sz;
        clGetMemObjectInfo(aa, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
        input_sizes.push_back(sz);

        void *ret = clEnqueueMapBuffer(command_queue, aa, CL_TRUE, CL_MAP_WRITE, 0, sz, 0, NULL, NULL, &err);
        assert(err == CL_SUCCESS);
        inputs.push_back(ret);
      }

      if (k->name == "image2d_to_buffer_float" && k->arg_names[i] == "output") {
        output = *(cl_mem*)(k->args[i].data());
      }
    }
  }
}

void Thneed::copy_inputs(float **finputs) {
  //cl_int ret;
  for (int idx = 0; idx < inputs.size(); ++idx) {
    if (record & THNEED_DEBUG) printf("copying %lu -- %p -> %p\n", input_sizes[idx], finputs[idx], inputs[idx]);
    if (finputs[idx] != NULL) memcpy(inputs[idx], finputs[idx], input_sizes[idx]);
  }
}

void Thneed::copy_output(float *foutput) {
  if (output != NULL) {
    size_t sz;
    clGetMemObjectInfo(output, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
    if (record & THNEED_DEBUG) printf("copying %lu for output %p -> %p\n", sz, output, foutput);
    clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sz, foutput, 0, NULL, NULL);
  } else {
    printf("CAUTION: model output is NULL, does it have no outputs?\n");
  }
}

void Thneed::wait() {
  struct kgsl_device_waittimestamp_ctxtid wait;
  wait.context_id = context_id;
  wait.timestamp = timestamp;
  wait.timeout = -1;

  uint64_t tb = nanos_since_boot();
  int wret = ioctl(fd, IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, &wait);
  uint64_t te = nanos_since_boot();

  if (record & THNEED_DEBUG) printf("wait %d after %lu us\n", wret, (te-tb)/1000);
}

void Thneed::execute(float **finputs, float *foutput, bool slow) {
  uint64_t tb, te;
  if (record & THNEED_DEBUG) tb = nanos_since_boot();

  // ****** copy inputs
  copy_inputs(finputs);

  // ****** set power constraint
  int ret;
  struct kgsl_device_constraint_pwrlevel pwrlevel;
  pwrlevel.level = KGSL_CONSTRAINT_PWR_MAX;

  struct kgsl_device_constraint constraint;
  constraint.type = KGSL_CONSTRAINT_PWRLEVEL;
  constraint.context_id = context_id;
  constraint.data = (void*)&pwrlevel;
  constraint.size = sizeof(pwrlevel);

  struct kgsl_device_getproperty prop;
  prop.type = KGSL_PROP_PWR_CONSTRAINT;
  prop.value = (void*)&constraint;
  prop.sizebytes = sizeof(constraint);
  ret = ioctl(fd, IOCTL_KGSL_SETPROPERTY, &prop);
  assert(ret == 0);

  // ****** run commands
  int i = 0;
  for (auto &it : cmds) {
    ++i;
    if (record & THNEED_DEBUG) printf("run %2d @ %7lu us: ", i, (nanos_since_boot()-tb)/1000);
    it->exec();
    if ((i == cmds.size()) || slow) wait();
  }

  // ****** copy outputs
  copy_output(foutput);

  // ****** unset power constraint
  constraint.type = KGSL_CONSTRAINT_NONE;
  constraint.data = NULL;
  constraint.size = 0;

  ret = ioctl(fd, IOCTL_KGSL_SETPROPERTY, &prop);
  assert(ret == 0);

  if (record & THNEED_DEBUG) {
    te = nanos_since_boot();
    printf("model exec in %lu us\n", (te-tb)/1000);
  }
}

void Thneed::clinit() {
  device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  context = CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err));
  //cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
  cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, 0, 0};
  command_queue = CL_CHECK_ERR(clCreateCommandQueueWithProperties(context, device_id, props, &err));
  printf("Thneed::clinit done\n");
}

cl_int Thneed::clexec() {
  printf("Thneed::clexec: running %lu queued kernels\n", kq.size());
  for (auto &k : kq) {
    if (record & THNEED_RECORD) ckq.push_back(k);
    cl_int ret = k->exec();
    assert(ret == CL_SUCCESS);
  }
  return clFinish(command_queue);
}

// *********** OpenCL interceptor ***********

cl_int thneed_clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
  g_args_size[make_pair(kernel, arg_index)] = arg_size;
  if (arg_value != NULL) {
    g_args[make_pair(kernel, arg_index)] = string((char*)arg_value, arg_size);
  } else {
    g_args[make_pair(kernel, arg_index)] = string("");
  }
  cl_int ret = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
  return ret;
}

cl_int thneed_clEnqueueNDRangeKernel(cl_command_queue command_queue,
  cl_kernel kernel,
  cl_uint work_dim,
  const size_t *global_work_offset,
  const size_t *global_work_size,
  const size_t *local_work_size,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event) {

  Thneed *thneed = g_thneed;

  // SNPE doesn't use these
  assert(num_events_in_wait_list == 0);
  assert(global_work_offset == NULL);
  assert(event_wait_list == NULL);

  cl_int ret = 0;
  if (thneed != NULL && thneed->record & THNEED_RECORD) {
    if (thneed->context == NULL) {
      thneed->command_queue = command_queue;
      clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(thneed->context), &thneed->context, NULL);
      clGetContextInfo(thneed->context, CL_CONTEXT_DEVICES, sizeof(thneed->device_id), &thneed->device_id, NULL);
    }

    // if we are recording, we don't actually enqueue the kernel
    thneed->kq.push_back(unique_ptr<CLQueuedKernel>(new CLQueuedKernel(thneed, kernel, work_dim, global_work_size, local_work_size)));
    *event = NULL;
  } else {
    ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
      global_work_offset, global_work_size, local_work_size,
      num_events_in_wait_list, event_wait_list, event);
  }

  return ret;
}

cl_int thneed_clFinish(cl_command_queue command_queue) {
  Thneed *thneed = g_thneed;

  if (thneed != NULL && thneed->record & THNEED_RECORD) {
    #ifdef RUN_OPTIMIZER
      thneed->optimize();
    #endif
    return thneed->clexec();
  } else {
    return clFinish(command_queue);
  }
}

cl_program thneed_clCreateProgramWithSource(cl_context context, cl_uint count, const char **strings, const size_t *lengths, cl_int *errcode_ret) {
  assert(count == 1);
  cl_program ret = clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
  g_program_source[ret] = strings[0];
  return ret;
}

void *dlsym(void *handle, const char *symbol) {
#if defined(QCOM) || defined(QCOM2)
  void *(*my_dlsym)(void *handle, const char *symbol) = (void *(*)(void *handle, const char *symbol))((uintptr_t)dlopen + DLSYM_OFFSET);
#else
  #error "Unsupported platform for thneed"
#endif
  if (memcmp("REAL_", symbol, 5) == 0) {
    return my_dlsym(handle, symbol+5);
  } else if (strcmp("clFinish", symbol) == 0) {
    return (void*)thneed_clFinish;
  } else if (strcmp("clEnqueueNDRangeKernel", symbol) == 0) {
    return (void*)thneed_clEnqueueNDRangeKernel;
  } else if (strcmp("clSetKernelArg", symbol) == 0) {
    return (void*)thneed_clSetKernelArg;
  } else if (strcmp("clCreateProgramWithSource", symbol) == 0) {
    return (void*)thneed_clCreateProgramWithSource;
  } else {
    return my_dlsym(handle, symbol);
  }
}

// *********** CLQueuedKernel ***********

CLQueuedKernel::CLQueuedKernel(Thneed *lthneed,
                               cl_kernel _kernel,
                               cl_uint _work_dim,
                               const size_t *_global_work_size,
                               const size_t *_local_work_size) {
  thneed = lthneed;
  kernel = _kernel;
  work_dim = _work_dim;
  assert(work_dim <= 3);
  for (int i = 0; i < work_dim; i++) {
    global_work_size[i] = _global_work_size[i];
    local_work_size[i] = _local_work_size[i];
  }

  char _name[0x100];
  clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(_name), _name, NULL);
  name = string(_name);
  clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);

  // get args
  for (int i = 0; i < num_args; i++) {
    char arg_name[0x100];
    clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
    arg_names.push_back(string(arg_name));
    clGetKernelArgInfo(kernel, i, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_name), arg_name, NULL);
    arg_types.push_back(string(arg_name));

    args.push_back(g_args[make_pair(kernel, i)]);
    args_size.push_back(g_args_size[make_pair(kernel, i)]);
  }

  // get program
  clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, NULL);
}

int CLQueuedKernel::get_arg_num(const char *search_arg_name) {
  for (int i = 0; i < num_args; i++) {
    if (arg_names[i] == search_arg_name) return i;
  }
  printf("failed to find %s in %s\n", search_arg_name, name.c_str());
  assert(false);
}

cl_int CLQueuedKernel::exec() {
  if (kernel == NULL) {
    kernel = clCreateKernel(program, name.c_str(), NULL);
    arg_names.clear();
    arg_types.clear();

    for (int j = 0; j < num_args; j++) {
      char arg_name[0x100];
      clGetKernelArgInfo(kernel, j, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
      arg_names.push_back(string(arg_name));
      clGetKernelArgInfo(kernel, j, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_name), arg_name, NULL);
      arg_types.push_back(string(arg_name));

      cl_int ret;
      if (args[j].size() != 0) {
        assert(args[j].size() == args_size[j]);
        ret = thneed_clSetKernelArg(kernel, j, args[j].size(), args[j].data());
      } else {
        ret = thneed_clSetKernelArg(kernel, j, args_size[j], NULL);
      }
      assert(ret == CL_SUCCESS);
    }
  }

  if (thneed->record & THNEED_DEBUG) {
    debug_print(thneed->record & THNEED_VERBOSE_DEBUG);
  }

  return clEnqueueNDRangeKernel(thneed->command_queue,
    kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
}

void CLQueuedKernel::debug_print(bool verbose) {
  printf("%p %56s -- ", kernel, name.c_str());
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", global_work_size[i]);
  }
  printf(" -- ");
  for (int i = 0; i < work_dim; i++) {
    printf("%4zu ", local_work_size[i]);
  }
  printf("\n");

  if (verbose) {
    for (int i = 0; i < num_args; i++) {
      string arg = args[i];
      printf("  %s %s", arg_types[i].c_str(), arg_names[i].c_str());
      void *arg_value = (void*)arg.data();
      int arg_size = arg.size();
      if (arg_size == 0) {
        printf(" (size) %d", args_size[i]);
      } else if (arg_size == 1) {
        printf(" = %d", *((char*)arg_value));
      } else if (arg_size == 2) {
        printf(" = %d", *((short*)arg_value));
      } else if (arg_size == 4) {
        if (arg_types[i] == "float") {
          printf(" = %f", *((float*)arg_value));
        } else {
          printf(" = %d", *((int*)arg_value));
        }
      } else if (arg_size == 8) {
        cl_mem val = (cl_mem)(*((uintptr_t*)arg_value));
        printf(" = %p", val);
        if (val != NULL) {
          if (arg_types[i] == "image2d_t" || arg_types[i] == "image1d_t") {
            cl_image_format format;
            size_t width, height, depth, array_size, row_pitch, slice_pitch;
            cl_mem buf;
            clGetImageInfo(val, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);
            assert(format.image_channel_order == CL_RGBA);
            assert(format.image_channel_data_type == CL_HALF_FLOAT);
            clGetImageInfo(val, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);
            clGetImageInfo(val, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);
            clGetImageInfo(val, CL_IMAGE_ROW_PITCH, sizeof(row_pitch), &row_pitch, NULL);
            clGetImageInfo(val, CL_IMAGE_DEPTH, sizeof(depth), &depth, NULL);
            clGetImageInfo(val, CL_IMAGE_ARRAY_SIZE, sizeof(array_size), &array_size, NULL);
            clGetImageInfo(val, CL_IMAGE_SLICE_PITCH, sizeof(slice_pitch), &slice_pitch, NULL);
            assert(depth == 0);
            assert(array_size == 0);
            assert(slice_pitch == 0);

            clGetImageInfo(val, CL_IMAGE_BUFFER, sizeof(buf), &buf, NULL);
            size_t sz;
            clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
            printf(" image %zu x %zu rp %zu @ %p buffer %zu", width, height, row_pitch, buf, sz);
          } else {
            size_t sz;
            clGetMemObjectInfo(val, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
            printf(" buffer %zu", sz);
          }
        }
      }
      printf("\n");
    }
  }
}
