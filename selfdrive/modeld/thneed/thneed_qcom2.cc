#include "selfdrive/modeld/thneed/thneed.h"

#include <dlfcn.h>
#include <sys/mman.h>

#include <cassert>
#include <cerrno>
#include <cstring>
#include <map>
#include <string>

#include "common/clutil.h"
#include "common/timing.h"

Thneed *g_thneed = NULL;
int g_fd = -1;

void hexdump(uint8_t *d, int len) {
  assert((len%4) == 0);
  printf("  dumping %p len 0x%x\n", d, len);
  for (int i = 0; i < len/4; i++) {
    if (i != 0 && (i%0x10) == 0) printf("\n");
    printf("%8x ", d[i]);
  }
  printf("\n");
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
    create->flags |= 6 << KGSL_CONTEXT_PRIORITY_SHIFT;   // priority from 1-15, 1 is max priority
    printf("IOCTL_KGSL_DRAWCTXT_CREATE: creating context with flags 0x%x\n", create->flags);
  }

  if (thneed != NULL) {
    if (request == IOCTL_KGSL_GPU_COMMAND) {
      struct kgsl_gpu_command *cmd = (struct kgsl_gpu_command *)argp;
      if (thneed->record) {
        thneed->timestamp = cmd->timestamp;
        thneed->context_id = cmd->context_id;
        thneed->cmds.push_back(unique_ptr<CachedCommand>(new CachedCommand(thneed, cmd)));
      }
      if (thneed->debug >= 1) {
        printf("IOCTL_KGSL_GPU_COMMAND(%2zu): flags: 0x%lx    context_id: %u  timestamp: %u  numcmds: %d  numobjs: %d\n",
            thneed->cmds.size(),
            cmd->flags,
            cmd->context_id, cmd->timestamp, cmd->numcmds, cmd->numobjs);
      }
    } else if (request == IOCTL_KGSL_GPUOBJ_SYNC) {
      struct kgsl_gpuobj_sync *cmd = (struct kgsl_gpuobj_sync *)argp;
      struct kgsl_gpuobj_sync_obj *objs = (struct kgsl_gpuobj_sync_obj *)(cmd->objs);

      if (thneed->debug >= 2) {
        printf("IOCTL_KGSL_GPUOBJ_SYNC count:%d ", cmd->count);
        for (int i = 0; i < cmd->count; i++) {
          printf(" -- offset:0x%lx len:0x%lx id:%d op:%d  ", objs[i].offset, objs[i].length, objs[i].id, objs[i].op);
        }
        printf("\n");
      }

      if (thneed->record) {
        thneed->cmds.push_back(unique_ptr<CachedSync>(new
              CachedSync(thneed, string((char *)objs, sizeof(struct kgsl_gpuobj_sync_obj)*cmd->count))));
      }
    } else if (request == IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID) {
      struct kgsl_device_waittimestamp_ctxtid *cmd = (struct kgsl_device_waittimestamp_ctxtid *)argp;
      if (thneed->debug >= 1) {
        printf("IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID: context_id: %d  timestamp: %d  timeout: %d\n",
            cmd->context_id, cmd->timestamp, cmd->timeout);
      }
    } else if (request == IOCTL_KGSL_SETPROPERTY) {
      if (thneed->debug >= 1) {
        struct kgsl_device_getproperty *prop = (struct kgsl_device_getproperty *)argp;
        printf("IOCTL_KGSL_SETPROPERTY: 0x%x sizebytes:%zu\n", prop->type, prop->sizebytes);
        if (thneed->debug >= 2) {
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
      if (thneed->debug >= 1) {
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

  if (thneed->debug >= 1) printf("CachedCommand::exec got %d\n", ret);

  if (thneed->debug >= 2) {
    for (auto &it : kq) {
      it->debug_print(false);
    }
  }

  assert(ret == 0);
}

// *********** Thneed ***********

Thneed::Thneed(bool do_clinit, cl_context _context) {
  // TODO: QCOM2 actually requires a different context
  //context = _context;
  if (do_clinit) clinit();
  assert(g_fd != -1);
  fd = g_fd;
  ram = make_unique<GPUMalloc>(0x80000, fd);
  timestamp = -1;
  g_thneed = this;
  char *thneed_debug_env = getenv("THNEED_DEBUG");
  debug = (thneed_debug_env != NULL) ? atoi(thneed_debug_env) : 0;
}

void Thneed::wait() {
  struct kgsl_device_waittimestamp_ctxtid wait;
  wait.context_id = context_id;
  wait.timestamp = timestamp;
  wait.timeout = -1;

  uint64_t tb = nanos_since_boot();
  int wret = ioctl(fd, IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID, &wait);
  uint64_t te = nanos_since_boot();

  if (debug >= 1) printf("wait %d after %lu us\n", wret, (te-tb)/1000);
}

void Thneed::execute(float **finputs, float *foutput, bool slow) {
  uint64_t tb, te;
  if (debug >= 1) tb = nanos_since_boot();

  // ****** copy inputs
  copy_inputs(finputs, true);

  // ****** run commands
  int i = 0;
  for (auto &it : cmds) {
    ++i;
    if (debug >= 1) printf("run %2d @ %7lu us: ", i, (nanos_since_boot()-tb)/1000);
    it->exec();
    if ((i == cmds.size()) || slow) wait();
  }

  // ****** copy outputs
  copy_output(foutput);

  if (debug >= 1) {
    te = nanos_since_boot();
    printf("model exec in %lu us\n", (te-tb)/1000);
  }
}
