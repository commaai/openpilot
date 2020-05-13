#include "thneed.h"
#include <cassert>
#include <sys/mman.h>
#include <dlfcn.h>

Thneed *g_thneed = NULL;
int gfd = -1;

static inline uint64_t nanos_since_boot() {
  struct timespec t;
  clock_gettime(CLOCK_BOOTTIME, &t);
  return t.tv_sec * 1000000000ULL + t.tv_nsec;
}

extern "C" {

int (*my_ioctl)(int filedes, unsigned long request, void *argp) = NULL;
#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  if (my_ioctl == NULL) my_ioctl = reinterpret_cast<decltype(my_ioctl)>(dlsym(RTLD_NEXT, "ioctl"));

  // save the fd
  if (request == IOCTL_KGSL_GPUOBJ_ALLOC) gfd = filedes;

  if (g_thneed != NULL && g_thneed->record) {
    if (request == IOCTL_KGSL_GPU_COMMAND) {
      struct kgsl_gpu_command *cmd = (struct kgsl_gpu_command *)argp;
      printf("IOCTL_KGSL_GPU_COMMAND: flags: 0x%lx    context_id: %u  timestamp: %u\n",
          cmd->flags,
          cmd->context_id, cmd->timestamp);

      CachedCommand *ccmd = new CachedCommand(g_thneed, cmd, filedes);
      g_thneed->cmds.push_back(ccmd);
    } else if (request == IOCTL_KGSL_GPUOBJ_SYNC) {
      struct kgsl_gpuobj_sync *cmd = (struct kgsl_gpuobj_sync *)argp;
      struct kgsl_gpuobj_sync_obj *objs = (struct kgsl_gpuobj_sync_obj *)(cmd->objs);

      printf("IOCTL_KGSL_GPUOBJ_SYNC count:%d ", cmd->count);
      for (int i = 0; i < cmd->count; i++) {
        printf(" -- offset:0x%lx len:0x%lx id:%d op:%d  ", objs[i].offset, objs[i].length, objs[i].id, objs[i].op);
      }
      printf("\n");
    } else if (request == IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID) {
      struct kgsl_device_waittimestamp_ctxtid *cmd = (struct kgsl_device_waittimestamp_ctxtid *)argp;
      printf("IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID: context_id: %d  timestamp: %d  timeout: %d\n",
          cmd->context_id, cmd->timestamp, cmd->timeout);
    } else if (request == IOCTL_KGSL_SETPROPERTY) {
      struct kgsl_device_getproperty *prop = (struct kgsl_device_getproperty *)argp;
      printf("IOCTL_KGSL_SETPROPERTY: 0x%x\n", prop->type);
    }
  }

  return my_ioctl(filedes, request, argp);
}

}

GPUMalloc::GPUMalloc(int size, int fd) {
  struct kgsl_gpuobj_alloc alloc;
  memset(&alloc, 0, sizeof(alloc));
  alloc.size = size;
  alloc.flags = 0x10000a00;
  int ret = ioctl(fd, IOCTL_KGSL_GPUOBJ_ALLOC, &alloc);
  void *addr = mmap64(NULL, alloc.mmapsize, 0x3, 0x1, fd, alloc.id*0x1000);
  assert(addr != MAP_FAILED);

  base = (uint64_t)addr;
  remaining = size;
}

void *GPUMalloc::alloc(int size) {
  if (size > remaining) return NULL;
  remaining -= size;
  void *ret = (void*)base;
  base += (size+0xff) & (~0xFF);
  return ret;
}

CachedCommand::CachedCommand(Thneed *lthneed, struct kgsl_gpu_command *cmd, int lfd) {
  fd = lfd;
  thneed = lthneed;
  assert(cmd->numcmds == 2);
  assert(cmd->numobjs == 1);
  assert(cmd->numsyncs == 0);
  thneed->timestamp = cmd->timestamp;

  memcpy(cmds, (void *)cmd->cmdlist, sizeof(struct kgsl_command_object)*2);
  memcpy(objs, (void *)cmd->objlist, sizeof(struct kgsl_command_object)*1);

  memcpy(&cache, cmd, sizeof(cache));
  cache.cmdlist = (uint64_t)cmds;
  cache.objlist = (uint64_t)objs;

  for (int i = 0; i < cmd->numcmds; i++) {
    void *nn = thneed->ram->alloc(cmds[i].size);
    memcpy(nn, (void*)cmds[i].gpuaddr, cmds[i].size);
    cmds[i].gpuaddr = (uint64_t)nn;
  }

  for (int i = 0; i < cmd->numobjs; i++) {
    void *nn = thneed->ram->alloc(objs[i].size);
    memset(nn, 0, objs[i].size);
    objs[i].gpuaddr = (uint64_t)nn;
  }
}

void CachedCommand::exec(bool wait) {
  cache.timestamp = ++thneed->timestamp;
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

Thneed::Thneed() {
  assert(gfd != -1);
  ram = new GPUMalloc(0x40000, gfd);
  record = 1;
  timestamp = 0;
  g_thneed = this;
}

void Thneed::stop() {
  record = 0;
}

void Thneed::execute() {
  int i;
  for (auto it = cmds.begin(); it != cmds.end(); ++it) {
    printf("run %2d: ", i++);
    (*it)->exec(i == cmds.size());
  }
}

