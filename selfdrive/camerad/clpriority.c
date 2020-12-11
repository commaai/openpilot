// raise GPU priority on QCOM devices
#if defined(QCOM) || defined(QCOM2)

#include <stdio.h>
#include <dlfcn.h>
#include "modeld/thneed/include/msm_kgsl.h"

int (*my_ioctl)(int filedes, unsigned long request, void *argp) = NULL;
#undef ioctl
int ioctl(int filedes, unsigned long request, void *argp) {
  request &= 0xFFFFFFFF;
  if (my_ioctl == NULL) my_ioctl = dlsym(RTLD_NEXT, "ioctl");

  if (request == IOCTL_KGSL_DRAWCTXT_CREATE) {
    struct kgsl_drawctxt_create *create = (struct kgsl_drawctxt_create *)argp;
    create->flags &= ~KGSL_CONTEXT_PRIORITY_MASK;
    create->flags |= 2 << KGSL_CONTEXT_PRIORITY_SHIFT;   // priority from 1-15, 1 is max priority, 2 is for camerad
    printf("IOCTL_KGSL_DRAWCTXT_CREATE: creating context with flags 0x%x\n", create->flags);
  }

  return my_ioctl(filedes, request, argp);
}

#endif

