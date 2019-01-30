#ifndef LOADYUV_H
#define LOADYUV_H

#include <inttypes.h>
#include <stdbool.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int width, height;
  cl_kernel loadys_krnl, loaduv_krnl;
} LoadYUVState;

void loadyuv_init(LoadYUVState* s, cl_context ctx, cl_device_id device_id, int width, int height);

void loadyuv_destroy(LoadYUVState* s);

void loadyuv_queue(LoadYUVState* s, cl_command_queue q,
                   cl_mem y_cl, cl_mem u_cl, cl_mem v_cl,
                   cl_mem out_cl);

#ifdef __cplusplus
}
#endif

#endif  // LOADYUV_H
