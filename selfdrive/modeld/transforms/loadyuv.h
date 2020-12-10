#pragma once

#include <inttypes.h>
#include <stdbool.h>

#include "clutil.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int width, height;
  cl_kernel loadys_krnl, loaduv_krnl;
} LoadYUVState;

void loadyuv_init(LoadYUVState* s, CLContext *ctx, int width, int height);

void loadyuv_destroy(LoadYUVState* s);

void loadyuv_queue(LoadYUVState* s, cl_command_queue q,
                   cl_mem y_cl, cl_mem u_cl, cl_mem v_cl,
                   cl_mem out_cl);

#ifdef __cplusplus
}
#endif
