#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <EGL/eglext.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FramebufferState FramebufferState;

FramebufferState* framebuffer_init(
    const char* name, int32_t layer,
    EGLDisplay *out_display, EGLSurface *out_surface,
    int *out_w, int *out_h);

#ifdef __cplusplus
}
#endif

#endif