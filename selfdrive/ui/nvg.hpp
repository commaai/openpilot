#pragma once

#ifdef __APPLE__
#include <OpenGL/gl3.h>
#define NANOVG_GL3_IMPLEMENTATION
#define nvgCreate nvgCreateGL3
#else
#include <GLES3/gl3.h>
#include <EGL/egl.h>
#define NANOVG_GLES3_IMPLEMENTATION
#define nvgCreate nvgCreateGLES3
#endif


// #include <GLES2/gl2.h>
// #include <EGL/egl.h>
// #define NANOVG_GLES2_IMPLEMENTATION
// #define nvgCreate nvgCreateGLES2
