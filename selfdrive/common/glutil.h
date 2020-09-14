#ifndef COMMON_GLUTIL_H
#define COMMON_GLUTIL_H

#ifdef __APPLE__
  #include <OpenGL/gl3.h>
#else
  #include <GLES3/gl3.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FrameShader{
  GLuint vert;
  GLuint frag;
  GLuint prog;
} FrameShader;

int frame_shader_init(FrameShader *s, const char *vert_src, const char *frag_src);
void frame_shader_destroy(FrameShader *s);

#ifdef __cplusplus
}
#endif

#endif
