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

GLuint load_shader(GLenum shaderType, const char *src);
GLuint load_program(const char *vert_src, const char *frag_src);

#ifdef __cplusplus
}
#endif

#endif
