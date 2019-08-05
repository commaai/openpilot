#ifndef COMMON_GLUTIL_H
#define COMMON_GLUTIL_H

#include <GLES3/gl3.h>
GLuint load_shader(GLenum shaderType, const char *src);
GLuint load_program(const char *vert_src, const char *frag_src);

#endif
