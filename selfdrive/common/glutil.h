#pragma once

#ifdef __APPLE__
  #include <OpenGL/gl3.h>
#else
  #include <GLES3/gl3.h>
#endif

class GLShader {
public:
  GLShader(const char *vert_src, const char *frag_src);
  ~GLShader();
  GLuint prog = 0, vert = 0, frag = 0;
  GLint texture_loc = 0, transform_loc = 0;
};
