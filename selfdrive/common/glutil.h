#pragma once

#ifdef __APPLE__
  #include <OpenGL/gl3.h>
#else
  #include <GLES3/gl3.h>
#endif
#include <map>

class GLShader {
public:
  GLShader(const char *vert_src, const char *frag_src);
  ~GLShader();
  GLuint getUniformLocation(const char * name);
  GLuint prog = 0;

private:
  GLuint vert = 0, frag = 0;
  std::map<const char*, GLint> uniform_loc_map;
};
