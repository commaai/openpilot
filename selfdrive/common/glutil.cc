#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string>

#include "glutil.h"

static GLuint load_shader(GLenum shaderType, const char *src) {
  GLint status = 0, len = 0;
  GLuint shader = glCreateShader(shaderType);
  assert(shader != 0);

  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (!status) {
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
    if (len) {
      std::string msg(len, '\0');
      glGetShaderInfoLog(shader, len, NULL, msg.data());
      fprintf(stderr, "error compiling shader:\n%s\n", msg.c_str());
    }
    assert(0);
  }
  return shader;
}

GLShader::GLShader(const char *vert_src, const char *frag_src) {
  GLint status = 0, len = 0;
  prog = glCreateProgram();
  assert(prog != 0);

  vert = load_shader(GL_VERTEX_SHADER, vert_src);
  frag = load_shader(GL_FRAGMENT_SHADER, frag_src);
  glAttachShader(prog, vert);
  glAttachShader(prog, frag);
  glLinkProgram(prog);

  glGetProgramiv(prog, GL_LINK_STATUS, &status);
  if (!status) {
    glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
    if (len) {
      std::string msg(len, '\0');
      glGetProgramInfoLog(prog, len, NULL, msg.data());
      fprintf(stderr, "error linking program:\n%s\n", msg.c_str());
    }
    assert(0);
  }
}

GLShader::~GLShader() {
  glDeleteProgram(prog);
  glDeleteShader(frag);
  glDeleteShader(vert);
}

GLuint GLShader::getUniformLocation(const char *name) {
  auto it = uniform_loc_map.find(name);
  if (it == uniform_loc_map.end()) {
    it = uniform_loc_map.insert(it, {name, glGetUniformLocation(prog, name)});
  }
  return it->second;
}
