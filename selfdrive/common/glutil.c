#include <stdlib.h>
#include <stdio.h>

#include "glutil.h"

GLuint load_shader(GLenum shaderType, const char *src) {
  GLint status = 0, len = 0;
  GLuint shader;

  if (!(shader = glCreateShader(shaderType)))
    return 0;

  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

  if (status)
    return shader;

  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
  if (len) {
    char *msg = (char*)malloc(len);
    if (msg) {
      glGetShaderInfoLog(shader, len, NULL, msg);
      msg[len-1] = 0;
      fprintf(stderr, "error compiling shader:\n%s\n", msg);
      free(msg);
    }
  }
  glDeleteShader(shader);
  return 0;
}

GLuint load_program(const char *vert_src, const char *frag_src) {
  GLuint vert, frag, prog;
  GLint status = 0, len = 0;

  if (!(vert = load_shader(GL_VERTEX_SHADER, vert_src)))
    return 0;
  if (!(frag = load_shader(GL_FRAGMENT_SHADER, frag_src)))
    goto fail_frag;
  if (!(prog = glCreateProgram()))
    goto fail_prog;

  glAttachShader(prog, vert);
  glAttachShader(prog, frag);
  glLinkProgram(prog);

  glGetProgramiv(prog, GL_LINK_STATUS, &status);
  if (status)
    return prog;

  glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
  if (len) {
    char *buf = (char*) malloc(len);
    if (buf) {
      glGetProgramInfoLog(prog, len, NULL, buf);
      buf[len-1] = 0;
      fprintf(stderr, "error linking program:\n%s\n", buf);
      free(buf);
    }
  }
  glDeleteProgram(prog);
fail_prog:
  glDeleteShader(frag);
fail_frag:
  glDeleteShader(vert);
  return 0;
}
