#include <stdlib.h>
#include <stdio.h>

#include "glutil.h"

static GLuint load_shader(GLenum shaderType, const char *src) {
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

int frame_shader_init(FrameShader *s, const char *vert_src, const char *frag_src) {
  GLint status = 0, len = 0;

  if (!(s->vert = load_shader(GL_VERTEX_SHADER, vert_src)))
    return 0;
  if (!(s->frag = load_shader(GL_FRAGMENT_SHADER, frag_src)))
    goto fail_frag;
  if (!(s->prog = glCreateProgram()))
    goto fail_prog;

  glAttachShader(s->prog, s->vert);
  glAttachShader(s->prog, s->frag);
  glLinkProgram(s->prog);

  glGetProgramiv(s->prog, GL_LINK_STATUS, &status);
  if (status)
    return 1;

  glGetProgramiv(s->prog, GL_INFO_LOG_LENGTH, &len);
  if (len) {
    char *buf = (char*) malloc(len);
    if (buf) {
      glGetProgramInfoLog(s->prog, len, NULL, buf);
      buf[len-1] = 0;
      fprintf(stderr, "error linking program:\n%s\n", buf);
      free(buf);
    }
  }
  glDeleteProgram(s->prog);
fail_prog:
  glDeleteShader(s->frag);
fail_frag:
  glDeleteShader(s->vert);
  return 0;
}

void frame_shader_destroy(FrameShader *s) {
  glDeleteProgram(s->prog);
  glDeleteShader(s->vert);
  glDeleteShader(s->frag);
}
