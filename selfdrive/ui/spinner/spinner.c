#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>

#include <GLES3/gl3.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "common/framebuffer.h"
#include "common/spinner.h"

int main(int argc, char** argv) {
  int err;

  spin(argc, argv);

  return 0;
}
