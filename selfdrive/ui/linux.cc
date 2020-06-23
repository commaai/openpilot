#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "ui.hpp"

#ifndef __APPLE__
#define GLFW_INCLUDE_ES2
#else
#define GLFW_INCLUDE_GLCOREARB
#endif

#define GLFW_INCLUDE_GLEXT
#include <GLFW/glfw3.h>

typedef struct FramebufferState FramebufferState;
typedef struct TouchState TouchState;

extern "C" {

FramebufferState* framebuffer_init(
    const char* name, int32_t layer, int alpha,
    int *out_w, int *out_h) {
  glfwInit();

#ifndef __APPLE__
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#else
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
#endif
  glfwWindowHint(GLFW_RESIZABLE, 0);
  GLFWwindow* window;
  window = glfwCreateWindow(1920, 1080, "ui", NULL, NULL);
  if (!window) {
    printf("glfwCreateWindow failed\n");
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // clear screen
  glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  framebuffer_swap((FramebufferState*)window);

  if (out_w) *out_w = 1920;
  if (out_h) *out_h = 1080;

  return (FramebufferState*)window;
}

void framebuffer_set_power(FramebufferState *s, int mode) {
}

void framebuffer_swap(FramebufferState *s) {
  glfwSwapBuffers((GLFWwindow*)s);
  glfwPollEvents();
}

bool set_brightness(int brightness) { return true; }

void touch_init(TouchState *s) {
  printf("touch_init\n");
}

int touch_poll(TouchState *s, int* out_x, int* out_y, int timeout) {
  return -1;
}

int touch_read(TouchState *s, int* out_x, int* out_y) {
  return -1;
}

}

#include "sound.hpp"

bool Sound::init(int volume) { return true; }
bool Sound::play(AudibleAlert alert) { return true; }
void Sound::stop() {}
void Sound::setVolume(int volume) {}
Sound::~Sound() {}

#include "common/visionimg.h"
#include <sys/mman.h>

GLuint visionimg_to_gl(const VisionImg *img, EGLImageKHR *pkhr, void **pph) {
  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img->width, img->height, 0, GL_RGB, GL_UNSIGNED_BYTE, *pph);
  glGenerateMipmap(GL_TEXTURE_2D);
  *pkhr = (EGLImageKHR)1; // not NULL
  return texture;
}

void visionimg_destroy_gl(EGLImageKHR khr, void *ph) {
  // empty
}

