#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>

#include "camera_qcom.h"
// TODO: add qcom2 test

bool do_exit = false;

void cloudlog_e(int levelnum, const char* filename, int lineno, const char* func,
                const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  printf("\n");
}

void set_thread_name(const char* name) {
}

// tbuffers

void tbuffer_init2(TBuffer *tb, int num_bufs, const char* name,
                   void (*release_cb)(void* c, int idx),
                   void* cb_cookie) {
  printf("tbuffer_init2\n");
}

void tbuffer_dispatch(TBuffer *tb, int idx) {
  printf("tbuffer_dispatch\n");
}

void tbuffer_stop(TBuffer *tb) {
  printf("tbuffer_stop\n");
}

int main() {
  MultiCameraState s={};
  cameras_init(&s);
  VisionBuf camera_bufs_rear[0x10] = {0};
  VisionBuf camera_bufs_focus[0x10] = {0};
  VisionBuf camera_bufs_stats[0x10] = {0};
  VisionBuf camera_bufs_front[0x10] = {0};
  cameras_open(&s,
    camera_bufs_rear, camera_bufs_focus,
    camera_bufs_stats, camera_bufs_front);
  cameras_close(&s);
}

