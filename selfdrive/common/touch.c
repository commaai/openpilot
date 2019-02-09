#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/poll.h>
#include <linux/input.h>

#include "touch.h"

static int find_dev() {
  int err;

  int ret = -1;

  DIR *dir = opendir("/dev/input");
  assert(dir);
  struct dirent* de = NULL;
  while ((de = readdir(dir))) {
    if (strncmp(de->d_name, "event", 5)) continue;

    int fd = openat(dirfd(dir), de->d_name, O_RDONLY);
    assert(fd >= 0);
    
    FILE *fp;
    FILE *fp2;
    char str[1000];
    char str2[5];
    char* filename = "/proc/cmdline";
    char* filename2 = "/VERSION";

    fp = fopen(filename, "r");
    fp2 = fopen(filename2, "r");
    if (fp == NULL){
      printf("Could not open file %s",filename);
      return 0;
    }
    if (fp2 == NULL){
      printf("Could not open file %s", filename2);
      return 0;
    }
    fgets(str, 1000, fp);
    fgets(str2, 5, fp2);
    fclose(fp);
    fclose(fp2);
    if (strstr(str, "letv") != NULL || strstr(str2, "6") != NULL){
      char name[128] = {0};
      err = ioctl(fd, EVIOCGNAME(sizeof(name) - 1), &name);
      assert(err >= 0);

      unsigned long ev_bits[8] = {0};
      err = ioctl(fd, EVIOCGBIT(0, sizeof(ev_bits)), ev_bits);
      assert(err >= 0);

      if (strncmp(name, "synaptics", 9) == 0 && ev_bits[0] == 0xb) {
        ret = fd;
        break;
      }
    } else {
      unsigned char ev_bits[KEY_MAX / 8 + 1];
      err = ioctl(fd, EVIOCGBIT(EV_ABS, sizeof(ev_bits)), ev_bits);
      assert(err >= 0);

      const int x_key = ABS_MT_POSITION_X / 8;
      const int y_key = ABS_MT_POSITION_Y / 8;
      if ((ev_bits[x_key] & (ABS_MT_POSITION_X - x_key)) &&
          (ev_bits[y_key] & (ABS_MT_POSITION_Y - y_key))) {
        ret = fd;
        break;
      }
    }

    
    close(fd);
  }
  closedir(dir);

  return ret;
}

void touch_init(TouchState *s) {
  s->fd = find_dev();
  assert(s->fd >= 0);
}

int touch_poll(TouchState *s, int* out_x, int* out_y, int timeout) {
  assert(out_x && out_y);
  bool up = false;
  while (true) {
    struct pollfd polls[] = {{
      .fd = s->fd,
      .events = POLLIN,
    }};
    int err = poll(polls, 1, timeout);
    if (err < 0) {
      return -1;
    }
    if (!(polls[0].revents & POLLIN)) {
      break;
    }

    struct input_event event;
    err = read(polls[0].fd, &event, sizeof(event));
    if (err < sizeof(event)) {
      return -1;
    }
    FILE *fp;
    FILE *fp2;
    char str[1000];
    char str2[5];
    char* filename = "/proc/cmdline";
    char* filename2 = "/VERSION";

    fp = fopen(filename, "r");
    fp2 = fopen(filename2, "r");
    if (fp == NULL){
      printf("Could not open file %s",filename);
      return 0;
    }
    if (fp2 == NULL){
      printf("Could not open file %s", filename2);
      return 0;
    }
    fgets(str, 1000, fp);
    fgets(str2, 5, fp2);
    fclose(fp);
    fclose(fp2);
    if (strstr(str, "letv") != NULL || strstr(str2, "6") != NULL){
      switch (event.type) { 
      case EV_ABS:
        if (event.code == ABS_MT_POSITION_X) {
          s->last_x = event.value;
        } else if (event.code == ABS_MT_POSITION_Y) {
          s->last_y = event.value;
        }
        break;
      case EV_KEY:
        if (event.code == BTN_TOOL_FINGER && event.value == 0) {
          // finger up
          up = true;
        }
        break;
      default:
        break;
      }
    } else {
      switch (event.type) { 
      case EV_ABS:
        if (event.code == ABS_MT_POSITION_X) {
          s->last_x = event.value;
        } else if (event.code == ABS_MT_POSITION_Y) {
          s->last_y = event.value;
        }
        up = true;
        break;
      default:
        break;
      }
    }
    
  }
  if (up) {
    // adjust for flippening
    *out_x = s->last_y;
    *out_y = 1080 - s->last_x;
  }
  return up;
}

