#include <time.h>

#define CAPTURE_STATE_NONE 0
#define CAPTURE_STATE_CAPTURING 1
#define CAPTURE_STATE_NOT_CAPTURING 2
#define RECORD_INTERVAL 60 // Time in seconds to rotate recordings
#define RECORD_FILES 3 // Number of files to create before looping over

int captureState = CAPTURE_STATE_NOT_CAPTURING;
int captureNum = 1;
int start_time = 0; 
int elapsed_time = 0; // Time of current recording

//TBD - need to implement locking current video
bool lock_current_video = false; // If true save the current video before rotating

void stop_capture() {
  if (captureState == CAPTURE_STATE_CAPTURING) {
    //printf("Stop capturing screen\n");
    system("killall -SIGINT screenrecord");
    captureState = CAPTURE_STATE_NOT_CAPTURING;
  }
}

int get_time() {
  // Get current time (in seconds)

  int iRet;
  struct timeval tv;
  int seconds = 0;

  iRet = gettimeofday(&tv,NULL);
  if (iRet == 0) {
    seconds = (int)tv.tv_sec;
  }
  return seconds;
}

void start_capture() {
  captureState = CAPTURE_STATE_CAPTURING;
  char cmd[50] = "";
  char videos_dir[50] = "/sdcard/videos";

  //////////////////////////////////
  // NOTE: make sure videos_dir folder exists on the device!
  //////////////////////////////////
  struct stat st = {0};
  if (stat(videos_dir, &st) == -1) {
    mkdir(videos_dir,0700);
  }

  snprintf(cmd,sizeof(cmd),"screenrecord %s/video%d.mp4&",videos_dir,captureNum);
  //printf("Capturing to file: %s\n",cmd);
  start_time = get_time();
  system(cmd);

  if (captureNum >= RECORD_FILES) {
    captureNum = 1;
  }
  else {
    captureNum++;
  }
}

bool screen_button_clicked(int touch_x, int touch_y) {
  if (touch_x >= 1660 && touch_x <= 1810) {
    if (touch_y >= 885 && touch_y <= 1035) {
      return true;
    }
  }
  return false;
}

void draw_date_time(UIState *s) {
  // Draw the current date/time

  int rect_w = 465;
  int rect_h = 80;
  int rect_x = (1920-rect_w)/2;
  int rect_y = (1080-rect_h-10);

  // Get local time to display
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);
  char now[50] = "";
  snprintf(now,sizeof(now),"%04d/%02d/%02d  %02d:%02d:%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);

  nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg, rect_x, rect_y, rect_w, rect_h, 15);
    nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
    nvgStrokeWidth(s->vg, 6);
    nvgStroke(s->vg);

  nvgFontSize(s->vg, 60);
    nvgFontFace(s->vg, "sans-semibold");
    nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
    nvgText(s->vg,rect_x+17,rect_y+55,now,NULL);
}

static void rotate_video() {
  // Overwrite the existing video (if needed)
  elapsed_time = 0;
  stop_capture();
  captureState = CAPTURE_STATE_CAPTURING;
  start_capture();
}

static void screen_draw_button(UIState *s, int touch_x, int touch_y) {
  // Set button to bottom left of screen
  if (s->vision_connected && s->plus_state == 0) {

    int btn_w = 150;
    int btn_h = 150;
    int btn_x = 1920 - btn_w;
    int btn_y = 1080 - btn_h;
    nvgBeginPath(s->vg);
      nvgRoundedRect(s->vg, btn_x-110, btn_y-45, btn_w, btn_h, 100);
      nvgStrokeColor(s->vg, nvgRGBA(255,255,255,80));
      nvgStrokeWidth(s->vg, 6);
      nvgStroke(s->vg);

      nvgFontSize(s->vg, 70);

      if (captureState == CAPTURE_STATE_CAPTURING) {
        NVGcolor fillColor = nvgRGBA(255,0,0,150);
        nvgFillColor(s->vg, fillColor);
        nvgFill(s->vg);
        nvgFillColor(s->vg, nvgRGBA(255,255,255,200));
      }
      else {
        nvgFillColor(s->vg, nvgRGBA(255, 255, 255, 200));
      }
      nvgText(s->vg,btn_x-88,btn_y+50,"REC",NULL);
  }

  if (captureState == CAPTURE_STATE_CAPTURING) {
    draw_date_time(s);

    elapsed_time = get_time() - start_time;

    if (elapsed_time >= RECORD_INTERVAL) {
      rotate_video();
    }
  }
}

void screen_toggle_record_state() {
  if (captureState == CAPTURE_STATE_CAPTURING) {
    stop_capture();
  }
  else {
    //captureState = CAPTURE_STATE_CAPTURING;
    start_capture();
  }
}

void screen_capture( UIState *s, int touch_x, int touch_y ) {
  screen_draw_button(s, touch_x, touch_y);
  if (screen_button_clicked(touch_x,touch_y)) {
    screen_toggle_record_state();

/*
    if (captureState == CAPTURE_STATE_CAPTURING) {
      start_capture();
    }
    else if (captureState == CAPTURE_STATE_NOT_CAPTURING) {
      stop_capture();
    }
*/
  }
  else if (!s->vision_connected) {
    // Assume car is not in drive so stop recording
    stop_capture();
  }
}
