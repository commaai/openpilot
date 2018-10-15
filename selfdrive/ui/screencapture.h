#define CAPTURE_STATE_NONE 0
#define CAPTURE_STATE_CAPTURING 1
#define CAPTURE_STATE_NOT_CAPTURING 2

int captureState = CAPTURE_STATE_NOT_CAPTURING;
int captureNum = 0;

void stop_capture() {
  printf("Stop capturing screen\n");
  system("killall -SIGINT screenrecord");
  captureState = CAPTURE_STATE_NOT_CAPTURING;
}

bool screen_button_clicked(int touch_x, int touch_y) {
  if (touch_x >= 1660 && touch_x <= 1810) {
    if (touch_y >= 885 && touch_y <= 1035) {
      return true;
    }
  }
  return false;
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

    //printf("touched: %d, %d\n",touch_x,touch_y);
  }
}

void screen_toggle_record_state() {
  if (captureState == CAPTURE_STATE_CAPTURING) {
    captureState = CAPTURE_STATE_NOT_CAPTURING;
  }
  else {
    captureState = CAPTURE_STATE_CAPTURING;
  }
}

void screen_capture( UIState *s, int touch_x, int touch_y ) {
  screen_draw_button(s, touch_x, touch_y);
  if (screen_button_clicked(touch_x,touch_y)) {
    screen_toggle_record_state();

    if (captureState == CAPTURE_STATE_CAPTURING) {
      char cmd[50] = "";
      captureNum++;

      //////////////////////////////////
      // NOTE: make sure /sdcard/videos/ folder exists on the device!
      //////////////////////////////////
      snprintf(cmd,sizeof(cmd),"screenrecord /sdcard/videos/video%d.mp4&",captureNum);
      printf("Capturing to file: %s\n",cmd);
      system(cmd);
    }
    else if (captureState == CAPTURE_STATE_NOT_CAPTURING) {
      stop_capture();
    }
  }
  else if (!s->vision_connected) {
    // Assume car is not in drive so stop recording
    stop_capture();
  }
}
