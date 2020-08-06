#include <signal.h>

#include "common/params.h"
#include "uicommon.hpp"
#include "nanovg_gl.h"

volatile sig_atomic_t do_exit = 0;
static void set_do_exit(int sig) {
  do_exit = 1;
}

struct DriverView {
  TouchState touch;

  const int ui_viz_rx = (box_x - sbr_w + (bdr_s * 2));
  const int ui_viz_rw = (box_w + sbr_w - (bdr_s * 2));
  
  UIVision vision;
  NVGcontext* vg;
  int img_face;

  SubMaster* sm;

  cereal::DriverState::Reader driver_state;
  cereal::DMonitoringState::Reader dmonitoring_state;

  DriverView() {
    signal(SIGINT, (sighandler_t)set_do_exit);
    signal(SIGTERM, (sighandler_t)set_do_exit);

    touch_init(&touch);

    vision.init(true);
#ifdef QCOM
    vg = nvgCreate(0);
#else
    vg = nvgCreate(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
#endif
    assert(vg);

    img_face = nvgCreateImage(vg, "../assets/img_driver_face.png", 1);
    assert(img_face != 0);

    sm = new SubMaster({"driverState", "dMonitoringState"});
  }

  ~DriverView() {
    delete sm;
    nvgDelete(vg);
  }

  void draw() {
    const int ff_xoffset = 32;
    const int frame_x = ui_viz_rx;
    const int frame_w = ui_viz_rw;
    const int valid_frame_w = 4 * box_h / 3;
    const int valid_frame_x = frame_x + (frame_w - valid_frame_w) / 2 + ff_xoffset;

    bool is_rhd = dmonitoring_state.getIsRHD();
    // blackout
    NVGpaint gradient = nvgLinearGradient(vg, is_rhd ? valid_frame_x : (valid_frame_x + valid_frame_w),
                                          box_y,
                                          is_rhd ? (valid_frame_w - box_h / 2) : (valid_frame_x + box_h / 2), box_y,
                                          nvgRGBA(0, 0, 0, 255), nvgRGBA(0, 0, 0, 0));
    ui_draw_rect(vg, is_rhd ? valid_frame_x : (valid_frame_x + box_h / 2), box_y, valid_frame_w - box_h / 2, box_h, gradient);
    ui_draw_rect(vg, is_rhd ? valid_frame_x : valid_frame_x + box_h / 2, box_y, valid_frame_w - box_h / 2, box_h, nvgRGBA(0, 0, 0, 144));

    // borders
    ui_draw_rect(vg, frame_x, box_y, valid_frame_x - frame_x, box_h, nvgRGBA(23, 51, 73, 255));
    ui_draw_rect(vg, valid_frame_x + valid_frame_w, box_y, frame_w - valid_frame_w - (valid_frame_x - frame_x), box_h, nvgRGBA(23, 51, 73, 255));

    // draw face box
    if (dmonitoring_state.getFaceDetected()) {
      auto fxy_list = driver_state.getFacePosition();
      const float face_x = fxy_list[0];
      const float face_y = fxy_list[1];
      float fbox_x;
      float fbox_y = box_y + (face_y + 0.5) * box_h - 0.5 * 0.6 * box_h / 2;
      ;
      if (!is_rhd) {
        fbox_x = valid_frame_x + (1 - (face_x + 0.5)) * (box_h / 2) - 0.5 * 0.6 * box_h / 2;
      } else {
        fbox_x = valid_frame_x + valid_frame_w - box_h / 2 + (face_x + 0.5) * (box_h / 2) - 0.5 * 0.6 * box_h / 2;
      }

      if (std::abs(face_x) <= 0.35 && std::abs(face_y) <= 0.4) {
        ui_draw_rect(vg, fbox_x, fbox_y, 0.6 * box_h / 2, 0.6 * box_h / 2,
                     nvgRGBAf(1.0, 1.0, 1.0, 0.8 - ((std::abs(face_x) > std::abs(face_y) ? std::abs(face_x) : std::abs(face_y))) * 0.6 / 0.375),
                     35, 10);
      } else {
        ui_draw_rect(vg, fbox_x, fbox_y, 0.6 * box_h / 2, 0.6 * box_h / 2, nvgRGBAf(1.0, 1.0, 1.0, 0.2), 35, 10);
      }
    }

    // draw face icon
    const int face_size = 85;
    const int x = (valid_frame_x + face_size + (bdr_s * 2)) + (is_rhd ? valid_frame_w - box_h / 2 : 0);
    const int y = (box_y + box_h - face_size - bdr_s - (bdr_s * 1.5));
    ui_draw_circle_image(vg, x, y, face_size, img_face, dmonitoring_state.getFaceDetected());
  }

  void run() {
    while (!do_exit) {
      int touch_x = -1, touch_y = -1;
      int touched = touch_poll(&touch, &touch_x, &touch_y, 0);
      if (touched == 1) {
        write_db_value("IsDriverViewEnabled", "0", 1);
        break;
      }

      if (sm->update(0) > 0) {
        if (sm->updated("driverState")) {
          driver_state = (*sm)["driverState"].getDriverState();
        }
        if (sm->updated("dMonitoringState")) {
          dmonitoring_state = (*sm)["dMonitoringState"].getDMonitoringState();
        }
      }

      vision.update(&do_exit);

      glClearColor(0, 0, 0, 1.0);
      glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glViewport(0, 0, vision.fb_w, vision.fb_h);
      nvgBeginFrame(vg, vision.fb_w, vision.fb_h, 1.0f);

      glEnable(GL_SCISSOR_TEST);
      glViewport(ui_viz_rx, vision.fb_h - (box_y + box_h), viz_w, box_h);
      glScissor(ui_viz_rx, vision.fb_h - (box_y + box_h), ui_viz_rw, box_h);
      vision.draw();
      glDisable(GL_SCISSOR_TEST);

      glViewport(0, 0, vision.fb_w, vision.fb_h);
      draw();

      nvgEndFrame(vg);
      glDisable(GL_BLEND);

      vision.swap();
    }
  }
};

int main(int argc, char* argv[]) {
  DriverView driver_view;
  driver_view.run();
  return 0;
}
