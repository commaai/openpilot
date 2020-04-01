#include <assert.h>
#include "ui.hpp"

#include "common/util.h"

#define NANOVG_GLES3_IMPLEMENTATION

#include "nanovg_gl.h"
#include "nanovg_gl_utils.h"

extern "C"{
#include "common/glutil.h"
}

// TODO: this is also hardcoded in common/transformations/camera.py
const mat3 intrinsic_matrix = (mat3){{
  910., 0., 582.,
  0., 910., 437.,
  0.,   0.,   1.
}};

const uint8_t alert_colors[][4] = {
  [STATUS_STOPPED] = {0x07, 0x23, 0x39, 0xf1},
  [STATUS_DISENGAGED] = {0x17, 0x33, 0x49, 0xc8},
  [STATUS_ENGAGED] = {0x17, 0x86, 0x44, 0xf1},
  [STATUS_WARNING] = {0xDA, 0x6F, 0x25, 0xf1},
  [STATUS_ALERT] = {0xC9, 0x22, 0x31, 0xf1},
};

const int alert_sizes[] = {
  [ALERTSIZE_NONE] = 0,
  [ALERTSIZE_SMALL] = 241,
  [ALERTSIZE_MID] = 390,
  [ALERTSIZE_FULL] = vwp_h,
};

// Projects a point in car to space to the corresponding point in full frame
// image space.
vec3 car_space_to_full_frame(const UIState *s, vec4 car_space_projective) {
  const UIScene *scene = &s->scene;

  // We'll call the car space point p.
  // First project into normalized image coordinates with the extrinsics matrix.
  const vec4 Ep4 = matvecmul(scene->extrinsic_matrix, car_space_projective);

  // The last entry is zero because of how we store E (to use matvecmul).
  const vec3 Ep = {{Ep4.v[0], Ep4.v[1], Ep4.v[2]}};
  const vec3 KEp = matvecmul3(intrinsic_matrix, Ep);

  // Project.
  const vec3 p_image = {{KEp.v[0] / KEp.v[2], KEp.v[1] / KEp.v[2], 1.}};
  return p_image;
}

// Calculate an interpolation between two numbers at a specific increment
static float lerp(float v0, float v1, float t) {
  return (1 - t) * v0 + t * v1;
}

static void draw_chevron(UIState *s, float x_in, float y_in, float sz,
                          NVGcolor fillColor, NVGcolor glowColor) {
  const UIScene *scene = &s->scene;

  nvgSave(s->vg);

  nvgTranslate(s->vg, 240.0f, 0.0);
  nvgTranslate(s->vg, -1440.0f / 2, -1080.0f / 2);
  nvgScale(s->vg, 2.0, 2.0);
  nvgScale(s->vg, 1440.0f / s->rgb_width, 1080.0f / s->rgb_height);

  const vec4 p_car_space = (vec4){{x_in, y_in, 0., 1.}};
  const vec3 p_full_frame = car_space_to_full_frame(s, p_car_space);

  sz *= 30;
  sz /= (x_in / 3 + 30);
  if (sz > 30) sz = 30;
  if (sz < 15) sz = 15;

  float x = p_full_frame.v[0];
  float y = p_full_frame.v[1];

  // glow
  nvgBeginPath(s->vg);
  float g_xo = sz/5;
  float g_yo = sz/10;
  if (x >= 0 && y >= 0.) {
    nvgMoveTo(s->vg, x+(sz*1.35)+g_xo, y+sz+g_yo);
    nvgLineTo(s->vg, x, y-g_xo);
    nvgLineTo(s->vg, x-(sz*1.35)-g_xo, y+sz+g_yo);
    nvgLineTo(s->vg, x+(sz*1.35)+g_xo, y+sz+g_yo);
    nvgClosePath(s->vg);
  }
  nvgFillColor(s->vg, glowColor);
  nvgFill(s->vg);

  // chevron
  nvgBeginPath(s->vg);
  if (x >= 0 && y >= 0.) {
    nvgMoveTo(s->vg, x+(sz*1.25), y+sz);
    nvgLineTo(s->vg, x, y);
    nvgLineTo(s->vg, x-(sz*1.25), y+sz);
    nvgLineTo(s->vg, x+(sz*1.25), y+sz);
    nvgClosePath(s->vg);
  }
  nvgFillColor(s->vg, fillColor);
  nvgFill(s->vg);

  nvgRestore(s->vg);
}

static void draw_lead(UIState *s, float d_rel, float v_rel, float y_rel){
    // Draw lead car indicator
    float fillAlpha = 0;
    float speedBuff = 10.;
    float leadBuff = 40.;
    if (d_rel < leadBuff) {
      fillAlpha = 255*(1.0-(d_rel/leadBuff));
      if (v_rel < 0) {
        fillAlpha += 255*(-1*(v_rel/speedBuff));
      }
      fillAlpha = (int)(fmin(fillAlpha, 255));
    }
    draw_chevron(s, d_rel, y_rel, 25,
                 nvgRGBA(201, 34, 49, fillAlpha), COLOR_YELLOW);
}

static void ui_draw_lane_line(UIState *s, const model_path_vertices_data *pvd, NVGcolor color) {
  nvgSave(s->vg);
  nvgTranslate(s->vg, 240.0f, 0.0); // rgb-box space
  nvgTranslate(s->vg, -1440.0f / 2, -1080.0f / 2); // zoom 2x
  nvgScale(s->vg, 2.0, 2.0);
  nvgScale(s->vg, 1440.0f / s->rgb_width, 1080.0f / s->rgb_height);
  nvgBeginPath(s->vg);

  bool started = false;
  for (int i=0; i<pvd->cnt; i++) {
    if (pvd->v[i].x < 0 || pvd->v[i].y < 0.) {
      continue;
    }
    if (!started) {
      nvgMoveTo(s->vg, pvd->v[i].x, pvd->v[i].y);
      started = true;
    } else {
      nvgLineTo(s->vg, pvd->v[i].x, pvd->v[i].y);
    }
  }

  nvgClosePath(s->vg);
  nvgFillColor(s->vg, color);
  nvgFill(s->vg);
  nvgRestore(s->vg);
}

static void update_track_data(UIState *s, bool is_mpc, track_vertices_data *pvd) {
  const UIScene *scene = &s->scene;
  const PathData path = scene->model.path;
  const float *mpc_x_coords = &scene->mpc_x[0];
  const float *mpc_y_coords = &scene->mpc_y[0];

  bool started = false;
  float off = is_mpc?0.3:0.5;
  float lead_d = scene->lead_d_rel*2.;
  float path_height = is_mpc?(lead_d>5.)?fmin(lead_d, 25.)-fmin(lead_d*0.35, 10.):20.
                            :(lead_d>0.)?fmin(lead_d, 50.)-fmin(lead_d*0.35, 10.):49.;
  pvd->cnt = 0;
  // left side up
  for (int i=0; i<=path_height; i++) {
    float px, py, mpx;
    if (is_mpc) {
      mpx = i==0?0.0:mpc_x_coords[i];
      px = lerp(mpx+1.0, mpx, i/100.0);
      py = mpc_y_coords[i] - off;
    } else {
      px = lerp(i+1.0, i, i/100.0);
      py = path.points[i] - off;
    }

    vec4 p_car_space = (vec4){{px, py, 0., 1.}};
    vec3 p_full_frame = car_space_to_full_frame(s, p_car_space);
    if (p_full_frame.v[0] < 0. || p_full_frame.v[1] < 0.) {
      continue;
    }
    pvd->v[pvd->cnt].x = p_full_frame.v[0];
    pvd->v[pvd->cnt].y = p_full_frame.v[1];
    pvd->cnt += 1;
  }

  // right side down
  for (int i=path_height; i>=0; i--) {
    float px, py, mpx;
    if (is_mpc) {
      mpx = i==0?0.0:mpc_x_coords[i];
      px = lerp(mpx+1.0, mpx, i/100.0);
      py = mpc_y_coords[i] + off;
    } else {
      px = lerp(i+1.0, i, i/100.0);
      py = path.points[i] + off;
    }

    vec4 p_car_space = (vec4){{px, py, 0., 1.}};
    vec3 p_full_frame = car_space_to_full_frame(s, p_car_space);
    pvd->v[pvd->cnt].x = p_full_frame.v[0];
    pvd->v[pvd->cnt].y = p_full_frame.v[1];
    pvd->cnt += 1;
  }
}

static void update_all_track_data(UIState *s) {
  const UIScene *scene = &s->scene;
  // Draw vision path
  update_track_data(s, false, &s->track_vertices[0]);

  if (scene->engaged) {
    // Draw MPC path when engaged
    update_track_data(s, true, &s->track_vertices[1]);
  }
}


static void ui_draw_track(UIState *s, bool is_mpc, track_vertices_data *pvd) {
  const UIScene *scene = &s->scene;
  const PathData path = scene->model.path;
  const float *mpc_x_coords = &scene->mpc_x[0];
  const float *mpc_y_coords = &scene->mpc_y[0];

  nvgSave(s->vg);
  nvgTranslate(s->vg, 240.0f, 0.0); // rgb-box space
  nvgTranslate(s->vg, -1440.0f / 2, -1080.0f / 2); // zoom 2x
  nvgScale(s->vg, 2.0, 2.0);
  nvgScale(s->vg, 1440.0f / s->rgb_width, 1080.0f / s->rgb_height);
  nvgBeginPath(s->vg);

  bool started = false;
  float off = is_mpc?0.3:0.5;
  float lead_d = scene->lead_d_rel*2.;
  float path_height = is_mpc?(lead_d>5.)?fmin(lead_d, 25.)-fmin(lead_d*0.35, 10.):20.
                            :(lead_d>0.)?fmin(lead_d, 50.)-fmin(lead_d*0.35, 10.):49.;
  int vi = 0;
  for(int i = 0;i < pvd->cnt;i++) {
    if (pvd->v[i].x < 0 || pvd->v[i].y < 0) {
      continue;
    }

    if (!started) {
      nvgMoveTo(s->vg, pvd->v[i].x, pvd->v[i].y);
      started = true;
    } else {
      nvgLineTo(s->vg, pvd->v[i].x, pvd->v[i].y);
    }
  }

  nvgClosePath(s->vg);

  NVGpaint track_bg;
  if (is_mpc) {
    // Draw colored MPC track
    const uint8_t *clr = bg_colors[s->status];
    track_bg = nvgLinearGradient(s->vg, vwp_w, vwp_h, vwp_w, vwp_h*.4,
      nvgRGBA(clr[0], clr[1], clr[2], 255), nvgRGBA(clr[0], clr[1], clr[2], 255/2));
  } else {
    // Draw white vision track
    track_bg = nvgLinearGradient(s->vg, vwp_w, vwp_h, vwp_w, vwp_h*.4,
      COLOR_WHITE, COLOR_WHITE_ALPHA(0));
  }

  nvgFillPaint(s->vg, track_bg);
  nvgFill(s->vg);
  nvgRestore(s->vg);
}

static void draw_steering(UIState *s, float curvature) {
  float points[50];
  for (int i = 0; i < 50; i++) {
    float y_actual = i * tan(asin(clamp(i * curvature, -0.999, 0.999)) / 2.);
    points[i] = y_actual;
  }

  // ui_draw_lane_edge(s, points, 0.0, nvgRGBA(0, 0, 255, 128), 5);
}

static void draw_frame(UIState *s) {
  const UIScene *scene = &s->scene;

  float x1, x2, y1, y2;
  if (s->scene.frontview) {
    glBindVertexArray(s->frame_vao[1]);
  } else {
    glBindVertexArray(s->frame_vao[0]);
  }

  mat4 *out_mat;
  if (s->scene.frontview || s->scene.fullview) {
    out_mat = &s->front_frame_mat;
  } else {
    out_mat = &s->rear_frame_mat;
  }
  glActiveTexture(GL_TEXTURE0);
  if (s->scene.frontview && s->cur_vision_front_idx >= 0) {
    glBindTexture(GL_TEXTURE_2D, s->frame_front_texs[s->cur_vision_front_idx]);
  } else if (!scene->frontview && s->cur_vision_idx >= 0) {
    glBindTexture(GL_TEXTURE_2D, s->frame_texs[s->cur_vision_idx]);
    #ifndef QCOM
      // TODO: a better way to do this?
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1164, 874, 0, GL_RGB, GL_UNSIGNED_BYTE, s->priv_hnds[s->cur_vision_idx]);
    #endif
  }

  glUseProgram(s->frame_program);
  glUniform1i(s->frame_texture_loc, 0);
  glUniformMatrix4fv(s->frame_transform_loc, 1, GL_TRUE, out_mat->v);

  assert(glGetError() == GL_NO_ERROR);
  glEnableVertexAttribArray(0);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, (const void*)0);
  glDisableVertexAttribArray(0);
  glBindVertexArray(0);
}

static inline bool valid_frame_pt(UIState *s, float x, float y) {
  return x >= 0 && x <= s->rgb_width && y >= 0 && y <= s->rgb_height;

}
static void update_lane_line_data(UIState *s, const float *points, float off, bool is_ghost, model_path_vertices_data *pvd) {
  pvd->cnt = 0;
  for (int i = 0; i < MODEL_PATH_MAX_VERTICES_CNT / 2; i++) {
    float px = (float)i;
    float py = points[i] - off;
    const vec4 p_car_space = (vec4){{px, py, 0., 1.}};
    const vec3 p_full_frame = car_space_to_full_frame(s, p_car_space);
    if(!valid_frame_pt(s, p_full_frame.v[0], p_full_frame.v[1]))
      continue;
    pvd->v[pvd->cnt].x = p_full_frame.v[0];
    pvd->v[pvd->cnt].y = p_full_frame.v[1];
    pvd->cnt += 1;
  }
  for (int i = MODEL_PATH_MAX_VERTICES_CNT / 2; i > 0; i--) {
    float px = (float)i;
    float py = is_ghost?(points[i]-off):(points[i]+off);
    const vec4 p_car_space = (vec4){{px, py, 0., 1.}};
    const vec3 p_full_frame = car_space_to_full_frame(s, p_car_space);
    if(!valid_frame_pt(s, p_full_frame.v[0], p_full_frame.v[1]))
      continue;
    pvd->v[pvd->cnt].x = p_full_frame.v[0];
    pvd->v[pvd->cnt].y = p_full_frame.v[1];
    pvd->cnt += 1;
  }
}

static void update_all_lane_lines_data(UIState *s, const PathData path, model_path_vertices_data *pstart) {
  update_lane_line_data(s, path.points, 0.025*path.prob, false, pstart);
  float var = fmin(path.std, 0.7);
  update_lane_line_data(s, path.points, -var, true, pstart + 1);
  update_lane_line_data(s, path.points, var, true, pstart + 2);
}

static void ui_draw_lane(UIState *s, const PathData *path, model_path_vertices_data *pstart, NVGcolor color) {
  ui_draw_lane_line(s, pstart, color);
  float var = fmin(path->std, 0.7);
  color.a /= 4;
  ui_draw_lane_line(s, pstart + 1, color);
  ui_draw_lane_line(s, pstart + 2, color);
}

static void ui_draw_vision_lanes(UIState *s) {
  const UIScene *scene = &s->scene;
  model_path_vertices_data *pvd = &s->model_path_vertices[0];
  if(s->model_changed) {
    update_all_lane_lines_data(s, scene->model.left_lane, pvd);
    update_all_lane_lines_data(s, scene->model.right_lane, pvd + MODEL_LANE_PATH_CNT);
    s->model_changed = false;
  }
  // Draw left lane edge
  ui_draw_lane(
      s, &scene->model.left_lane,
      pvd,
      nvgRGBAf(1.0, 1.0, 1.0, scene->model.left_lane.prob));

  // Draw right lane edge
  ui_draw_lane(
      s, &scene->model.right_lane,
      pvd + MODEL_LANE_PATH_CNT,
      nvgRGBAf(1.0, 1.0, 1.0, scene->model.right_lane.prob));

  if(s->livempc_or_radarstate_changed) {
    update_all_track_data(s);
    s->livempc_or_radarstate_changed = false;
  }
  // Draw vision path
  ui_draw_track(s, false, &s->track_vertices[0]);
  if (scene->engaged) {
    // Draw MPC path when engaged
    ui_draw_track(s, true, &s->track_vertices[1]);
  }
}

// Draw all world space objects.
static void ui_draw_world(UIState *s) {
  const UIScene *scene = &s->scene;
  if (!scene->world_objects_visible) {
    return;
  }

  // Draw lane edges and vision/mpc tracks
  ui_draw_vision_lanes(s);

  if (scene->lead_status) {
    draw_lead(s, scene->lead_d_rel, scene->lead_v_rel, scene->lead_y_rel);
  }
  if ((scene->lead_status2) && (fabs(scene->lead_d_rel - scene->lead_d_rel2) > 3.0)) {
    draw_lead(s, scene->lead_d_rel2, scene->lead_v_rel2, scene->lead_y_rel2);
  }
}

static void ui_draw_vision_maxspeed(UIState *s) {
  /*if (!s->longitudinal_control){
    return;
  }*/

  const UIScene *scene = &s->scene;
  int ui_viz_rx = scene->ui_viz_rx;
  int ui_viz_rw = scene->ui_viz_rw;

  char maxspeed_str[32];
  float maxspeed = s->scene.v_cruise;
  int maxspeed_calc = maxspeed * 0.6225 + 0.5;
  float speedlimit = s->scene.speedlimit;
  int speedlim_calc = speedlimit * 2.2369363 + 0.5;
  int speed_lim_off = s->speed_lim_off * 2.2369363 + 0.5;
  if (s->is_metric) {
    maxspeed_calc = maxspeed + 0.5;
    speedlim_calc = speedlimit * 3.6 + 0.5;
    speed_lim_off = s->speed_lim_off * 3.6 + 0.5;
  }

  bool is_cruise_set = (maxspeed != 0 && maxspeed != SET_SPEED_NA);
  bool is_speedlim_valid = s->scene.speedlimit_valid;
  bool is_set_over_limit = is_speedlim_valid && s->scene.engaged &&
                       is_cruise_set && maxspeed_calc > (speedlim_calc + speed_lim_off);

  int viz_maxspeed_w = 184;
  int viz_maxspeed_h = 202;
  int viz_maxspeed_x = (ui_viz_rx + (bdr_s*2));
  int viz_maxspeed_y = (box_y + (bdr_s*1.5));
  int viz_maxspeed_xo = 180;

#ifdef SHOW_SPEEDLIMIT
  viz_maxspeed_w += viz_maxspeed_xo;
  viz_maxspeed_x += viz_maxspeed_w - (viz_maxspeed_xo * 2);
#else
  viz_maxspeed_xo = 0;
#endif

  // Draw Background
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, viz_maxspeed_x, viz_maxspeed_y, viz_maxspeed_w, viz_maxspeed_h, 30);
  if (is_set_over_limit) {
    nvgFillColor(s->vg, nvgRGBA(218, 111, 37, 180));
  } else {
    nvgFillColor(s->vg, COLOR_BLACK_ALPHA(100));
  }
  nvgFill(s->vg);

  // Draw Border
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, viz_maxspeed_x, viz_maxspeed_y, viz_maxspeed_w, viz_maxspeed_h, 20);
  if (is_set_over_limit) {
    nvgStrokeColor(s->vg, COLOR_OCHRE);
  } else if (is_speedlim_valid && !s->is_ego_over_limit) {
    nvgStrokeColor(s->vg, COLOR_WHITE);
  } else if (is_speedlim_valid && s->is_ego_over_limit) {
    nvgStrokeColor(s->vg, COLOR_WHITE_ALPHA(20));
  } else {
    nvgStrokeColor(s->vg, COLOR_WHITE_ALPHA(100));
  }
  nvgStrokeWidth(s->vg, 10);
  nvgStroke(s->vg);

  // Draw "MAX" Text
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);
  nvgFontFaceId(s->vg,  s->font_sans_regular);
  nvgFontSize(s->vg, 26*2.5);
  if (is_cruise_set) {
    nvgFillColor(s->vg, COLOR_WHITE_ALPHA(200));
  } else {
    nvgFillColor(s->vg, COLOR_WHITE_ALPHA(100));
  }
  nvgText(s->vg, viz_maxspeed_x+(viz_maxspeed_xo/2)+(viz_maxspeed_w/2), 148, "MAX", NULL);

  // Draw Speed Text
  nvgFontFaceId(s->vg, s->font_sans_bold);
  nvgFontSize(s->vg, 48*2.5);
  if (is_cruise_set) {
    snprintf(maxspeed_str, sizeof(maxspeed_str), "%d", maxspeed_calc);
    nvgFillColor(s->vg, COLOR_WHITE);
    nvgText(s->vg, viz_maxspeed_x+(viz_maxspeed_xo/2)+(viz_maxspeed_w/2), 242, maxspeed_str, NULL);
  } else {
    nvgFontFaceId(s->vg, s->font_sans_semibold);
    nvgFontSize(s->vg, 42*2.5);
    nvgFillColor(s->vg, COLOR_WHITE_ALPHA(100));
    nvgText(s->vg, viz_maxspeed_x+(viz_maxspeed_xo/2)+(viz_maxspeed_w/2), 242, "N/A", NULL);
  }

}

static void ui_draw_vision_speedlimit(UIState *s) {
  const UIScene *scene = &s->scene;
  int ui_viz_rx = scene->ui_viz_rx;
  int ui_viz_rw = scene->ui_viz_rw;

  char speedlim_str[32];
  float speedlimit = s->scene.speedlimit;
  int speedlim_calc = speedlimit * 2.2369363 + 0.5;
  if (s->is_metric) {
    speedlim_calc = speedlimit * 3.6 + 0.5;
  }

  bool is_speedlim_valid = s->scene.speedlimit_valid;
  float hysteresis_offset = 0.5;
  if (s->is_ego_over_limit) {
    hysteresis_offset = 0.0;
  }
  s->is_ego_over_limit = is_speedlim_valid && s->scene.v_ego > (speedlimit + s->speed_lim_off + hysteresis_offset);

  int viz_speedlim_w = 180;
  int viz_speedlim_h = 202;
  int viz_speedlim_x = (ui_viz_rx + (bdr_s*2));
  int viz_speedlim_y = (box_y + (bdr_s*1.5));
  if (!is_speedlim_valid) {
    viz_speedlim_w -= 5;
    viz_speedlim_h -= 10;
    viz_speedlim_x += 9;
    viz_speedlim_y += 5;
  }
  int viz_speedlim_bdr = is_speedlim_valid ? 30 : 15;

  // Draw Background
  nvgBeginPath(s->vg);
  nvgRoundedRect(s->vg, viz_speedlim_x, viz_speedlim_y, viz_speedlim_w, viz_speedlim_h, viz_speedlim_bdr);
  if (is_speedlim_valid && s->is_ego_over_limit) {
    nvgFillColor(s->vg, nvgRGBA(218, 111, 37, 180));
  } else if (is_speedlim_valid) {
    nvgFillColor(s->vg, COLOR_WHITE);
  } else {
    nvgFillColor(s->vg, COLOR_WHITE_ALPHA(100));
  }
  nvgFill(s->vg);

  // Draw Border
  if (is_speedlim_valid) {
    nvgStrokeWidth(s->vg, 10);
    nvgStroke(s->vg);
    nvgBeginPath(s->vg);
    nvgRoundedRect(s->vg, viz_speedlim_x, viz_speedlim_y, viz_speedlim_w, viz_speedlim_h, 20);
    if (s->is_ego_over_limit) {
      nvgStrokeColor(s->vg, COLOR_OCHRE);
    } else if (is_speedlim_valid) {
      nvgStrokeColor(s->vg, COLOR_WHITE);
    }
  }

  // Draw "Speed Limit" Text
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);
  nvgFontFaceId(s->vg, s->font_sans_semibold);
  nvgFontSize(s->vg, 50);
  nvgFillColor(s->vg, COLOR_BLACK);
  if (is_speedlim_valid && s->is_ego_over_limit) {
    nvgFillColor(s->vg, COLOR_WHITE);
  }
  nvgText(s->vg, viz_speedlim_x+viz_speedlim_w/2 + (is_speedlim_valid ? 6 : 0), viz_speedlim_y + (is_speedlim_valid ? 50 : 45), "SMART", NULL);
  nvgText(s->vg, viz_speedlim_x+viz_speedlim_w/2 + (is_speedlim_valid ? 6 : 0), viz_speedlim_y + (is_speedlim_valid ? 90 : 85), "SPEED", NULL);

  // Draw Speed Text
  nvgFontFaceId(s->vg, s->font_sans_bold);
  nvgFontSize(s->vg, 48*2.5);
  if (s->is_ego_over_limit) {
    nvgFillColor(s->vg, COLOR_WHITE);
  } else {
    nvgFillColor(s->vg, COLOR_BLACK);
  }
  if (is_speedlim_valid) {
    snprintf(speedlim_str, sizeof(speedlim_str), "%d", speedlim_calc);
    nvgText(s->vg, viz_speedlim_x+viz_speedlim_w/2, viz_speedlim_y + (is_speedlim_valid ? 170 : 165), speedlim_str, NULL);
  } else {
    nvgFontFaceId(s->vg, s->font_sans_semibold);
    nvgFontSize(s->vg, 42*2.5);
    nvgText(s->vg, viz_speedlim_x+viz_speedlim_w/2, viz_speedlim_y + (is_speedlim_valid ? 170 : 165), "N/A", NULL);
  }
}

static void ui_draw_vision_speed(UIState *s) {
  const UIScene *scene = &s->scene;
  int ui_viz_rx = scene->ui_viz_rx;
  int ui_viz_rw = scene->ui_viz_rw;
  float speed = s->scene.v_ego;

  const int viz_speed_w = 280;
  const int viz_speed_x = ui_viz_rx+((ui_viz_rw/2)-(viz_speed_w/2));
  char speed_str[32];

  nvgBeginPath(s->vg);
  nvgRect(s->vg, viz_speed_x, box_y, viz_speed_w, header_h);
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);

  if (s->is_metric) {
    snprintf(speed_str, sizeof(speed_str), "%d", (int)(speed * 3.6 + 0.5));
  } else {
    snprintf(speed_str, sizeof(speed_str), "%d", (int)(speed * 2.2369363 + 0.5));
  }
  nvgFontFaceId(s->vg, s->font_sans_bold);
  nvgFontSize(s->vg, 96*2.5);
  nvgFillColor(s->vg, COLOR_WHITE);
  nvgText(s->vg, viz_speed_x+viz_speed_w/2, 240, speed_str, NULL);

  nvgFontFaceId(s->vg,  s->font_sans_regular);
  nvgFontSize(s->vg, 36*2.5);
  nvgFillColor(s->vg, COLOR_WHITE_ALPHA(200));

  if (s->is_metric) {
    nvgText(s->vg, viz_speed_x+viz_speed_w/2, 320, "kph", NULL);
  } else {
    nvgText(s->vg, viz_speed_x+viz_speed_w/2, 320, "mph", NULL);
  }
}

static void ui_draw_vision_event(UIState *s) {
  const UIScene *scene = &s->scene;
  const int ui_viz_rx = scene->ui_viz_rx;
  const int ui_viz_rw = scene->ui_viz_rw;
  const int viz_event_w = 220;
  const int viz_event_x = ((ui_viz_rx + ui_viz_rw) - (viz_event_w + (bdr_s*2)));
  const int viz_event_y = (box_y + (bdr_s*1.5));
  const int viz_event_h = (header_h - (bdr_s*1.5));
  if (s->scene.decel_for_model && s->scene.engaged) {
    // draw winding road sign
    const int img_turn_size = 160*1.5;
    const int img_turn_x = viz_event_x-(img_turn_size/4);
    const int img_turn_y = viz_event_y+bdr_s-25;
    float img_turn_alpha = 1.0f;
    nvgBeginPath(s->vg);
    NVGpaint imgPaint = nvgImagePattern(s->vg, img_turn_x, img_turn_y,
      img_turn_size, img_turn_size, 0, s->img_turn, img_turn_alpha);
    nvgRect(s->vg, img_turn_x, img_turn_y, img_turn_size, img_turn_size);
    nvgFillPaint(s->vg, imgPaint);
    nvgFill(s->vg);
  } else {
    // draw steering wheel
    const int bg_wheel_size = 96;
    const int bg_wheel_x = viz_event_x + (viz_event_w-bg_wheel_size);
    const int bg_wheel_y = viz_event_y + (bg_wheel_size/2);
    const int img_wheel_size = bg_wheel_size*1.5;
    const int img_wheel_x = bg_wheel_x-(img_wheel_size/2);
    const int img_wheel_y = bg_wheel_y-25;
    float img_wheel_alpha = 0.1f;
    bool is_engaged = (s->status == STATUS_ENGAGED);
    bool is_warning = (s->status == STATUS_WARNING);
    bool is_engageable = scene->engageable;
    if (is_engaged || is_warning || is_engageable) {
      nvgBeginPath(s->vg);
      nvgCircle(s->vg, bg_wheel_x, (bg_wheel_y + (bdr_s*1.5)), bg_wheel_size);
      if (is_engaged) {
        nvgFillColor(s->vg, nvgRGBA(23, 134, 68, 255));
      } else if (is_warning) {
        nvgFillColor(s->vg, COLOR_OCHRE);
      } else if (is_engageable) {
        nvgFillColor(s->vg, nvgRGBA(23, 51, 73, 255));
      }
      nvgFill(s->vg);
      img_wheel_alpha = 1.0f;
    }
    nvgBeginPath(s->vg);
    NVGpaint imgPaint = nvgImagePattern(s->vg, img_wheel_x, img_wheel_y,
      img_wheel_size, img_wheel_size, 0, s->img_wheel, img_wheel_alpha);
    nvgRect(s->vg, img_wheel_x, img_wheel_y, img_wheel_size, img_wheel_size);
    nvgFillPaint(s->vg, imgPaint);
    nvgFill(s->vg);
  }
}

static void ui_draw_vision_map(UIState *s) {
  const UIScene *scene = &s->scene;
  const int map_size = 96;
  const int map_x = (scene->ui_viz_rx + (map_size * 3) + (bdr_s * 3));
  const int map_y = (footer_y + ((footer_h - map_size) / 2));
  const int map_img_size = (map_size * 1.5);
  const int map_img_x = (map_x - (map_img_size / 2));
  const int map_img_y = (map_y - (map_size / 4));

  bool map_valid = s->scene.map_valid;
  float map_img_alpha = map_valid ? 1.0f : 0.15f;
  float map_bg_alpha = map_valid ? 0.3f : 0.1f;
  NVGcolor map_bg = nvgRGBA(0, 0, 0, (255 * map_bg_alpha));
  NVGpaint map_img = nvgImagePattern(s->vg, map_img_x, map_img_y,
    map_img_size, map_img_size, 0, s->img_map, map_img_alpha);

  nvgBeginPath(s->vg);
  nvgCircle(s->vg, map_x, (map_y + (bdr_s * 1.5)), map_size);
  nvgFillColor(s->vg, map_bg);
  nvgFill(s->vg);

  nvgBeginPath(s->vg);
  nvgRect(s->vg, map_img_x, map_img_y, map_img_size, map_img_size);
  nvgFillPaint(s->vg, map_img);
  nvgFill(s->vg);
}

static void ui_draw_vision_face(UIState *s) {
  const UIScene *scene = &s->scene;
  const int face_size = 96;
  const int face_x = (scene->ui_viz_rx + face_size + (bdr_s * 2));
  const int face_y = (footer_y + ((footer_h - face_size) / 2));
  const int face_img_size = (face_size * 1.5);
  const int face_img_x = (face_x - (face_img_size / 2));
  const int face_img_y = (face_y - (face_size / 4));
  float face_img_alpha = scene->monitoring_active ? 1.0f : 0.15f;
  float face_bg_alpha = scene->monitoring_active ? 0.3f : 0.1f;
  NVGcolor face_bg = nvgRGBA(0, 0, 0, (255 * face_bg_alpha));
  NVGpaint face_img = nvgImagePattern(s->vg, face_img_x, face_img_y,
    face_img_size, face_img_size, 0, s->img_face, face_img_alpha);

  nvgBeginPath(s->vg);
  nvgCircle(s->vg, face_x, (face_y + (bdr_s * 1.5)), face_size);
  nvgFillColor(s->vg, face_bg);
  nvgFill(s->vg);

  nvgBeginPath(s->vg);
  nvgRect(s->vg, face_img_x, face_img_y, face_img_size, face_img_size);
  nvgFillPaint(s->vg, face_img);
  nvgFill(s->vg);
}

static void ui_draw_vision_header(UIState *s) {
  const UIScene *scene = &s->scene;
  int ui_viz_rx = scene->ui_viz_rx;
  int ui_viz_rw = scene->ui_viz_rw;

  nvgBeginPath(s->vg);
  NVGpaint gradient = nvgLinearGradient(s->vg, ui_viz_rx,
                        (box_y+(header_h-(header_h/2.5))),
                        ui_viz_rx, box_y+header_h,
                        nvgRGBAf(0,0,0,0.45), nvgRGBAf(0,0,0,0));
  nvgFillPaint(s->vg, gradient);
  nvgRect(s->vg, ui_viz_rx, box_y, ui_viz_rw, header_h);
  nvgFill(s->vg);

  ui_draw_vision_maxspeed(s);

#ifdef SHOW_SPEEDLIMIT
  ui_draw_vision_speedlimit(s);
#endif
  ui_draw_vision_speed(s);
  ui_draw_vision_event(s);
}

static void ui_draw_vision_footer(UIState *s) {
  const UIScene *scene = &s->scene;
  int ui_viz_rx = scene->ui_viz_rx;
  int ui_viz_rw = scene->ui_viz_rw;

  nvgBeginPath(s->vg);
  nvgRect(s->vg, ui_viz_rx, footer_y, ui_viz_rw, footer_h);

  ui_draw_vision_face(s);

#ifdef SHOW_SPEEDLIMIT
  // ui_draw_vision_map(s);
#endif
}

void ui_draw_vision_alert(UIState *s, int va_size, int va_color,
                          const char* va_text1, const char* va_text2) {
  const UIScene *scene = &s->scene;
  int ui_viz_rx = scene->ui_viz_rx;
  int ui_viz_rw = scene->ui_viz_rw;
  bool hasSidebar = !s->scene.uilayout_sidebarcollapsed;
  bool mapEnabled = s->scene.uilayout_mapenabled;
  bool longAlert1 = strlen(va_text1) > 15;

  const uint8_t *color = alert_colors[va_color];
  const int alr_s = alert_sizes[va_size];
  const int alr_x = ui_viz_rx-(mapEnabled?(hasSidebar?nav_w:(nav_ww)):0)-bdr_s;
  const int alr_w = ui_viz_rw+(mapEnabled?(hasSidebar?nav_w:(nav_ww)):0)+(bdr_s*2);
  const int alr_h = alr_s+(va_size==ALERTSIZE_NONE?0:bdr_s);
  const int alr_y = vwp_h-alr_h;

  nvgBeginPath(s->vg);
  nvgRect(s->vg, alr_x, alr_y, alr_w, alr_h);
  nvgFillColor(s->vg, nvgRGBA(color[0],color[1],color[2],(color[3]*s->alert_blinking_alpha)));
  nvgFill(s->vg);

  nvgBeginPath(s->vg);
  NVGpaint gradient = nvgLinearGradient(s->vg, alr_x, alr_y, alr_x, alr_y+alr_h,
                        nvgRGBAf(0.0,0.0,0.0,0.05), nvgRGBAf(0.0,0.0,0.0,0.35));
  nvgFillPaint(s->vg, gradient);
  nvgRect(s->vg, alr_x, alr_y, alr_w, alr_h);
  nvgFill(s->vg);

  nvgFillColor(s->vg, COLOR_WHITE);
  nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);

  if (va_size == ALERTSIZE_SMALL) {
    nvgFontFaceId(s->vg, s->font_sans_semibold);
    nvgFontSize(s->vg, 40*2.5);
    nvgText(s->vg, alr_x+alr_w/2, alr_y+alr_h/2+15, va_text1, NULL);
  } else if (va_size== ALERTSIZE_MID) {
    nvgFontFaceId(s->vg, s->font_sans_bold);
    nvgFontSize(s->vg, 48*2.5);
    nvgText(s->vg, alr_x+alr_w/2, alr_y+alr_h/2-45, va_text1, NULL);
    nvgFontFaceId(s->vg,  s->font_sans_regular);
    nvgFontSize(s->vg, 36*2.5);
    nvgText(s->vg, alr_x+alr_w/2, alr_y+alr_h/2+75, va_text2, NULL);
  } else if (va_size== ALERTSIZE_FULL) {
    nvgFontSize(s->vg, (longAlert1?72:96)*2.5);
    nvgFontFaceId(s->vg, s->font_sans_bold);
    nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_MIDDLE);
    nvgTextBox(s->vg, alr_x, alr_y+(longAlert1?360:420), alr_w-60, va_text1, NULL);
    nvgFontSize(s->vg, 48*2.5);
    nvgFontFaceId(s->vg,  s->font_sans_regular);
    nvgTextAlign(s->vg, NVG_ALIGN_CENTER | NVG_ALIGN_BOTTOM);
    nvgTextBox(s->vg, alr_x, alr_h-(longAlert1?300:360), alr_w-60, va_text2, NULL);
  }
}

static void ui_draw_vision(UIState *s) {
  const UIScene *scene = &s->scene;
  int ui_viz_rx = scene->ui_viz_rx;
  int ui_viz_rw = scene->ui_viz_rw;
  int ui_viz_ro = scene->ui_viz_ro;

  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  // Draw video frames
  glEnable(GL_SCISSOR_TEST);
  glViewport(ui_viz_rx+ui_viz_ro, s->fb_h-(box_y+box_h), viz_w, box_h);
  glScissor(ui_viz_rx, s->fb_h-(box_y+box_h), ui_viz_rw, box_h);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  draw_frame(s);
  glViewport(0, 0, s->fb_w, s->fb_h);
  glDisable(GL_SCISSOR_TEST);

  glClear(GL_STENCIL_BUFFER_BIT);

  nvgBeginFrame(s->vg, s->fb_w, s->fb_h, 1.0f);
  nvgSave(s->vg);

  // Draw augmented elements
  const int inner_height = viz_w*9/16;
  nvgScissor(s->vg, ui_viz_rx, box_y, ui_viz_rw, box_h);
  nvgTranslate(s->vg, ui_viz_rx+ui_viz_ro, box_y + (box_h-inner_height)/2.0);
  nvgScale(s->vg, (float)viz_w / s->fb_w, (float)inner_height / s->fb_h);
  if (!scene->frontview && !scene->fullview) {
    ui_draw_world(s);
  }

  nvgRestore(s->vg);

  // Set Speed, Current Speed, Status/Events
  ui_draw_vision_header(s);

  if (s->scene.alert_size != ALERTSIZE_NONE) {
    // Controls Alerts
    ui_draw_vision_alert(s, s->scene.alert_size, s->status,
                            s->scene.alert_text1, s->scene.alert_text2);
  } else {
    ui_draw_vision_footer(s);
  }


  nvgEndFrame(s->vg);
  glDisable(GL_BLEND);
}

static void ui_draw_blank(UIState *s) {
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
}

static void ui_draw_background(UIState *s) {
  int bg_status = s->status;
  assert(bg_status < ARRAYSIZE(bg_colors));
  const uint8_t *color = bg_colors[bg_status];

  glClearColor(color[0]/256.0, color[1]/256.0, color[2]/256.0, 1.0);
  glClear(GL_STENCIL_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
}

void ui_draw(UIState *s) {
#ifdef QCOM
  // TODO: once offroad can become transparent, we can remove bg_thread and this
  ui_draw_blank(s);
#else
  ui_draw_background(s);
#endif
  if (s->vision_connected && s->active_app == cereal_UiLayoutState_App_home && s->status != STATUS_STOPPED) {
    ui_draw_sidebar(s);
    ui_draw_vision(s);
  } else {
    if (!s->scene.uilayout_sidebarcollapsed) {
      ui_draw_sidebar(s);
    }
  }

  {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClear(GL_STENCIL_BUFFER_BIT);

    nvgBeginFrame(s->vg, s->fb_w, s->fb_h, 1.0f);

    nvgEndFrame(s->vg);
    glDisable(GL_BLEND);
  }
}

#ifdef NANOVG_GL3_IMPLEMENTATION
static const char frame_vertex_shader[] =
  "#version 150 core\n"
  "in vec4 aPosition;\n"
  "in vec4 aTexCoord;\n"
  "uniform mat4 uTransform;\n"
  "out vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vTexCoord = aTexCoord;\n"
  "}\n";

static const char frame_fragment_shader[] =
  "#version 150 core\n"
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "out vec4 vTexCoord;\n"
  "out vec4 outColor;\n"
  "void main() {\n"
  "  outColor = texture(uTexture, vTexCoord.xy);\n"
  "}\n";
#else
static const char frame_vertex_shader[] =
  "attribute vec4 aPosition;\n"
  "attribute vec4 aTexCoord;\n"
  "uniform mat4 uTransform;\n"
  "varying vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vTexCoord = aTexCoord;\n"
  "}\n";

static const char frame_fragment_shader[] =
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "varying vec4 vTexCoord;\n"
  "void main() {\n"
  "  gl_FragColor = texture2D(uTexture, vTexCoord.xy);\n"
  "}\n";
#endif

static const mat4 device_transform = {{
  1.0,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

// frame from 4/3 to box size with a 2x zoom
static const mat4 frame_transform = {{
  2*(4./3.)/((float)viz_w/box_h), 0.0, 0.0, 0.0,
  0.0, 2.0, 0.0, 0.0,
  0.0, 0.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 1.0,
}};

// frame from 4/3 to 16/9 display
static const mat4 full_to_wide_frame_transform = {{
  .75,  0.0, 0.0, 0.0,
  0.0,  1.0, 0.0, 0.0,
  0.0,  0.0, 1.0, 0.0,
  0.0,  0.0, 0.0, 1.0,
}};

void ui_nvg_init(UIState *s) {
  // init drawing
  s->vg = nvgCreate(NVG_ANTIALIAS | NVG_STENCIL_STROKES | NVG_DEBUG);
  assert(s->vg);

  s->font_courbd = nvgCreateFont(s->vg, "courbd", "../assets/fonts/courbd.ttf");
  assert(s->font_courbd >= 0);
  s->font_sans_regular = nvgCreateFont(s->vg, "sans-regular", "../assets/fonts/opensans_regular.ttf");
  assert(s->font_sans_regular >= 0);
  s->font_sans_semibold = nvgCreateFont(s->vg, "sans-semibold", "../assets/fonts/opensans_semibold.ttf");
  assert(s->font_sans_semibold >= 0);
  s->font_sans_bold = nvgCreateFont(s->vg, "sans-bold", "../assets/fonts/opensans_bold.ttf");
  assert(s->font_sans_bold >= 0);

  assert(s->img_wheel >= 0);
  s->img_wheel = nvgCreateImage(s->vg, "../assets/img_chffr_wheel.png", 1);

  assert(s->img_turn >= 0);
  s->img_turn = nvgCreateImage(s->vg, "../assets/img_trafficSign_turn.png", 1);

  assert(s->img_face >= 0);
  s->img_face = nvgCreateImage(s->vg, "../assets/img_driver_face.png", 1);

  assert(s->img_map >= 0);
  s->img_map = nvgCreateImage(s->vg, "../assets/img_map.png", 1);

  assert(s->img_button_settings >= 0);
  s->img_button_settings = nvgCreateImage(s->vg, "../assets/images/button_settings.png", 1);

  assert(s->img_button_home >= 0);
  s->img_button_home = nvgCreateImage(s->vg, "../assets/images/button_home.png", 1);

  assert(s->img_battery >= 0);
  s->img_battery = nvgCreateImage(s->vg, "../assets/images/battery.png", 1);

  assert(s->img_battery_charging >= 0);
  s->img_battery_charging = nvgCreateImage(s->vg, "../assets/images/battery_charging.png", 1);

  for(int i=0;i<=5;++i) {
    assert(s->img_network[i] >= 0);
    char network_asset[32];
    snprintf(network_asset, sizeof(network_asset), "../assets/images/network_%d.png", i);
    s->img_network[i] = nvgCreateImage(s->vg, network_asset, 1);
  }

  // init gl
  s->frame_program = load_program(frame_vertex_shader, frame_fragment_shader);
  assert(s->frame_program);

  s->frame_pos_loc = glGetAttribLocation(s->frame_program, "aPosition");
  s->frame_texcoord_loc = glGetAttribLocation(s->frame_program, "aTexCoord");

  s->frame_texture_loc = glGetUniformLocation(s->frame_program, "uTexture");
  s->frame_transform_loc = glGetUniformLocation(s->frame_program, "uTransform");

  glViewport(0, 0, s->fb_w, s->fb_h);

  glDisable(GL_DEPTH_TEST);

  assert(glGetError() == GL_NO_ERROR);

  for(int i = 0; i < 2; i++) {
    float x1, x2, y1, y2;
    if (i == 1) {
      // flip horizontally so it looks like a mirror
      x1 = 0.0;
      x2 = 1.0;
      y1 = 1.0;
      y2 = 0.0;
    } else {
      x1 = 1.0;
      x2 = 0.0;
      y1 = 1.0;
      y2 = 0.0;
    }
    const uint8_t frame_indicies[] = {0, 1, 2, 0, 2, 3};
    const float frame_coords[4][4] = {
      {-1.0, -1.0, x2, y1}, //bl
      {-1.0,  1.0, x2, y2}, //tl
      { 1.0,  1.0, x1, y2}, //tr
      { 1.0, -1.0, x1, y1}, //br
    };

    glGenVertexArrays(1,&s->frame_vao[i]);
    glBindVertexArray(s->frame_vao[i]);
    glGenBuffers(1, &s->frame_vbo[i]);
    glBindBuffer(GL_ARRAY_BUFFER, s->frame_vbo[i]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(frame_coords), frame_coords, GL_STATIC_DRAW);
    glEnableVertexAttribArray(s->frame_pos_loc);
    glVertexAttribPointer(s->frame_pos_loc, 2, GL_FLOAT, GL_FALSE,
                          sizeof(frame_coords[0]), (const void *)0);
    glEnableVertexAttribArray(s->frame_texcoord_loc);
    glVertexAttribPointer(s->frame_texcoord_loc, 2, GL_FLOAT, GL_FALSE,
                          sizeof(frame_coords[0]), (const void *)(sizeof(float) * 2));
    glGenBuffers(1, &s->frame_ibo[i]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s->frame_ibo[i]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(frame_indicies), frame_indicies, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
  }

  s->front_frame_mat = matmul(device_transform, full_to_wide_frame_transform);
  s->rear_frame_mat = matmul(device_transform, frame_transform);

  for(int i = 0;i < UI_BUF_COUNT; i++) {
    s->khr[i] = NULL;
    s->priv_hnds[i] = NULL;
    s->khr_front[i] = NULL;
    s->priv_hnds_front[i] = NULL;
  }
}
