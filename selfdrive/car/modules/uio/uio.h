#define UI_BUF_COUNT 4

typedef struct UIScene {
  int frontview;

  int transformed_width, transformed_height;

  uint64_t model_ts;
  ModelData model;

  float mpc_x[50];
  float mpc_y[50];

  bool world_objects_visible;
  mat3 warp_matrix;           // transformed box -> frame.
  mat4 extrinsic_matrix;      // Last row is 0 so we can use mat4.

  float v_cruise;
  uint64_t v_cruise_update_ts;
  float v_ego;
  float curvature;
  int engaged;
  bool engageable;

  bool uilayout_sidebarcollapsed;
  bool uilayout_mapenabled;
  // responsive layout
  int ui_viz_rx;
  int ui_viz_rw;
  int ui_viz_ro;

  int lead_status;
  float lead_d_rel, lead_y_rel, lead_v_rel;

  int front_box_x, front_box_y, front_box_width, front_box_height;

  uint64_t alert_ts;
  char alert_text1[1024];
  char alert_text2[1024];
  uint8_t alert_size;
  float alert_blinkingrate;

  float awareness_status;

  uint64_t started_ts;
  
  // Used to display calibration progress
  int cal_status;
  int cal_perc;
  // Used to show gps planner status
  bool gps_planner_active;

} UIScene;

typedef struct UIState {
  pthread_mutex_t lock;
  pthread_cond_t bg_cond;

  FramebufferState *fb;
  int fb_w, fb_h;
  EGLDisplay display;
  EGLSurface surface;

  NVGcontext *vg;

  int font_courbd;
  int font_sans_regular;
  int font_sans_semibold;
  int font_sans_bold;
  int img_wheel;

  zsock_t *thermal_sock;
  void *thermal_sock_raw;
  zsock_t *model_sock;
  void *model_sock_raw;
  zsock_t *live100_sock;
  void *live100_sock_raw;
  zsock_t *livecalibration_sock;
  void *livecalibration_sock_raw;
  zsock_t *live20_sock;
  void *live20_sock_raw;
  zsock_t *livempc_sock;
  void *livempc_sock_raw;
  zsock_t *plus_sock;
  void *plus_sock_raw;
  zsock_t *gps_sock;
  void *gps_sock_raw;

  zsock_t *uilayout_sock;
  void *uilayout_sock_raw;

  int plus_state;

  // vision state
  bool vision_connected;
  bool vision_connect_firstrun;
  int ipc_fd;

  VIPCBuf bufs[UI_BUF_COUNT];
  VIPCBuf front_bufs[UI_BUF_COUNT];
  int cur_vision_idx;
  int cur_vision_front_idx;

  GLuint frame_program;
  GLuint frame_texs[UI_BUF_COUNT];
  GLuint frame_front_texs[UI_BUF_COUNT];

  GLint frame_pos_loc, frame_texcoord_loc;
  GLint frame_texture_loc, frame_transform_loc;

  GLuint line_program;
  GLint line_pos_loc, line_color_loc;
  GLint line_transform_loc;

  unsigned int rgb_width, rgb_height, rgb_stride;
  size_t rgb_buf_len;
  mat4 rgb_transform;

  unsigned int rgb_front_width, rgb_front_height, rgb_front_stride;
  size_t rgb_front_buf_len;

  bool intrinsic_matrix_loaded;
  mat3 intrinsic_matrix;

  UIScene scene;

  bool awake;
  int awake_timeout;

  int status;
  bool is_metric;
  bool passive;
  int alert_size;
  float alert_blinking_alpha;
  bool alert_blinked;
  bool acc_enabled;

  float light_sensor;
} UIState;

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

static const char line_vertex_shader[] =
  "attribute vec4 aPosition;\n"
  "attribute vec4 aColor;\n"
  "uniform mat4 uTransform;\n"
  "varying vec4 vColor;\n"
  "void main() {\n"
  "  gl_Position = uTransform * aPosition;\n"
  "  vColor = aColor;\n"
  "}\n";

static const char line_fragment_shader[] =
  "precision mediump float;\n"
  "uniform sampler2D uTexture;\n"
  "varying vec4 vColor;\n"
  "void main() {\n"
  "  gl_FragColor = vColor;\n"
  "}\n";