// TODO: cleanup AE tests
// needed by camera_common.cc
void camera_autoexposure(CameraState *s, float grey_frac) {}
void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {}
void cameras_open(MultiCameraState *s) {}
void cameras_run(MultiCameraState *s) {}

typedef struct CameraState {
  int camera_num;
  CameraInfo ci;

  int fps;
  float digital_gain = 0;

  CameraBuf buf;
} CameraState;

typedef struct MultiCameraState {
  CameraState road_cam;
  CameraState driver_cam;

  PubMaster *pm = nullptr;
} MultiCameraState;


