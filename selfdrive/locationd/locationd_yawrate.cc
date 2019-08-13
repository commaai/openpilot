#include <iostream>
#include <cmath>

#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <eigen3/Eigen/Dense>

#include "cereal/gen/cpp/log.capnp.h"

#include "locationd_yawrate.h"


void Localizer::update_state(const Eigen::Matrix<double, 1, 2> &C, const double R, double current_time, double meas) {
  double dt = current_time - prev_update_time;

  if (dt < 0) {
    dt = 0;
  } else {
    prev_update_time = current_time;
  }

  // x = A * x;
  // P = A * P * A.transpose() + dt * Q;
  // Simplify because A is unity
  P = P + dt * Q;

  double y = meas - C * x;
  double S = R + C * P * C.transpose();
  Eigen::Vector2d K = P * C.transpose() * (1.0 / S);
  x = x + K * y;
  P = (I - K * C) * P;
}

void Localizer::handle_sensor_events(capnp::List<cereal::SensorEventData>::Reader sensor_events, double current_time) {
  for (cereal::SensorEventData::Reader sensor_event : sensor_events){
    if (sensor_event.getType() == 4) {
      sensor_data_time = current_time;

      double meas = -sensor_event.getGyro().getV()[0];
      update_state(C_gyro, R_gyro, current_time, meas);
    }
  }
}

void Localizer::handle_camera_odometry(cereal::CameraOdometry::Reader camera_odometry, double current_time) {
  double R = 250.0 * pow(camera_odometry.getRotStd()[2], 2);
  double meas = camera_odometry.getRot()[2];
  update_state(C_posenet, R, current_time, meas);

  auto trans = camera_odometry.getTrans();
  posenet_speed = sqrt(trans[0]*trans[0] + trans[1]*trans[1] + trans[2]*trans[2]);
}

void Localizer::handle_controls_state(cereal::ControlsState::Reader controls_state, double current_time) {
  steering_angle = controls_state.getAngleSteers() * DEGREES_TO_RADIANS;
  car_speed = controls_state.getVEgo();
  controls_state_time = current_time;
}


Localizer::Localizer() {
  A << 1, 0, 0, 1;
  I << 1, 0, 0, 1;

  Q << pow(0.1, 2.0), 0, 0, pow(0.005 / 100.0, 2.0);
  P << pow(1.0, 2.0), 0, 0, pow(0.05, 2.0);

  C_posenet << 1, 0;
  C_gyro << 1, 1;
  x << 0, 0;

  R_gyro = pow(0.05, 2.0);
}

void Localizer::handle_log(cereal::Event::Reader event) {
  double current_time = event.getLogMonoTime() / 1.0e9;

  // Initialize update_time on first update
  if (prev_update_time < 0) {
    prev_update_time = current_time;
  }

  auto type = event.which();
  switch(type) {
  case cereal::Event::CONTROLS_STATE:
    handle_controls_state(event.getControlsState(), current_time);
    break;
  case cereal::Event::CAMERA_ODOMETRY:
    handle_camera_odometry(event.getCameraOdometry(), current_time);
    break;
  case cereal::Event::SENSOR_EVENTS:
    handle_sensor_events(event.getSensorEvents(), current_time);
    break;
  default:
    break;
  }
}


extern "C" {
  void *localizer_init(void) {
    Localizer * localizer = new Localizer;
    return (void*)localizer;
  }

  void localizer_handle_log(void * localizer, const unsigned char * data, size_t len) {
    const kj::ArrayPtr<const capnp::word> view((const capnp::word*)data, len);
    capnp::FlatArrayMessageReader msg(view);
    cereal::Event::Reader event = msg.getRoot<cereal::Event>();

    Localizer * loc = (Localizer*) localizer;
    loc->handle_log(event);
  }

  double localizer_get_yaw(void * localizer) {
    Localizer * loc = (Localizer*) localizer;
    return loc->x[0];
  }
  double localizer_get_bias(void * localizer) {
    Localizer * loc = (Localizer*) localizer;
    return loc->x[1];
  }

  void localizer_set_yaw(void * localizer, double yaw) {
    Localizer * loc = (Localizer*) localizer;
    loc->x[0] = yaw;
  }
  void localizer_set_bias(void * localizer, double bias) {
    Localizer * loc = (Localizer*) localizer;
    loc->x[1] = bias;
  }

  double localizer_get_t(void * localizer) {
    Localizer * loc = (Localizer*) localizer;
    return loc->prev_update_time;
  }
}
