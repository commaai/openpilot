#include <sys/time.h>
#include <sys/resource.h>

#include "locationd.h"

using namespace EKFS;
using namespace Eigen;

ExitHandler do_exit;

static VectorXd floatlist2vector(const capnp::List<float, capnp::Kind::PRIMITIVE>::Reader& floatlist) {
  VectorXd res(floatlist.size());
  for (int i = 0; i < floatlist.size(); i++) {
    res[i] = floatlist[i];
  }
  return res;
}

static Vector4d quat2vector(const Quaterniond& quat) {
  return Vector4d(quat.w(), quat.x(), quat.y(), quat.z());
}

static Quaterniond vector2quat(const VectorXd& vec) {
  return Quaterniond(vec(0), vec(1), vec(2), vec(3));
}

static void init_measurement(cereal::LiveLocationKalman::Measurement::Builder meas, const VectorXd& val, const VectorXd& std, bool valid) {
  meas.setValue(kj::arrayPtr(val.data(), val.size()));
  meas.setStd(kj::arrayPtr(std.data(), std.size()));
  meas.setValid(valid);
}

Localizer::Localizer() {
  this->kf = std::make_unique<LiveKalman>();
  this->reset_kalman();

  this->calib = Vector3d(0.0, 0.0, 0.0);
  this->device_from_calib = MatrixXdr::Identity(3, 3);
  this->calib_from_device = MatrixXdr::Identity(3, 3);

  for (int i = 0; i < POSENET_STD_HIST_HALF * 2; i++) {
    this->posenet_stds.push_back(10.0);
  }

  VectorXd ecef_pos = this->kf->get_x().segment<STATE_ECEF_POS_LEN>(STATE_ECEF_POS_START);
  this->converter = std::make_unique<LocalCoord>((ECEF) { .x = ecef_pos[0], .y = ecef_pos[1], .z = ecef_pos[2] });
}

void Localizer::build_live_location(cereal::LiveLocationKalman::Builder& fix) {
  VectorXd predicted_state = this->kf->get_x();
  MatrixXdr predicted_cov = this->kf->get_P();
  VectorXd predicted_std = predicted_cov.diagonal().array().sqrt();

  VectorXd fix_ecef = predicted_state.segment<STATE_ECEF_POS_LEN>(STATE_ECEF_POS_START);
  ECEF fix_ecef_ecef = { .x = fix_ecef(0), .y = fix_ecef(1), .z = fix_ecef(2) };
  VectorXd fix_ecef_std = predicted_std.segment<STATE_ECEF_POS_ERR_LEN>(STATE_ECEF_POS_ERR_START);
  VectorXd vel_ecef = predicted_state.segment<STATE_ECEF_VELOCITY_LEN>(STATE_ECEF_VELOCITY_START);
  VectorXd vel_ecef_std = predicted_std.segment<STATE_ECEF_VELOCITY_ERR_LEN>(STATE_ECEF_VELOCITY_ERR_START);
  VectorXd fix_pos_geo_vec = this->get_position_geodetic();
  //fix_pos_geo_std = np.abs(coord.ecef2geodetic(fix_ecef + fix_ecef_std) - fix_pos_geo)
  VectorXd orientation_ecef = quat2euler(vector2quat(predicted_state.segment<STATE_ECEF_ORIENTATION_LEN>(STATE_ECEF_ORIENTATION_START)));
  VectorXd orientation_ecef_std = predicted_std.segment<STATE_ECEF_ORIENTATION_ERR_LEN>(STATE_ECEF_ORIENTATION_ERR_START);
  MatrixXdr device_from_ecef = quat2rot(vector2quat(predicted_state.segment<STATE_ECEF_ORIENTATION_LEN>(STATE_ECEF_ORIENTATION_START))).transpose();
  VectorXd calibrated_orientation_ecef = rot2euler(this->calib_from_device * device_from_ecef);

  VectorXd acc_calib = this->calib_from_device * predicted_state.segment<STATE_ACCELERATION_LEN>(STATE_ACCELERATION_START);
  VectorXd acc_calib_std = ((this->calib_from_device * predicted_cov.block<STATE_ACCELERATION_ERR_LEN, STATE_ACCELERATION_ERR_LEN>(STATE_ACCELERATION_ERR_START, STATE_ACCELERATION_ERR_START)) * this->calib_from_device.transpose()).diagonal().array().sqrt();
  VectorXd ang_vel_calib = this->calib_from_device * predicted_state.segment<STATE_ANGULAR_VELOCITY_LEN>(STATE_ANGULAR_VELOCITY_START);

  MatrixXdr vel_angular_err = predicted_cov.block<STATE_ANGULAR_VELOCITY_ERR_LEN, STATE_ANGULAR_VELOCITY_ERR_LEN>(STATE_ANGULAR_VELOCITY_ERR_START, STATE_ANGULAR_VELOCITY_ERR_START);
  VectorXd ang_vel_calib_std = ((this->calib_from_device * vel_angular_err) * this->calib_from_device.transpose()).diagonal().array().sqrt();

  VectorXd vel_device = device_from_ecef * vel_ecef;
  VectorXd device_from_ecef_eul = quat2euler(vector2quat(predicted_state.segment<STATE_ECEF_ORIENTATION_LEN>(STATE_ECEF_ORIENTATION_START))).transpose();
  MatrixXdr condensed_cov(STATE_ECEF_ORIENTATION_ERR_LEN + STATE_ECEF_VELOCITY_ERR_LEN, STATE_ECEF_ORIENTATION_ERR_LEN + STATE_ECEF_VELOCITY_ERR_LEN);
  condensed_cov.topLeftCorner<STATE_ECEF_ORIENTATION_ERR_LEN, STATE_ECEF_ORIENTATION_ERR_LEN>() =
    predicted_cov.block<STATE_ECEF_ORIENTATION_ERR_LEN, STATE_ECEF_ORIENTATION_ERR_LEN>(STATE_ECEF_ORIENTATION_ERR_START, STATE_ECEF_ORIENTATION_ERR_START);
  condensed_cov.topRightCorner<STATE_ECEF_ORIENTATION_ERR_LEN, STATE_ECEF_VELOCITY_ERR_LEN>() =
    predicted_cov.block<STATE_ECEF_ORIENTATION_ERR_LEN, STATE_ECEF_VELOCITY_ERR_LEN>(STATE_ECEF_ORIENTATION_ERR_START, STATE_ECEF_VELOCITY_ERR_START);
  condensed_cov.bottomRightCorner<STATE_ECEF_VELOCITY_ERR_LEN, STATE_ECEF_VELOCITY_ERR_LEN>() =
    predicted_cov.block<STATE_ECEF_VELOCITY_ERR_LEN, STATE_ECEF_VELOCITY_ERR_LEN>(STATE_ECEF_VELOCITY_ERR_START, STATE_ECEF_VELOCITY_ERR_START);
  condensed_cov.bottomLeftCorner<STATE_ECEF_VELOCITY_ERR_LEN, STATE_ECEF_ORIENTATION_ERR_LEN>() =
    predicted_cov.block<STATE_ECEF_VELOCITY_ERR_LEN, STATE_ECEF_ORIENTATION_ERR_LEN>(STATE_ECEF_VELOCITY_ERR_START, STATE_ECEF_ORIENTATION_ERR_START);
  VectorXd H_input(device_from_ecef_eul.size() + vel_ecef.size());
  H_input << device_from_ecef_eul, vel_ecef;
  MatrixXdr HH = this->kf->H(H_input);
  MatrixXdr vel_device_cov = (HH * condensed_cov) * HH.transpose();
  VectorXd vel_device_std = vel_device_cov.diagonal().array().sqrt();

  VectorXd vel_calib = this->calib_from_device * vel_device;
  VectorXd vel_calib_std = ((this->calib_from_device * vel_device_cov) * this->calib_from_device.transpose()).diagonal().array().sqrt();

  VectorXd orientation_ned = ned_euler_from_ecef(fix_ecef_ecef, orientation_ecef);
  //orientation_ned_std = ned_euler_from_ecef(fix_ecef, orientation_ecef + orientation_ecef_std) - orientation_ned
  VectorXd nextfix_ecef = fix_ecef + vel_ecef;
  VectorXd ned_vel = this->converter->ecef2ned((ECEF) { .x = nextfix_ecef(0), .y = nextfix_ecef(1), .z = nextfix_ecef(2) }).to_vector() - converter->ecef2ned(fix_ecef_ecef).to_vector();
  //ned_vel_std = self.converter->ecef2ned(fix_ecef + vel_ecef + vel_ecef_std) - self.converter->ecef2ned(fix_ecef + vel_ecef)

  VectorXd accDevice = predicted_state.segment<STATE_ACCELERATION_LEN>(STATE_ACCELERATION_START);
  VectorXd accDeviceErr = predicted_std.segment<STATE_ACCELERATION_ERR_LEN>(STATE_ACCELERATION_ERR_START);

  VectorXd angVelocityDevice = predicted_state.segment<STATE_ANGULAR_VELOCITY_LEN>(STATE_ANGULAR_VELOCITY_START);
  VectorXd angVelocityDeviceErr = predicted_std.segment<STATE_ANGULAR_VELOCITY_ERR_LEN>(STATE_ANGULAR_VELOCITY_ERR_START);

  Vector3d nans = Vector3d(NAN, NAN, NAN);

  // write measurements to msg
  init_measurement(fix.initPositionGeodetic(), fix_pos_geo_vec, nans, true);
  init_measurement(fix.initPositionECEF(), fix_ecef, fix_ecef_std, true);
  init_measurement(fix.initVelocityECEF(), vel_ecef, vel_ecef_std, true);
  init_measurement(fix.initVelocityNED(), ned_vel, nans, true);
  init_measurement(fix.initVelocityDevice(), vel_device, vel_device_std, true);
  init_measurement(fix.initAccelerationDevice(), accDevice, accDeviceErr, true);
  init_measurement(fix.initOrientationECEF(), orientation_ecef, orientation_ecef_std, true);
  init_measurement(fix.initCalibratedOrientationECEF(), calibrated_orientation_ecef, nans, this->calibrated);
  init_measurement(fix.initOrientationNED(), orientation_ned, nans, true);
  init_measurement(fix.initAngularVelocityDevice(), angVelocityDevice, angVelocityDeviceErr, true);
  init_measurement(fix.initVelocityCalibrated(), vel_calib, vel_calib_std, this->calibrated);
  init_measurement(fix.initAngularVelocityCalibrated(), ang_vel_calib, ang_vel_calib_std, this->calibrated);
  init_measurement(fix.initAccelerationCalibrated(), acc_calib, acc_calib_std, this->calibrated);

  double old_mean = 0.0, new_mean = 0.0;
  int i = 0;
  for (double x : this->posenet_stds) {
    if (i < POSENET_STD_HIST_HALF) {
      old_mean += x;
    } else {
      new_mean += x;
    }
    i++;
  }
  old_mean /= POSENET_STD_HIST_HALF;
  new_mean /= POSENET_STD_HIST_HALF;
  // experimentally found these values, no false positives in 20k minutes of driving
  bool std_spike = (new_mean / old_mean > 4.0 && new_mean > 7.0);

  fix.setPosenetOK(!(std_spike && this->car_speed > 5.0));
  fix.setDeviceStable(!this->device_fell);
  this->device_fell = false;

  //fix.setGpsWeek(this->time.week);
  //fix.setGpsTimeOfWeek(this->time.tow);
  fix.setUnixTimestampMillis(this->unix_timestamp_millis);

  if (fix_ecef_std.norm() < 50.0 && this->calibrated) {
    fix.setStatus(cereal::LiveLocationKalman::Status::VALID);
  } else if (fix_ecef_std.norm() < 50.0) {
    fix.setStatus(cereal::LiveLocationKalman::Status::UNCALIBRATED);
  } else {
    fix.setStatus(cereal::LiveLocationKalman::Status::UNINITIALIZED);
  }
}

VectorXd Localizer::get_position_geodetic() {
  VectorXd fix_ecef = this->kf->get_x().segment<STATE_ECEF_POS_LEN>(STATE_ECEF_POS_START);
  ECEF fix_ecef_ecef = { .x = fix_ecef(0), .y = fix_ecef(1), .z = fix_ecef(2) };
  Geodetic fix_pos_geo = ecef2geodetic(fix_ecef_ecef);
  return Vector3d(fix_pos_geo.lat, fix_pos_geo.lon, fix_pos_geo.alt);
}

void Localizer::handle_sensors(double current_time, const capnp::List<cereal::SensorEventData, capnp::Kind::STRUCT>::Reader& log) {
  // TODO does not yet account for double sensor readings in the log
  for (int i = 0; i < log.size(); i++) {
    const cereal::SensorEventData::Reader& sensor_reading = log[i];
    double sensor_time = 1e-9 * sensor_reading.getTimestamp();
    // TODO: handle messages from two IMUs at the same time
    if (sensor_reading.getSource() == cereal::SensorEventData::SensorSource::LSM6DS3) {
      continue;
    }

    // Gyro Uncalibrated
    if (sensor_reading.getSensor() == SENSOR_GYRO_UNCALIBRATED && sensor_reading.getType() == SENSOR_TYPE_GYROSCOPE_UNCALIBRATED) {
      this->gyro_counter++;
      if (this->gyro_counter % SENSOR_DECIMATION == 0) {
        auto v = sensor_reading.getGyroUncalibrated().getV();
        this->kf->predict_and_observe(sensor_time, OBSERVATION_PHONE_GYRO, { Vector3d(-v[2], -v[1], -v[0]) });
      }
    }

    // Accelerometer
    if (sensor_reading.getSensor() == SENSOR_ACCELEROMETER && sensor_reading.getType() == SENSOR_TYPE_ACCELEROMETER) {
      auto v = sensor_reading.getAcceleration().getV();

      // check if device fell, estimate 10 for g
      // 40m/s**2 is a good filter for falling detection, no false positives in 20k minutes of driving
      this->device_fell |= (floatlist2vector(v) - Vector3d(10.0, 0.0, 0.0)).norm() > 40.0;

      this->acc_counter++;
      if (this->acc_counter % SENSOR_DECIMATION == 0) {
        this->kf->predict_and_observe(sensor_time, OBSERVATION_PHONE_ACCEL, { Vector3d(-v[2], -v[1], -v[0]) });
      }
    }
  }
}

void Localizer::handle_gps(double current_time, const cereal::GpsLocationData::Reader& log) {
  // ignore the message if the fix is invalid
  if (log.getFlags() % 2 == 0) {
    return;
  }

  this->last_gps_fix = current_time;

  Geodetic geodetic = { log.getLatitude(), log.getLongitude(), log.getAltitude() };
  this->converter = std::make_unique<LocalCoord>(geodetic);

  VectorXd ecef_pos = this->converter->ned2ecef({ 0.0, 0.0, 0.0 }).to_vector();
  VectorXd ecef_vel = this->converter->ned2ecef({ log.getVNED()[0], log.getVNED()[1], log.getVNED()[2] }).to_vector() - ecef_pos;
  MatrixXdr ecef_pos_R = Vector3d::Constant(std::pow(3.0 * log.getVerticalAccuracy(), 2)).asDiagonal();
  MatrixXdr ecef_vel_R = Vector3d::Constant(std::pow(log.getSpeedAccuracy(), 2)).asDiagonal();

  this->unix_timestamp_millis = log.getTimestamp();
  double gps_est_error = (this->kf->get_x().head(3) - ecef_pos).norm();

  VectorXd orientation_ecef = quat2euler(vector2quat(this->kf->get_x().segment<STATE_ECEF_ORIENTATION_LEN>(STATE_ECEF_ORIENTATION_START)));
  VectorXd orientation_ned = ned_euler_from_ecef({ ecef_pos(0), ecef_pos(1), ecef_pos(2) }, orientation_ecef);
  VectorXd orientation_ned_gps = Vector3d(0.0, 0.0, DEG2RAD(log.getBearingDeg()));
  VectorXd orientation_error = (orientation_ned - orientation_ned_gps).array() - M_PI;
  for (int i = 0; i < orientation_error.size(); i++) {
    orientation_error(i) = std::fmod(orientation_error(i), 2.0 * M_PI);
    if (orientation_error(i) < 0.0) {
      orientation_error(i) += 2.0 * M_PI;
    }
    orientation_error(i) -= M_PI;
  }
  VectorXd initial_pose_ecef_quat = quat2vector(euler2quat(ecef_euler_from_ned({ ecef_pos(0), ecef_pos(1), ecef_pos(2) }, orientation_ned_gps)));

  if (ecef_vel.norm() > 5.0 && orientation_error.norm() > 1.0) {
    LOGE("Locationd vs ubloxLocation orientation difference too large, kalman reset");
    this->reset_kalman(NAN, initial_pose_ecef_quat, ecef_pos);
    this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_ORIENTATION_FROM_GPS, { initial_pose_ecef_quat });
  } else if (gps_est_error > 50.0) {
    LOGE("Locationd vs ubloxLocation position difference too large, kalman reset");
    this->reset_kalman(NAN, initial_pose_ecef_quat, ecef_pos);
  }

  this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_POS, { ecef_pos }, { ecef_pos_R });
  this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_VEL, { ecef_vel }, { ecef_vel_R });
}

void Localizer::handle_car_state(double current_time, const cereal::CarState::Reader& log) {
  this->speed_counter++;

  if (this->speed_counter % SENSOR_DECIMATION == 0) {
    this->kf->predict_and_observe(current_time, OBSERVATION_ODOMETRIC_SPEED, { (VectorXd(1) << log.getVEgoRaw()).finished() });
    this->car_speed = std::abs(log.getVEgo());
    if (this->car_speed < 1e-3) {
      this->kf->predict_and_observe(current_time, OBSERVATION_NO_ROT, { Vector3d(0.0, 0.0, 0.0) });
    }
  }
}

void Localizer::handle_cam_odo(double current_time, const cereal::CameraOdometry::Reader& log) {
  this->cam_counter++;

  if (this->cam_counter % VISION_DECIMATION == 0) {
    VectorXd rot_device = this->device_from_calib * floatlist2vector(log.getRot());
    VectorXd rot_device_std = (this->device_from_calib * floatlist2vector(log.getRotStd())) * 10.0;
    this->kf->predict_and_observe(current_time, OBSERVATION_CAMERA_ODO_ROTATION,
      { (VectorXd(rot_device.rows() + rot_device_std.rows()) << rot_device, rot_device_std).finished() });

    VectorXd trans_device = this->device_from_calib * floatlist2vector(log.getTrans());
    VectorXd trans_device_std = this->device_from_calib * floatlist2vector(log.getTransStd());

    this->posenet_stds.pop_front();
    this->posenet_stds.push_back(trans_device_std[0]);

    trans_device_std *= 10.0;
    this->kf->predict_and_observe(current_time, OBSERVATION_CAMERA_ODO_TRANSLATION,
      { (VectorXd(trans_device.rows() + trans_device_std.rows()) << trans_device, trans_device_std).finished() });
  }
}

void Localizer::handle_live_calib(double current_time, const cereal::LiveCalibrationData::Reader& log) {
  if (log.getRpyCalib().size() > 0) {
    this->calib = floatlist2vector(log.getRpyCalib());
    this->device_from_calib = euler2rot(this->calib);
    this->calib_from_device = this->device_from_calib.transpose();
    this->calibrated = log.getCalStatus() == 1;
  }
}

void Localizer::reset_kalman(double current_time) {
  VectorXd init_x = this->kf->get_initial_x();
  this->reset_kalman(current_time, init_x.segment<4>(3), init_x.head(3));
}

void Localizer::reset_kalman(double current_time, VectorXd init_orient, VectorXd init_pos) {
  // too nonlinear to init on completely wrong
  VectorXd init_x = this->kf->get_initial_x();
  MatrixXdr init_P = this->kf->get_initial_P();
  init_x.segment<4>(3) = init_orient;
  init_x.head(3) = init_pos;

  this->kf->init_state(init_x, init_P, current_time);

  this->gyro_counter = 0;
  this->acc_counter = 0;
  this->speed_counter = 0;
  this->cam_counter = 0;
}

void Localizer::handle_msg_bytes(const char *data, const size_t size) {
  AlignedBuffer aligned_buf;

  capnp::FlatArrayMessageReader cmsg(aligned_buf.align(data, size));
  cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

  this->handle_msg(event);
}

void Localizer::handle_msg(const cereal::Event::Reader& log) {
  double t = log.getLogMonoTime() * 1e-9;
  if (log.isSensorEvents()) {
    this->handle_sensors(t, log.getSensorEvents());
  } else if (log.isGpsLocationExternal()) {
    this->handle_gps(t, log.getGpsLocationExternal());
  } else if (log.isCarState()) {
    this->handle_car_state(t, log.getCarState());
  } else if (log.isCameraOdometry()) {
    this->handle_cam_odo(t, log.getCameraOdometry());
  } else if (log.isLiveCalibration()) {
    this->handle_live_calib(t, log.getLiveCalibration());
  }
}

kj::ArrayPtr<capnp::byte> Localizer::get_message_bytes(MessageBuilder& msg_builder, uint64_t logMonoTime,
  bool inputsOK, bool sensorsOK, bool gpsOK)
{
  cereal::Event::Builder evt = msg_builder.initEvent();
  evt.setLogMonoTime(logMonoTime);
  cereal::LiveLocationKalman::Builder liveLoc = evt.initLiveLocationKalman();
  this->build_live_location(liveLoc);
  liveLoc.setInputsOK(inputsOK);
  liveLoc.setSensorsOK(sensorsOK);
  liveLoc.setGpsOK(gpsOK);
  return msg_builder.toBytes();
}

int Localizer::locationd_thread() {
  const std::initializer_list<const char *> service_list =
      { "gpsLocationExternal", "sensorEvents", "cameraOdometry", "liveCalibration", "carState" };
  SubMaster sm(service_list, nullptr, { "gpsLocationExternal" });
  PubMaster pm({ "liveLocationKalman" });

  Params params;

  while (!do_exit) {
    sm.update();
    for (const char* service : service_list) {
      if (sm.updated(service) && sm.valid(service)) {
        const cereal::Event::Reader log = sm[service];
        this->handle_msg(log);
      }
    }

    if (sm.updated("cameraOdometry")) {
      uint64_t logMonoTime = sm["cameraOdometry"].getLogMonoTime();
      bool inputsOK = sm.allAliveAndValid();
      bool sensorsOK = sm.alive("sensorEvents") && sm.valid("sensorEvents");
      bool gpsOK = (logMonoTime / 1e9) - this->last_gps_fix < 1.0;

      MessageBuilder msg_builder;
      kj::ArrayPtr<capnp::byte> bytes = this->get_message_bytes(msg_builder, logMonoTime, inputsOK, sensorsOK, gpsOK);
      pm.send("liveLocationKalman", bytes.begin(), bytes.size());

      if (sm.frame % 1200 == 0 && gpsOK) {  // once a minute
        VectorXd posGeo = this->get_position_geodetic();
        std::string lastGPSPosJSON = util::string_format(
          "{\"latitude\": %.15f, \"longitude\": %.15f, \"altitude\": %.15f}", posGeo(0), posGeo(1), posGeo(2));

        std::thread([&params] (const std::string gpsjson) {
          params.put("LastGPSPosition", gpsjson);
        }, lastGPSPosJSON).detach();
      }
    }
  }
  return 0;
}

int main() {
  setpriority(PRIO_PROCESS, 0, -20);

  Localizer localizer;
  return localizer.locationd_thread();
}
