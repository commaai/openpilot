#include <sys/time.h>
#include <sys/resource.h>

#include <cmath>

#include "locationd.h"

using namespace EKFS;
using namespace Eigen;

ExitHandler do_exit;
const double ACCEL_SANITY_CHECK = 100.0;  // m/s^2
const double ROTATION_SANITY_CHECK = 10.0;  // rad/s
const double TRANS_SANITY_CHECK = 200.0;  // m/s
const double CALIB_RPY_SANITY_CHECK = 0.5; // rad (+- 30 deg)
const double ALTITUDE_SANITY_CHECK = 10000; // m
const double MIN_STD_SANITY_CHECK = 1e-5; // m or rad
const double VALID_TIME_SINCE_RESET = 1.0; // s
const double VALID_POS_STD = 50.0; // m
const double MAX_RESET_TRACKER = 5.0;
const double SANE_GPS_UNCERTAINTY = 1500.0; // m

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


static MatrixXdr rotate_cov(const MatrixXdr& rot_matrix, const MatrixXdr& cov_in) {
  // To rotate a covariance matrix, the cov matrix needs to multiplied left and right by the transform matrix
  return ((rot_matrix *  cov_in) * rot_matrix.transpose());
}

static VectorXd rotate_std(const MatrixXdr& rot_matrix, const VectorXd& std_in) {
  // Stds cannot be rotated like values, only covariances can be rotated
  return rotate_cov(rot_matrix, std_in.array().square().matrix().asDiagonal()).diagonal().array().sqrt();
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
  VectorXd orientation_ecef = quat2euler(vector2quat(predicted_state.segment<STATE_ECEF_ORIENTATION_LEN>(STATE_ECEF_ORIENTATION_START)));
  VectorXd orientation_ecef_std = predicted_std.segment<STATE_ECEF_ORIENTATION_ERR_LEN>(STATE_ECEF_ORIENTATION_ERR_START);
  MatrixXdr orientation_ecef_cov = predicted_cov.block<STATE_ECEF_ORIENTATION_ERR_LEN, STATE_ECEF_ORIENTATION_ERR_LEN>(STATE_ECEF_ORIENTATION_ERR_START, STATE_ECEF_ORIENTATION_ERR_START);
  MatrixXdr device_from_ecef = euler2rot(orientation_ecef).transpose();
  VectorXd calibrated_orientation_ecef = rot2euler((this->calib_from_device * device_from_ecef).transpose());

  VectorXd acc_calib = this->calib_from_device * predicted_state.segment<STATE_ACCELERATION_LEN>(STATE_ACCELERATION_START);
  MatrixXdr acc_calib_cov = predicted_cov.block<STATE_ACCELERATION_ERR_LEN, STATE_ACCELERATION_ERR_LEN>(STATE_ACCELERATION_ERR_START, STATE_ACCELERATION_ERR_START);
  VectorXd acc_calib_std = rotate_cov(this->calib_from_device, acc_calib_cov).diagonal().array().sqrt();
  VectorXd ang_vel_calib = this->calib_from_device * predicted_state.segment<STATE_ANGULAR_VELOCITY_LEN>(STATE_ANGULAR_VELOCITY_START);

  MatrixXdr vel_angular_cov = predicted_cov.block<STATE_ANGULAR_VELOCITY_ERR_LEN, STATE_ANGULAR_VELOCITY_ERR_LEN>(STATE_ANGULAR_VELOCITY_ERR_START, STATE_ANGULAR_VELOCITY_ERR_START);
  VectorXd ang_vel_calib_std = rotate_cov(this->calib_from_device, vel_angular_cov).diagonal().array().sqrt();

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
  VectorXd vel_calib_std = rotate_cov(this->calib_from_device, vel_device_cov).diagonal().array().sqrt();

  VectorXd orientation_ned = ned_euler_from_ecef(fix_ecef_ecef, orientation_ecef);
  VectorXd orientation_ned_std = rotate_cov(this->converter->ecef2ned_matrix, orientation_ecef_cov).diagonal().array().sqrt();
  VectorXd calibrated_orientation_ned = ned_euler_from_ecef(fix_ecef_ecef, calibrated_orientation_ecef);
  VectorXd nextfix_ecef = fix_ecef + vel_ecef;
  VectorXd ned_vel = this->converter->ecef2ned((ECEF) { .x = nextfix_ecef(0), .y = nextfix_ecef(1), .z = nextfix_ecef(2) }).to_vector() - converter->ecef2ned(fix_ecef_ecef).to_vector();

  VectorXd accDevice = predicted_state.segment<STATE_ACCELERATION_LEN>(STATE_ACCELERATION_START);
  VectorXd accDeviceErr = predicted_std.segment<STATE_ACCELERATION_ERR_LEN>(STATE_ACCELERATION_ERR_START);

  VectorXd angVelocityDevice = predicted_state.segment<STATE_ANGULAR_VELOCITY_LEN>(STATE_ANGULAR_VELOCITY_START);
  VectorXd angVelocityDeviceErr = predicted_std.segment<STATE_ANGULAR_VELOCITY_ERR_LEN>(STATE_ANGULAR_VELOCITY_ERR_START);

  Vector3d nans = Vector3d(NAN, NAN, NAN);

  // TODO fill in NED and Calibrated stds
  // write measurements to msg
  init_measurement(fix.initPositionGeodetic(), fix_pos_geo_vec, nans, this->gps_mode);
  init_measurement(fix.initPositionECEF(), fix_ecef, fix_ecef_std, this->gps_mode);
  init_measurement(fix.initVelocityECEF(), vel_ecef, vel_ecef_std, this->gps_mode);
  init_measurement(fix.initVelocityNED(), ned_vel, nans, this->gps_mode);
  init_measurement(fix.initVelocityDevice(), vel_device, vel_device_std, true);
  init_measurement(fix.initAccelerationDevice(), accDevice, accDeviceErr, true);
  init_measurement(fix.initOrientationECEF(), orientation_ecef, orientation_ecef_std, this->gps_mode);
  init_measurement(fix.initCalibratedOrientationECEF(), calibrated_orientation_ecef, nans, this->calibrated && this->gps_mode);
  init_measurement(fix.initOrientationNED(), orientation_ned, orientation_ned_std, this->gps_mode);
  init_measurement(fix.initCalibratedOrientationNED(), calibrated_orientation_ned, nans, this->calibrated && this->gps_mode);
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
  fix.setExcessiveResets(this->reset_tracker > MAX_RESET_TRACKER);
  this->device_fell = false;

  //fix.setGpsWeek(this->time.week);
  //fix.setGpsTimeOfWeek(this->time.tow);
  fix.setUnixTimestampMillis(this->unix_timestamp_millis);

  double time_since_reset = this->kf->get_filter_time() - this->last_reset_time;
  fix.setTimeSinceReset(time_since_reset);
  if (fix_ecef_std.norm() < VALID_POS_STD && this->calibrated && time_since_reset > VALID_TIME_SINCE_RESET) {
    fix.setStatus(cereal::LiveLocationKalman::Status::VALID);
  } else if (fix_ecef_std.norm() < VALID_POS_STD && time_since_reset > VALID_TIME_SINCE_RESET) {
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

VectorXd Localizer::get_state() {
  return this->kf->get_x();
}

VectorXd Localizer::get_stdev() {
  return this->kf->get_P().diagonal().array().sqrt();
}

void Localizer::handle_sensors(double current_time, const capnp::List<cereal::SensorEventData, capnp::Kind::STRUCT>::Reader& log) {
  // TODO does not yet account for double sensor readings in the log
  for (int i = 0; i < log.size(); i++) {
    const cereal::SensorEventData::Reader& sensor_reading = log[i];

    // Ignore empty readings (e.g. in case the magnetometer had no data ready)
    if (sensor_reading.getTimestamp() == 0) {
      continue;
    }

    double sensor_time = 1e-9 * sensor_reading.getTimestamp();

    // sensor time and log time should be close
    if (std::abs(current_time - sensor_time) > 0.1) {
      LOGE("Sensor reading ignored, sensor timestamp more than 100ms off from log time");
      return;
    }

      // TODO: handle messages from two IMUs at the same time
    if (sensor_reading.getSource() == cereal::SensorEventData::SensorSource::BMX055) {
      continue;
    }

    // Gyro Uncalibrated
    if (sensor_reading.getSensor() == SENSOR_GYRO_UNCALIBRATED && sensor_reading.getType() == SENSOR_TYPE_GYROSCOPE_UNCALIBRATED) {
      auto v = sensor_reading.getGyroUncalibrated().getV();
      auto meas = Vector3d(-v[2], -v[1], -v[0]);
      if (meas.norm() < ROTATION_SANITY_CHECK) {
        this->kf->predict_and_observe(sensor_time, OBSERVATION_PHONE_GYRO, { meas });
      }
    }

    // Accelerometer
    if (sensor_reading.getSensor() == SENSOR_ACCELEROMETER && sensor_reading.getType() == SENSOR_TYPE_ACCELEROMETER) {
      auto v = sensor_reading.getAcceleration().getV();

      // TODO: reduce false positives and re-enable this check
      // check if device fell, estimate 10 for g
      // 40m/s**2 is a good filter for falling detection, no false positives in 20k minutes of driving
      //this->device_fell |= (floatlist2vector(v) - Vector3d(10.0, 0.0, 0.0)).norm() > 40.0;

      auto meas = Vector3d(-v[2], -v[1], -v[0]);
      if (meas.norm() < ACCEL_SANITY_CHECK) {
        this->kf->predict_and_observe(sensor_time, OBSERVATION_PHONE_ACCEL, { meas });
      }
    }
  }
}

void Localizer::input_fake_gps_observations(double current_time) {
  // This is done to make sure that the error estimate of the position does not blow up
  // when the filter is in no-gps mode
  // Steps : first predict -> observe current obs with reasonable STD
  this->kf->predict(current_time);

  VectorXd current_x = this->kf->get_x();  
  VectorXd ecef_pos = current_x.segment<STATE_ECEF_POS_LEN>(STATE_ECEF_POS_START);
  VectorXd ecef_vel = current_x.segment<STATE_ECEF_VELOCITY_LEN>(STATE_ECEF_VELOCITY_START);
  MatrixXdr ecef_pos_R = this->kf->get_fake_gps_pos_cov();
  MatrixXdr ecef_vel_R = this->kf->get_fake_gps_vel_cov();
  
  this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_POS, { ecef_pos }, { ecef_pos_R });
  this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_VEL, { ecef_vel }, { ecef_vel_R });
}

void Localizer::handle_gps(double current_time, const cereal::GpsLocationData::Reader& log) {
  // ignore the message if the fix is invalid
  bool gps_invalid_flag = (log.getFlags() % 2 == 0);
  bool gps_unreasonable = (Vector2d(log.getAccuracy(), log.getVerticalAccuracy()).norm() >= SANE_GPS_UNCERTAINTY);
  bool gps_accuracy_insane = ((log.getVerticalAccuracy() <= 0) || (log.getSpeedAccuracy() <= 0) || (log.getBearingAccuracyDeg() <= 0));
  bool gps_lat_lng_alt_insane = ((std::abs(log.getLatitude()) > 90) || (std::abs(log.getLongitude()) > 180) || (std::abs(log.getAltitude()) > ALTITUDE_SANITY_CHECK));
  bool gps_vel_insane = (floatlist2vector(log.getVNED()).norm() > TRANS_SANITY_CHECK);

  if (gps_invalid_flag || gps_unreasonable || gps_accuracy_insane || gps_lat_lng_alt_insane || gps_vel_insane){
    this->determine_gps_mode(current_time);
    return;
  }

  // Process message
  this->last_gps_fix = current_time;
  this->gps_mode = true;
  Geodetic geodetic = { log.getLatitude(), log.getLongitude(), log.getAltitude() };
  this->converter = std::make_unique<LocalCoord>(geodetic);

  VectorXd ecef_pos = this->converter->ned2ecef({ 0.0, 0.0, 0.0 }).to_vector();
  VectorXd ecef_vel = this->converter->ned2ecef({ log.getVNED()[0], log.getVNED()[1], log.getVNED()[2] }).to_vector() - ecef_pos;
  MatrixXdr ecef_pos_R = Vector3d::Constant(std::pow(10.0 * log.getAccuracy(),2) + std::pow(10.0 * log.getVerticalAccuracy(),2)).asDiagonal();
  MatrixXdr ecef_vel_R = Vector3d::Constant(std::pow(log.getSpeedAccuracy() * 10.0, 2)).asDiagonal();
  
  this->unix_timestamp_millis = log.getTimestamp();
  double gps_est_error = (this->kf->get_x().segment<STATE_ECEF_POS_LEN>(STATE_ECEF_POS_START) - ecef_pos).norm();

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
    this->reset_kalman(NAN, initial_pose_ecef_quat, ecef_pos, ecef_vel, ecef_pos_R, ecef_vel_R);
    this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_ORIENTATION_FROM_GPS, { initial_pose_ecef_quat });
  } else if (gps_est_error > 100.0) {
    LOGE("Locationd vs ubloxLocation position difference too large, kalman reset");
    this->reset_kalman(NAN, initial_pose_ecef_quat, ecef_pos, ecef_vel, ecef_pos_R, ecef_vel_R);
  }

  this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_POS, { ecef_pos }, { ecef_pos_R });
  this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_VEL, { ecef_vel }, { ecef_vel_R });
}

void Localizer::handle_car_state(double current_time, const cereal::CarState::Reader& log) {
  this->car_speed = std::abs(log.getVEgo());
  if (log.getStandstill()) {
    this->kf->predict_and_observe(current_time, OBSERVATION_NO_ROT, { Vector3d(0.0, 0.0, 0.0) });
    this->kf->predict_and_observe(current_time, OBSERVATION_NO_ACCEL, { Vector3d(0.0, 0.0, 0.0) });
  }
}

void Localizer::handle_cam_odo(double current_time, const cereal::CameraOdometry::Reader& log) {
  VectorXd rot_device = this->device_from_calib * floatlist2vector(log.getRot());
  VectorXd trans_device = this->device_from_calib * floatlist2vector(log.getTrans());

  if ((rot_device.norm() > ROTATION_SANITY_CHECK) || (trans_device.norm() > TRANS_SANITY_CHECK)) {
    return;
  }

  VectorXd rot_calib_std = floatlist2vector(log.getRotStd());
  VectorXd trans_calib_std = floatlist2vector(log.getTransStd());

  if ((rot_calib_std.minCoeff() <= MIN_STD_SANITY_CHECK) || (trans_calib_std.minCoeff() <= MIN_STD_SANITY_CHECK)) {
    return;
  }

  if ((rot_calib_std.norm() > 10 * ROTATION_SANITY_CHECK) || (trans_calib_std.norm() > 10 * TRANS_SANITY_CHECK)) {
    return;
  }

  this->posenet_stds.pop_front();
  this->posenet_stds.push_back(trans_calib_std[0]);

  // Multiply by 10 to avoid to high certainty in kalman filter because of temporally correlated noise
  trans_calib_std *= 10.0;
  rot_calib_std *= 10.0;
  MatrixXdr rot_device_cov = rotate_std(this->device_from_calib, rot_calib_std).array().square().matrix().asDiagonal();
  MatrixXdr trans_device_cov = rotate_std(this->device_from_calib, trans_calib_std).array().square().matrix().asDiagonal();
  this->kf->predict_and_observe(current_time, OBSERVATION_CAMERA_ODO_ROTATION,
    { rot_device }, { rot_device_cov });
  this->kf->predict_and_observe(current_time, OBSERVATION_CAMERA_ODO_TRANSLATION,
    { trans_device }, { trans_device_cov });
}

void Localizer::handle_live_calib(double current_time, const cereal::LiveCalibrationData::Reader& log) {
  if (log.getRpyCalib().size() > 0) {
    auto live_calib = floatlist2vector(log.getRpyCalib());
    if ((live_calib.minCoeff() < -CALIB_RPY_SANITY_CHECK) || (live_calib.maxCoeff() > CALIB_RPY_SANITY_CHECK)) {
      return;
    }

    this->calib = live_calib;
    this->device_from_calib = euler2rot(this->calib);
    this->calib_from_device = this->device_from_calib.transpose();
    this->calibrated = log.getCalStatus() == 1;
  }
}

void Localizer::reset_kalman(double current_time) {
  VectorXd init_x = this->kf->get_initial_x();
  MatrixXdr init_P = this->kf->get_initial_P();
  this->reset_kalman(current_time, init_x, init_P);
}

void Localizer::finite_check(double current_time) {
  bool all_finite = this->kf->get_x().array().isFinite().all() or this->kf->get_P().array().isFinite().all();
  if (!all_finite) {
    LOGE("Non-finite values detected, kalman reset");
    this->reset_kalman(current_time);
  }
}

void Localizer::time_check(double current_time) {
  if (std::isnan(this->last_reset_time)) {
    this->last_reset_time = current_time;
  }
  double filter_time = this->kf->get_filter_time();
  bool big_time_gap = !std::isnan(filter_time) && (current_time - filter_time > 10);
  if (big_time_gap) {
    LOGE("Time gap of over 10s detected, kalman reset");
    this->reset_kalman(current_time);
  }
}

void Localizer::update_reset_tracker() {
  // reset tracker is tuned to trigger when over 1reset/10s over 2min period
  if (this->isGpsOK()) {
    this->reset_tracker *= .99995;
  } else {
    this->reset_tracker = 0.0;
  }
}

void Localizer::reset_kalman(double current_time, VectorXd init_orient, VectorXd init_pos, VectorXd init_vel, MatrixXdr init_pos_R, MatrixXdr init_vel_R) {
  // too nonlinear to init on completely wrong
  VectorXd current_x = this->kf->get_x();
  MatrixXdr current_P = this->kf->get_P();
  MatrixXdr init_P = this->kf->get_initial_P();
  MatrixXdr reset_orientation_P = this->kf->get_reset_orientation_P();
  int non_ecef_state_err_len = init_P.rows() - (STATE_ECEF_POS_ERR_LEN + STATE_ECEF_ORIENTATION_ERR_LEN + STATE_ECEF_VELOCITY_ERR_LEN);

  current_x.segment<STATE_ECEF_ORIENTATION_LEN>(STATE_ECEF_ORIENTATION_START) = init_orient;
  current_x.segment<STATE_ECEF_VELOCITY_LEN>(STATE_ECEF_VELOCITY_START) = init_vel;
  current_x.segment<STATE_ECEF_POS_LEN>(STATE_ECEF_POS_START) = init_pos;

  init_P.block<STATE_ECEF_POS_ERR_LEN, STATE_ECEF_POS_ERR_LEN>(STATE_ECEF_POS_ERR_START, STATE_ECEF_POS_ERR_START).diagonal() = init_pos_R.diagonal();
  init_P.block<STATE_ECEF_ORIENTATION_ERR_LEN, STATE_ECEF_ORIENTATION_ERR_LEN>(STATE_ECEF_ORIENTATION_ERR_START, STATE_ECEF_ORIENTATION_ERR_START).diagonal() = reset_orientation_P.diagonal();
  init_P.block<STATE_ECEF_VELOCITY_ERR_LEN, STATE_ECEF_VELOCITY_ERR_LEN>(STATE_ECEF_VELOCITY_ERR_START, STATE_ECEF_VELOCITY_ERR_START).diagonal() = init_vel_R.diagonal();
  init_P.block(STATE_ANGULAR_VELOCITY_ERR_START, STATE_ANGULAR_VELOCITY_ERR_START, non_ecef_state_err_len, non_ecef_state_err_len).diagonal() = current_P.block(STATE_ANGULAR_VELOCITY_ERR_START, STATE_ANGULAR_VELOCITY_ERR_START, non_ecef_state_err_len, non_ecef_state_err_len).diagonal();
  
  this->reset_kalman(current_time, current_x, init_P);
}

void Localizer::reset_kalman(double current_time, VectorXd init_x, MatrixXdr init_P) {
  this->kf->init_state(init_x, init_P, current_time);
  this->last_reset_time = current_time;
  this->reset_tracker += 1.0;
}

void Localizer::handle_msg_bytes(const char *data, const size_t size) {
  AlignedBuffer aligned_buf;

  capnp::FlatArrayMessageReader cmsg(aligned_buf.align(data, size));
  cereal::Event::Reader event = cmsg.getRoot<cereal::Event>();

  this->handle_msg(event);
}

void Localizer::handle_msg(const cereal::Event::Reader& log) {
  double t = log.getLogMonoTime() * 1e-9;
  this->time_check(t);
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
  this->finite_check();
  this->update_reset_tracker();
}

kj::ArrayPtr<capnp::byte> Localizer::get_message_bytes(MessageBuilder& msg_builder, bool inputsOK,
                                                       bool sensorsOK, bool gpsOK, bool msgValid) {
  cereal::Event::Builder evt = msg_builder.initEvent();
  evt.setValid(msgValid);
  cereal::LiveLocationKalman::Builder liveLoc = evt.initLiveLocationKalman();
  this->build_live_location(liveLoc);
  liveLoc.setSensorsOK(sensorsOK);
  liveLoc.setGpsOK(gpsOK);
  liveLoc.setInputsOK(inputsOK);
  return msg_builder.toBytes();
}


bool Localizer::isGpsOK() {
  return this->kf->get_filter_time() - this->last_gps_fix < 1.0;
}

void Localizer::determine_gps_mode(double current_time) {
  // 1. If the pos_std is greater than what's not acceptible and localizer is in gps-mode, reset to no-gps-mode
  // 2. If the pos_std is greater than what's not acceptible and localizer is in no-gps-mode, fake obs
  // 3. If the pos_std is smaller than what's not acceptible, let gps-mode be whatever it is
  VectorXd current_pos_std = this->kf->get_P().block<STATE_ECEF_POS_ERR_LEN, STATE_ECEF_POS_ERR_LEN>(STATE_ECEF_POS_ERR_START, STATE_ECEF_POS_ERR_START).diagonal().array().sqrt();
  if (current_pos_std.norm() > SANE_GPS_UNCERTAINTY){
    if (this->gps_mode){
      this->gps_mode = false;
      this->reset_kalman(current_time);
    }
    else{
      this->input_fake_gps_observations(current_time);
    }
  }
}

int Localizer::locationd_thread() {
  const std::initializer_list<const char *> service_list = {"gpsLocationExternal", "sensorEvents", "cameraOdometry", "liveCalibration", "carState", "carParams"};
  PubMaster pm({"liveLocationKalman"});

  // TODO: remove carParams once we're always sending at 100Hz
  SubMaster sm(service_list, nullptr, {"gpsLocationExternal", "carParams"});

  uint64_t cnt = 0;
  bool filterInitialized = false;

  while (!do_exit) {
    sm.update();
    if (filterInitialized){
      for (const char* service : service_list) {
        if (sm.updated(service) && sm.valid(service)){
          const cereal::Event::Reader log = sm[service];
          this->handle_msg(log);
        }
      }
    } else {
      filterInitialized = sm.allAliveAndValid();
    }

    // 100Hz publish for notcars, 20Hz for cars
    const char* trigger_msg = sm["carParams"].getCarParams().getNotCar() ? "sensorEvents" : "cameraOdometry";
    if (sm.updated(trigger_msg)) {
      bool inputsOK = sm.allAliveAndValid();
      bool sensorsOK = sm.alive("sensorEvents") && sm.valid("sensorEvents");
      bool gpsOK = this->isGpsOK();

      MessageBuilder msg_builder;
      kj::ArrayPtr<capnp::byte> bytes = this->get_message_bytes(msg_builder, inputsOK, sensorsOK, gpsOK, filterInitialized);
      pm.send("liveLocationKalman", bytes.begin(), bytes.size());

      if (cnt % 1200 == 0 && gpsOK) {  // once a minute
        VectorXd posGeo = this->get_position_geodetic();
        std::string lastGPSPosJSON = util::string_format(
          "{\"latitude\": %.15f, \"longitude\": %.15f, \"altitude\": %.15f}", posGeo(0), posGeo(1), posGeo(2));

        std::thread([] (const std::string gpsjson) {
          Params().put("LastGPSPosition", gpsjson);
        }, lastGPSPosJSON).detach();
      }
      cnt++;
    }
  }
  return 0;
}

int main() {
  util::set_realtime_priority(5);

  Localizer localizer;
  return localizer.locationd_thread();
}
