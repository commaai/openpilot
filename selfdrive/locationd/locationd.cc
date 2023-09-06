#include "selfdrive/locationd/locationd.h"

#include <sys/time.h>
#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <vector>

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
const double INPUT_INVALID_THRESHOLD = 0.5; // same as reset tracker
const double RESET_TRACKER_DECAY = 0.99995;
const double DECAY = 0.9993; // ~10 secs to resume after a bad input
const double MAX_FILTER_REWIND_TIME = 0.8; // s
const double YAWRATE_CROSS_ERR_CHECK_FACTOR = 30;

// TODO: GPS sensor time offsets are empirically calculated
// They should be replaced with synced time from a real clock
const double GPS_QUECTEL_SENSOR_TIME_OFFSET = 0.630; // s
const double GPS_UBLOX_SENSOR_TIME_OFFSET = 0.095; // s
const float  GPS_POS_STD_THRESHOLD = 50.0;
const float  GPS_VEL_STD_THRESHOLD = 5.0;
const float  GPS_POS_ERROR_RESET_THRESHOLD = 300.0;
const float  GPS_POS_STD_RESET_THRESHOLD = 2.0;
const float  GPS_VEL_STD_RESET_THRESHOLD = 0.5;
const float  GPS_ORIENTATION_ERROR_RESET_THRESHOLD = 1.0;
const int    GPS_ORIENTATION_ERROR_RESET_CNT = 3;

const bool   DEBUG = getenv("DEBUG") != nullptr && std::string(getenv("DEBUG")) != "0";

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

Localizer::Localizer(LocalizerGnssSource gnss_source) {
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
  this->configure_gnss_source(gnss_source);
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
  if (DEBUG) {
    init_measurement(fix.initFilterState(), predicted_state, predicted_std, true);
  }

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
  fix.setTimeToFirstFix(std::isnan(this->ttff) ? -1. : this->ttff);
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

bool Localizer::are_inputs_ok() {
  return this->critical_services_valid(this->observation_values_invalid) && !this->observation_timings_invalid;
}

void Localizer::observation_timings_invalid_reset(){
  this->observation_timings_invalid = false;
}

void Localizer::handle_sensor(double current_time, const cereal::SensorEventData::Reader& log) {
  // TODO does not yet account for double sensor readings in the log

  // Ignore empty readings (e.g. in case the magnetometer had no data ready)
  if (log.getTimestamp() == 0) {
    return;
  }

  double sensor_time = 1e-9 * log.getTimestamp();

  // sensor time and log time should be close
  if (std::abs(current_time - sensor_time) > 0.1) {
    LOGE("Sensor reading ignored, sensor timestamp more than 100ms off from log time");
    this->observation_timings_invalid = true;
    return;
  } else if (!this->is_timestamp_valid(sensor_time)) {
    this->observation_timings_invalid = true;
    return;
  }

  // TODO: handle messages from two IMUs at the same time
  if (log.getSource() == cereal::SensorEventData::SensorSource::BMX055) {
    return;
  }

  // Gyro Uncalibrated
  if (log.getSensor() == SENSOR_GYRO_UNCALIBRATED && log.getType() == SENSOR_TYPE_GYROSCOPE_UNCALIBRATED) {
    auto v = log.getGyroUncalibrated().getV();
    auto meas = Vector3d(-v[2], -v[1], -v[0]);

    VectorXd gyro_bias = this->kf->get_x().segment<STATE_GYRO_BIAS_LEN>(STATE_GYRO_BIAS_START);
    float gyro_camodo_yawrate_err = std::abs((meas[2] - gyro_bias[2]) - this->camodo_yawrate_distribution[0]);
    float gyro_camodo_yawrate_err_threshold = YAWRATE_CROSS_ERR_CHECK_FACTOR * this->camodo_yawrate_distribution[1];
    bool gyro_valid = gyro_camodo_yawrate_err < gyro_camodo_yawrate_err_threshold;

    if ((meas.norm() < ROTATION_SANITY_CHECK) && gyro_valid) {
      this->kf->predict_and_observe(sensor_time, OBSERVATION_PHONE_GYRO, { meas });
      this->observation_values_invalid["gyroscope"] *= DECAY;
    } else {
      this->observation_values_invalid["gyroscope"] += 1.0;
    }
  }

  // Accelerometer
  if (log.getSensor() == SENSOR_ACCELEROMETER && log.getType() == SENSOR_TYPE_ACCELEROMETER) {
    auto v = log.getAcceleration().getV();

    // TODO: reduce false positives and re-enable this check
    // check if device fell, estimate 10 for g
    // 40m/s**2 is a good filter for falling detection, no false positives in 20k minutes of driving
    // this->device_fell |= (floatlist2vector(v) - Vector3d(10.0, 0.0, 0.0)).norm() > 40.0;

    auto meas = Vector3d(-v[2], -v[1], -v[0]);
    if (meas.norm() < ACCEL_SANITY_CHECK) {
      this->kf->predict_and_observe(sensor_time, OBSERVATION_PHONE_ACCEL, { meas });
      this->observation_values_invalid["accelerometer"] *= DECAY;
    } else {
      this->observation_values_invalid["accelerometer"] += 1.0;
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
  const MatrixXdr &ecef_pos_R = this->kf->get_fake_gps_pos_cov();
  const MatrixXdr &ecef_vel_R = this->kf->get_fake_gps_vel_cov();

  this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_POS, { ecef_pos }, { ecef_pos_R });
  this->kf->predict_and_observe(current_time, OBSERVATION_ECEF_VEL, { ecef_vel }, { ecef_vel_R });
}

void Localizer::handle_gps(double current_time, const cereal::GpsLocationData::Reader& log, const double sensor_time_offset) {
  // ignore the message if the fix is invalid
  bool gps_invalid_flag = (log.getFlags() % 2 == 0);
  bool gps_unreasonable = (Vector2d(log.getAccuracy(), log.getVerticalAccuracy()).norm() >= SANE_GPS_UNCERTAINTY);
  bool gps_accuracy_insane = ((log.getVerticalAccuracy() <= 0) || (log.getSpeedAccuracy() <= 0) || (log.getBearingAccuracyDeg() <= 0));
  bool gps_lat_lng_alt_insane = ((std::abs(log.getLatitude()) > 90) || (std::abs(log.getLongitude()) > 180) || (std::abs(log.getAltitude()) > ALTITUDE_SANITY_CHECK));
  bool gps_vel_insane = (floatlist2vector(log.getVNED()).norm() > TRANS_SANITY_CHECK);

  if (gps_invalid_flag || gps_unreasonable || gps_accuracy_insane || gps_lat_lng_alt_insane || gps_vel_insane) {
    //this->gps_valid = false;
    this->determine_gps_mode(current_time);
    return;
  }

  double sensor_time = current_time - sensor_time_offset;

  // Process message
  //this->gps_valid = true;
  this->gps_mode = true;
  Geodetic geodetic = { log.getLatitude(), log.getLongitude(), log.getAltitude() };
  this->converter = std::make_unique<LocalCoord>(geodetic);

  VectorXd ecef_pos = this->converter->ned2ecef({ 0.0, 0.0, 0.0 }).to_vector();
  VectorXd ecef_vel = this->converter->ned2ecef({ log.getVNED()[0], log.getVNED()[1], log.getVNED()[2] }).to_vector() - ecef_pos;
  float ecef_pos_std = std::sqrt(this->gps_variance_factor * std::pow(log.getAccuracy(), 2) + this->gps_vertical_variance_factor * std::pow(log.getVerticalAccuracy(), 2));
  MatrixXdr ecef_pos_R = Vector3d::Constant(std::pow(this->gps_std_factor * ecef_pos_std, 2)).asDiagonal();
  MatrixXdr ecef_vel_R = Vector3d::Constant(std::pow(this->gps_std_factor * log.getSpeedAccuracy(), 2)).asDiagonal();

  this->unix_timestamp_millis = log.getUnixTimestampMillis();
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
    this->kf->predict_and_observe(sensor_time, OBSERVATION_ECEF_ORIENTATION_FROM_GPS, { initial_pose_ecef_quat });
  } else if (gps_est_error > 100.0) {
    LOGE("Locationd vs ubloxLocation position difference too large, kalman reset");
    this->reset_kalman(NAN, initial_pose_ecef_quat, ecef_pos, ecef_vel, ecef_pos_R, ecef_vel_R);
  }

  this->last_gps_msg = sensor_time;
  this->kf->predict_and_observe(sensor_time, OBSERVATION_ECEF_POS, { ecef_pos }, { ecef_pos_R });
  this->kf->predict_and_observe(sensor_time, OBSERVATION_ECEF_VEL, { ecef_vel }, { ecef_vel_R });
}

void Localizer::handle_gnss(double current_time, const cereal::GnssMeasurements::Reader& log) {

  if (!log.getPositionECEF().getValid() || !log.getVelocityECEF().getValid()) {
    this->determine_gps_mode(current_time);
    return;
  }

  double sensor_time = log.getMeasTime() * 1e-9;
  sensor_time -= this->gps_time_offset;

  auto ecef_pos_v = log.getPositionECEF().getValue();
  VectorXd ecef_pos = Vector3d(ecef_pos_v[0], ecef_pos_v[1], ecef_pos_v[2]);

  // indexed at 0 cause all std values are the same MAE
  auto ecef_pos_std = log.getPositionECEF().getStd()[0];
  MatrixXdr ecef_pos_R = Vector3d::Constant(pow(this->gps_std_factor*ecef_pos_std, 2)).asDiagonal();

  auto ecef_vel_v = log.getVelocityECEF().getValue();
  VectorXd ecef_vel = Vector3d(ecef_vel_v[0], ecef_vel_v[1], ecef_vel_v[2]);

  // indexed at 0 cause all std values are the same MAE
  auto ecef_vel_std = log.getVelocityECEF().getStd()[0];
  MatrixXdr ecef_vel_R = Vector3d::Constant(pow(this->gps_std_factor*ecef_vel_std, 2)).asDiagonal();

  double gps_est_error = (this->kf->get_x().segment<STATE_ECEF_POS_LEN>(STATE_ECEF_POS_START) - ecef_pos).norm();

  VectorXd orientation_ecef = quat2euler(vector2quat(this->kf->get_x().segment<STATE_ECEF_ORIENTATION_LEN>(STATE_ECEF_ORIENTATION_START)));
  VectorXd orientation_ned = ned_euler_from_ecef({ ecef_pos[0], ecef_pos[1], ecef_pos[2] }, orientation_ecef);

  LocalCoord convs((ECEF){ .x = ecef_pos[0], .y = ecef_pos[1], .z = ecef_pos[2] });
  ECEF next_ecef = {.x = ecef_pos[0] + ecef_vel[0], .y = ecef_pos[1] + ecef_vel[1], .z = ecef_pos[2] + ecef_vel[2]};
  VectorXd ned_vel = convs.ecef2ned(next_ecef).to_vector();
  double bearing_rad = atan2(ned_vel[1], ned_vel[0]);

  VectorXd orientation_ned_gps = Vector3d(0.0, 0.0, bearing_rad);
  VectorXd orientation_error = (orientation_ned - orientation_ned_gps).array() - M_PI;
  for (int i = 0; i < orientation_error.size(); i++) {
    orientation_error(i) = std::fmod(orientation_error(i), 2.0 * M_PI);
    if (orientation_error(i) < 0.0) {
      orientation_error(i) += 2.0 * M_PI;
    }
    orientation_error(i) -= M_PI;
  }
  VectorXd initial_pose_ecef_quat = quat2vector(euler2quat(ecef_euler_from_ned({ ecef_pos(0), ecef_pos(1), ecef_pos(2) }, orientation_ned_gps)));

  if (ecef_pos_std > GPS_POS_STD_THRESHOLD || ecef_vel_std > GPS_VEL_STD_THRESHOLD) {
    this->determine_gps_mode(current_time);
    return;
  }

  // prevent jumping gnss measurements (covered lots, standstill...)
  bool orientation_reset = ecef_vel_std < GPS_VEL_STD_RESET_THRESHOLD;
  orientation_reset &= orientation_error.norm() > GPS_ORIENTATION_ERROR_RESET_THRESHOLD;
  orientation_reset &= !this->standstill;
  if (orientation_reset) {
    this->orientation_reset_count++;
  } else {
    this->orientation_reset_count = 0;
  }

  if ((gps_est_error > GPS_POS_ERROR_RESET_THRESHOLD && ecef_pos_std < GPS_POS_STD_RESET_THRESHOLD) || this->last_gps_msg == 0) {
    // always reset on first gps message and if the location is off but the accuracy is high
    LOGE("Locationd vs gnssMeasurement position difference too large, kalman reset");
    this->reset_kalman(NAN, initial_pose_ecef_quat, ecef_pos, ecef_vel, ecef_pos_R, ecef_vel_R);
  } else if (orientation_reset_count > GPS_ORIENTATION_ERROR_RESET_CNT) {
    LOGE("Locationd vs gnssMeasurement orientation difference too large, kalman reset");
    this->reset_kalman(NAN, initial_pose_ecef_quat, ecef_pos, ecef_vel, ecef_pos_R, ecef_vel_R);
    this->kf->predict_and_observe(sensor_time, OBSERVATION_ECEF_ORIENTATION_FROM_GPS, { initial_pose_ecef_quat });
    this->orientation_reset_count = 0;
  }

  this->gps_mode = true;
  this->last_gps_msg = sensor_time;
  this->kf->predict_and_observe(sensor_time, OBSERVATION_ECEF_POS, { ecef_pos }, { ecef_pos_R });
  this->kf->predict_and_observe(sensor_time, OBSERVATION_ECEF_VEL, { ecef_vel }, { ecef_vel_R });
}

void Localizer::handle_car_state(double current_time, const cereal::CarState::Reader& log) {
  this->car_speed = std::abs(log.getVEgo());
  this->standstill = log.getStandstill();
  if (this->standstill) {
    this->kf->predict_and_observe(current_time, OBSERVATION_NO_ROT, { Vector3d(0.0, 0.0, 0.0) });
    this->kf->predict_and_observe(current_time, OBSERVATION_NO_ACCEL, { Vector3d(0.0, 0.0, 0.0) });
  }
}

void Localizer::handle_cam_odo(double current_time, const cereal::CameraOdometry::Reader& log) {
  VectorXd rot_device = this->device_from_calib * floatlist2vector(log.getRot());
  VectorXd trans_device = this->device_from_calib * floatlist2vector(log.getTrans());

  if (!this->is_timestamp_valid(current_time)) {
    this->observation_timings_invalid = true;
    return;
  }

  if ((rot_device.norm() > ROTATION_SANITY_CHECK) || (trans_device.norm() > TRANS_SANITY_CHECK)) {
    this->observation_values_invalid["cameraOdometry"] += 1.0;
    return;
  }

  VectorXd rot_calib_std = floatlist2vector(log.getRotStd());
  VectorXd trans_calib_std = floatlist2vector(log.getTransStd());

  if ((rot_calib_std.minCoeff() <= MIN_STD_SANITY_CHECK) || (trans_calib_std.minCoeff() <= MIN_STD_SANITY_CHECK)) {
    this->observation_values_invalid["cameraOdometry"] += 1.0;
    return;
  }

  if ((rot_calib_std.norm() > 10 * ROTATION_SANITY_CHECK) || (trans_calib_std.norm() > 10 * TRANS_SANITY_CHECK)) {
    this->observation_values_invalid["cameraOdometry"] += 1.0;
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
  this->observation_values_invalid["cameraOdometry"] *= DECAY;
  this->camodo_yawrate_distribution = Vector2d(rot_device[2], rotate_std(this->device_from_calib, rot_calib_std)[2]);
}

void Localizer::handle_live_calib(double current_time, const cereal::LiveCalibrationData::Reader& log) {
  if (!this->is_timestamp_valid(current_time)) {
    this->observation_timings_invalid = true;
    return;
  }

  if (log.getRpyCalib().size() > 0) {
    auto live_calib = floatlist2vector(log.getRpyCalib());
    if ((live_calib.minCoeff() < -CALIB_RPY_SANITY_CHECK) || (live_calib.maxCoeff() > CALIB_RPY_SANITY_CHECK)) {
      this->observation_values_invalid["liveCalibration"] += 1.0;
      return;
    }

    this->calib = live_calib;
    this->device_from_calib = euler2rot(this->calib);
    this->calib_from_device = this->device_from_calib.transpose();
    this->calibrated = log.getCalStatus() == cereal::LiveCalibrationData::Status::CALIBRATED;
    this->observation_values_invalid["liveCalibration"] *= DECAY;
  }
}

void Localizer::reset_kalman(double current_time) {
  const VectorXd &init_x = this->kf->get_initial_x();
  const MatrixXdr &init_P = this->kf->get_initial_P();
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
  if (std::isnan(this->first_valid_log_time)) {
    this->first_valid_log_time = current_time;
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
  if (this->is_gps_ok()) {
    this->reset_tracker *= RESET_TRACKER_DECAY;
  } else {
    this->reset_tracker = 0.0;
  }
}

void Localizer::reset_kalman(double current_time, const VectorXd &init_orient, const VectorXd &init_pos, const VectorXd &init_vel, const MatrixXdr &init_pos_R, const MatrixXdr &init_vel_R) {
  // too nonlinear to init on completely wrong
  VectorXd current_x = this->kf->get_x();
  MatrixXdr current_P = this->kf->get_P();
  MatrixXdr init_P = this->kf->get_initial_P();
  const MatrixXdr &reset_orientation_P = this->kf->get_reset_orientation_P();
  int non_ecef_state_err_len = init_P.rows() - (STATE_ECEF_POS_ERR_LEN + STATE_ECEF_ORIENTATION_ERR_LEN + STATE_ECEF_VELOCITY_ERR_LEN);

  current_x.segment<STATE_ECEF_ORIENTATION_LEN>(STATE_ECEF_ORIENTATION_START) = init_orient;
  current_x.segment<STATE_ECEF_VELOCITY_LEN>(STATE_ECEF_VELOCITY_START) = init_vel;
  current_x.segment<STATE_ECEF_POS_LEN>(STATE_ECEF_POS_START) = init_pos;

  init_P.block<STATE_ECEF_POS_ERR_LEN, STATE_ECEF_POS_ERR_LEN>(STATE_ECEF_POS_ERR_START, STATE_ECEF_POS_ERR_START).diagonal() = init_pos_R.diagonal();
  init_P.block<STATE_ECEF_ORIENTATION_ERR_LEN, STATE_ECEF_ORIENTATION_ERR_LEN>(STATE_ECEF_ORIENTATION_ERR_START, STATE_ECEF_ORIENTATION_ERR_START).diagonal() = reset_orientation_P.diagonal();
  init_P.block<STATE_ECEF_VELOCITY_ERR_LEN, STATE_ECEF_VELOCITY_ERR_LEN>(STATE_ECEF_VELOCITY_ERR_START, STATE_ECEF_VELOCITY_ERR_START).diagonal() = init_vel_R.diagonal();
  init_P.block(STATE_ANGULAR_VELOCITY_ERR_START, STATE_ANGULAR_VELOCITY_ERR_START, non_ecef_state_err_len, non_ecef_state_err_len).diagonal() = current_P.block(STATE_ANGULAR_VELOCITY_ERR_START,
    STATE_ANGULAR_VELOCITY_ERR_START, non_ecef_state_err_len, non_ecef_state_err_len).diagonal();

  this->reset_kalman(current_time, current_x, init_P);
}

void Localizer::reset_kalman(double current_time, const VectorXd &init_x, const MatrixXdr &init_P) {
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
  if (log.isAccelerometer()) {
    this->handle_sensor(t, log.getAccelerometer());
  } else if (log.isGyroscope()) {
    this->handle_sensor(t, log.getGyroscope());
  } else if (log.isGpsLocation()) {
    this->handle_gps(t, log.getGpsLocation(), GPS_QUECTEL_SENSOR_TIME_OFFSET);
  } else if (log.isGpsLocationExternal()) {
    this->handle_gps(t, log.getGpsLocationExternal(), GPS_UBLOX_SENSOR_TIME_OFFSET);
  //} else if (log.isGnssMeasurements()) {
  //  this->handle_gnss(t, log.getGnssMeasurements());
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

bool Localizer::is_gps_ok() {
  return (this->kf->get_filter_time() - this->last_gps_msg) < 2.0;
}

bool Localizer::critical_services_valid(const std::map<std::string, double> &critical_services) {
  for (auto &kv : critical_services){
    if (kv.second >= INPUT_INVALID_THRESHOLD){
      return false;
    }
  }
  return true;
}

bool Localizer::is_timestamp_valid(double current_time) {
  double filter_time = this->kf->get_filter_time();
  if (!std::isnan(filter_time) && ((filter_time - current_time) > MAX_FILTER_REWIND_TIME)) {
    LOGE("Observation timestamp is older than the max rewind threshold of the filter");
    return false;
  }
  return true;
}

void Localizer::determine_gps_mode(double current_time) {
  // 1. If the pos_std is greater than what's not acceptable and localizer is in gps-mode, reset to no-gps-mode
  // 2. If the pos_std is greater than what's not acceptable and localizer is in no-gps-mode, fake obs
  // 3. If the pos_std is smaller than what's not acceptable, let gps-mode be whatever it is
  VectorXd current_pos_std = this->kf->get_P().block<STATE_ECEF_POS_ERR_LEN, STATE_ECEF_POS_ERR_LEN>(STATE_ECEF_POS_ERR_START, STATE_ECEF_POS_ERR_START).diagonal().array().sqrt();
  if (current_pos_std.norm() > SANE_GPS_UNCERTAINTY){
    if (this->gps_mode){
      this->gps_mode = false;
      this->reset_kalman(current_time);
    } else {
      this->input_fake_gps_observations(current_time);
    }
  }
}

void Localizer::configure_gnss_source(const LocalizerGnssSource &source) {
  this->gnss_source = source;
  if (source == LocalizerGnssSource::UBLOX) {
    this->gps_std_factor = 10.0;
    this->gps_variance_factor = 1.0;
    this->gps_vertical_variance_factor = 1.0;
    this->gps_time_offset = GPS_UBLOX_SENSOR_TIME_OFFSET;
  } else {
    this->gps_std_factor = 2.0;
    this->gps_variance_factor = 0.0;
    this->gps_vertical_variance_factor = 3.0;
    this->gps_time_offset = GPS_QUECTEL_SENSOR_TIME_OFFSET;
  }
}

int Localizer::locationd_thread() {
  LocalizerGnssSource source;
  const char* gps_location_socket;
  if (Params().getBool("UbloxAvailable")) {
    source = LocalizerGnssSource::UBLOX;
    gps_location_socket = "gpsLocationExternal";
  } else {
    source = LocalizerGnssSource::QCOM;
    gps_location_socket = "gpsLocation";
  }

  this->configure_gnss_source(source);
  const std::initializer_list<const char *> service_list = {gps_location_socket, "cameraOdometry", "liveCalibration",
                                                          "carState", "carParams", "accelerometer", "gyroscope"};

  // TODO: remove carParams once we're always sending at 100Hz
  SubMaster sm(service_list, {}, nullptr, {gps_location_socket, "carParams"});
  PubMaster pm({"liveLocationKalman"});

  uint64_t cnt = 0;
  bool filterInitialized = false;
  const std::vector<std::string> critical_input_services = {"cameraOdometry", "liveCalibration", "accelerometer", "gyroscope"};
  for (std::string service : critical_input_services) {
    this->observation_values_invalid.insert({service, 0.0});
  }

  while (!do_exit) {
    sm.update();
    if (filterInitialized){
      this->observation_timings_invalid_reset();
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
    const char* trigger_msg = sm["carParams"].getCarParams().getNotCar() ? "accelerometer" : "cameraOdometry";
    if (sm.updated(trigger_msg)) {
      bool inputsOK = sm.allAliveAndValid() && this->are_inputs_ok();
      bool gpsOK = this->is_gps_ok();
      bool sensorsOK = sm.allAliveAndValid({"accelerometer", "gyroscope"});

      // Log time to first fix
      if (gpsOK && std::isnan(this->ttff) && !std::isnan(this->first_valid_log_time)) {
        this->ttff = std::max(1e-3, (sm[trigger_msg].getLogMonoTime() * 1e-9) - this->first_valid_log_time);
      }

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
