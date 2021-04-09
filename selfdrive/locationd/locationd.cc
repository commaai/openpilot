#include "locationd.h"

using namespace EKFS;
using namespace Eigen;

ExitHandler do_exit;

VectorXd floatlist_to_vector(const capnp::List<float, capnp::Kind::PRIMITIVE>::Reader& floatlist) {
  VectorXd res(floatlist.size());
  for (int i = 0; i < floatlist.size(); i++) {
    res[i] = floatlist[i];
  }
  return res;
}

VectorXd quat2vector(const Quaterniond& quat) {
  VectorXd res(4);
  res << quat.w(), quat.x(), quat.y(), quat.z();
  return res;
}

Quaterniond vector2quat(const VectorXd& vec) {
  return Quaterniond(vec(0), vec(1), vec(2), vec(3));
}

Localizer::Localizer() {
  this->kf = std::make_shared<LiveKalman>();
  this->reset_kalman();

  this->calib = VectorXd(3);
  this->calib << 0.0, 0.0, 0.0;
  this->device_from_calib = MatrixXdr::Identity(3, 3);
  this->calib_from_device = MatrixXdr::Identity(3, 3);

  this->posenet_stds = VectorXd(POSENET_STD_HIST);
  for (int i = 0; i < POSENET_STD_HIST; i++) {
    this->posenet_stds[i] = 10.0;
  }

//     self.H = get_H()

  VectorXd ecef_pos = this->kf->get_x().segment<STATE_ECEF_POS_END - STATE_ECEF_POS_START>(STATE_ECEF_POS_START);
  ECEF ecef = { ecef_pos[0], ecef_pos[1], ecef_pos[2] };
  this->converter = std::make_shared<LocalCoord>(ecef);
}

//   @staticmethod
//   def msg_from_state(converter, calib_from_device, H, predicted_state, predicted_cov, calibrated):
//     predicted_std = np.sqrt(np.diagonal(predicted_cov))

//     fix_ecef = predicted_state[States.ECEF_POS]
//     fix_ecef_std = predicted_std[States.ECEF_POS_ERR]
//     vel_ecef = predicted_state[States.ECEF_VELOCITY]
//     vel_ecef_std = predicted_std[States.ECEF_VELOCITY_ERR]
//     fix_pos_geo = coord.ecef2geodetic(fix_ecef)
//     #fix_pos_geo_std = np.abs(coord.ecef2geodetic(fix_ecef + fix_ecef_std) - fix_pos_geo)
//     orientation_ecef = euler_from_quat(predicted_state[States.ECEF_ORIENTATION])
//     orientation_ecef_std = predicted_std[States.ECEF_ORIENTATION_ERR]
//     device_from_ecef = rot_from_quat(predicted_state[States.ECEF_ORIENTATION]).T
//     calibrated_orientation_ecef = euler_from_rot(calib_from_device.dot(device_from_ecef))

//     acc_calib = calib_from_device.dot(predicted_state[States.ACCELERATION])
//     acc_calib_std = np.sqrt(np.diagonal(calib_from_device.dot(
//       predicted_cov[States.ACCELERATION_ERR, States.ACCELERATION_ERR]).dot(
//         calib_from_device.T)))
//     ang_vel_calib = calib_from_device.dot(predicted_state[States.ANGULAR_VELOCITY])
//     ang_vel_calib_std = np.sqrt(np.diagonal(calib_from_device.dot(
//       predicted_cov[States.ANGULAR_VELOCITY_ERR, States.ANGULAR_VELOCITY_ERR]).dot(
//         calib_from_device.T)))

//     vel_device = device_from_ecef.dot(vel_ecef)
//     device_from_ecef_eul = euler_from_quat(predicted_state[States.ECEF_ORIENTATION]).T
//     idxs = list(range(States.ECEF_ORIENTATION_ERR.start, States.ECEF_ORIENTATION_ERR.stop)) + \
//            list(range(States.ECEF_VELOCITY_ERR.start, States.ECEF_VELOCITY_ERR.stop))
//     condensed_cov = predicted_cov[idxs][:, idxs]
//     HH = H(*list(np.concatenate([device_from_ecef_eul, vel_ecef])))
//     vel_device_cov = HH.dot(condensed_cov).dot(HH.T)
//     vel_device_std = np.sqrt(np.diagonal(vel_device_cov))

//     vel_calib = calib_from_device.dot(vel_device)
//     vel_calib_std = np.sqrt(np.diagonal(calib_from_device.dot(
//       vel_device_cov).dot(calib_from_device.T)))

//     orientation_ned = ned_euler_from_ecef(fix_ecef, orientation_ecef)
//     #orientation_ned_std = ned_euler_from_ecef(fix_ecef, orientation_ecef + orientation_ecef_std) - orientation_ned
//     ned_vel = converter.ecef2ned(fix_ecef + vel_ecef) - converter.ecef2ned(fix_ecef)
//     #ned_vel_std = self.converter.ecef2ned(fix_ecef + vel_ecef + vel_ecef_std) - self.converter.ecef2ned(fix_ecef + vel_ecef)

//     fix = messaging.log.LiveLocationKalman.new_message()

//     # write measurements to msg
//     measurements = [
//       # measurement field, value, std, valid
//       (fix.positionGeodetic, fix_pos_geo, np.nan*np.zeros(3), True),
//       (fix.positionECEF, fix_ecef, fix_ecef_std, True),
//       (fix.velocityECEF, vel_ecef, vel_ecef_std, True),
//       (fix.velocityNED, ned_vel, np.nan*np.zeros(3), True),
//       (fix.velocityDevice, vel_device, vel_device_std, True),
//       (fix.accelerationDevice, predicted_state[States.ACCELERATION], predicted_std[States.ACCELERATION_ERR], True),
//       (fix.orientationECEF, orientation_ecef, orientation_ecef_std, True),
//       (fix.calibratedOrientationECEF, calibrated_orientation_ecef, np.nan*np.zeros(3), calibrated),
//       (fix.orientationNED, orientation_ned, np.nan*np.zeros(3), True),
//       (fix.angularVelocityDevice, predicted_state[States.ANGULAR_VELOCITY], predicted_std[States.ANGULAR_VELOCITY_ERR], True),
//       (fix.velocityCalibrated, vel_calib, vel_calib_std, calibrated),
//       (fix.angularVelocityCalibrated, ang_vel_calib, ang_vel_calib_std, calibrated),
//       (fix.accelerationCalibrated, acc_calib, acc_calib_std, calibrated),
//     ]

//     for field, value, std, valid in measurements:
//       # TODO: can we write the lists faster?
//       field.value = to_float(value)
//       field.std = to_float(std)
//       field.valid = valid

//     return fix

//   def liveLocationMsg(self):
//     fix = self.msg_from_state(self.converter, self.calib_from_device, self.H, self.kf.x, self.kf.P, self.calibrated)
//     # experimentally found these values, no false positives in 20k minutes of driving
//     old_mean, new_mean = np.mean(self.posenet_stds[:POSENET_STD_HIST//2]), np.mean(self.posenet_stds[POSENET_STD_HIST//2:])
//     std_spike = new_mean/old_mean > 4 and new_mean > 7

//     fix.posenetOK = not (std_spike and self.car_speed > 5)
//     fix.deviceStable = not self.device_fell
//     self.device_fell = False

//     #fix.gpsWeek = self.time.week
//     #fix.gpsTimeOfWeek = self.time.tow
//     fix.unixTimestampMillis = self.unix_timestamp_millis

//     if np.linalg.norm(fix.positionECEF.std) < 50 and self.calibrated:
//       fix.status = 'valid'
//     elif np.linalg.norm(fix.positionECEF.std) < 50:
//       fix.status = 'uncalibrated'
//     else:
//       fix.status = 'uninitialized'
//     return fix

void Localizer::update_kalman(double t, int kind, std::vector<VectorXd> meas, std::vector<MatrixXdr> R) {
  try {
    this->kf->predict_and_observe(t, kind, meas, R);
  }
  catch (std::exception e) {  // TODO specify exception
    std::cout << "Error in predict and observe, kalman reset" << std::endl;  // TODO cloudlog
    this->reset_kalman();
  }
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
    if (sensor_reading.getSensor() == 5 && sensor_reading.getType() == 16) {
      this->gyro_counter += 1;
      if (this->gyro_counter % SENSOR_DECIMATION == 0) {
        auto v = sensor_reading.getGyroUncalibrated().getV();
        this->update_kalman(sensor_time, KIND_PHONE_GYRO, { (VectorXd(3) << -v[2], -v[1], -v[0]).finished() });
      }
    }

    // Accelerometer
    if (sensor_reading.getSensor() == 1 && sensor_reading.getType() == 1) {
      auto v = sensor_reading.getAcceleration().getV();

      // check if device fell, estimate 10 for g
      // 40m/s**2 is a good filter for falling detection, no false positives in 20k minutes of driving
      this->device_fell |= (floatlist_to_vector(v) - (VectorXd(3) << 10.0, 0.0, 0.0).finished()).norm() > 40;

      this->acc_counter += 1;
      if (this->acc_counter % SENSOR_DECIMATION == 0) {
        this->update_kalman(sensor_time, KIND_PHONE_ACCEL, { (VectorXd(3) << -v[2], -v[1], -v[0]).finished() });
      }
    }
  }
}

void Localizer::handle_gps(double current_time, const cereal::GpsLocationData::Reader& log) {
  // ignore the message if the fix is invalid
  if (log.getFlags() % 2 == 0) {
    return;
  }

  Quaterniond(Vector4d(1.0, 0.0, 0.0, 0.0));

  this->last_gps_fix = current_time;

  Geodetic geodetic = { log.getLatitude(), log.getLongitude(), log.getAltitude() };
  this->converter = std::make_shared<LocalCoord>(geodetic);

  VectorXd ecef_pos = this->converter->ned2ecef({ 0.0, 0.0, 0.0 }).to_vector();
  VectorXd ecef_vel = this->converter->ned2ecef({ log.getVNED()[0], log.getVNED()[1], log.getVNED()[2] }).to_vector() - ecef_pos;
  double vertical_accuracy = std::pow(3.0 * log.getVerticalAccuracy(), 2);
  MatrixXdr ecef_pos_R = (Vector3d() << vertical_accuracy, vertical_accuracy, vertical_accuracy).finished().asDiagonal();
  double speed_accuracy = std::pow(log.getSpeedAccuracy(), 2);
  MatrixXdr ecef_vel_R = (Vector3d() << speed_accuracy, speed_accuracy, speed_accuracy).finished().asDiagonal();

  this->unix_timestamp_millis = log.getTimestamp();
  double gps_est_error = (this->kf->get_x().head(3) - ecef_pos).norm();

  VectorXd orientation_ecef = quat2euler(vector2quat(this->kf->get_x().segment<STATE_ECEF_ORIENTATION_END - STATE_ECEF_ORIENTATION_START>(STATE_ECEF_ORIENTATION_START)));
  VectorXd orientation_ned = ned_euler_from_ecef({ ecef_pos(0), ecef_pos(1), ecef_pos(2) }, orientation_ecef);
  VectorXd orientation_ned_gps = (VectorXd(3) << 0.0, 0.0, DEG2RAD(log.getBearingDeg())).finished();
  VectorXd orientation_error = (orientation_ned - orientation_ned_gps).array() - M_PI;
  for (int i = 0; i < orientation_error.size(); i++) {
    orientation_error(i) = std::fmod(orientation_error(i), 2.0 * M_PI) - M_PI;
  }
  VectorXd initial_pose_ecef_quat = quat2vector(euler2quat(ecef_euler_from_ned({ ecef_pos(0), ecef_pos(1), ecef_pos(2) }, orientation_ned_gps)));

  if (ecef_vel.norm() > 5.0 && orientation_error.norm() > 1.0) {
    std::cout << "Locationd vs ubloxLocation orientation difference too large, kalman reset" << std::endl;
    this->reset_kalman(NAN, ecef_pos, initial_pose_ecef_quat);
    this->update_kalman(current_time, KIND_ECEF_ORIENTATION_FROM_GPS, { initial_pose_ecef_quat });
  } else if (gps_est_error > 50.0) {
    std::cout << "Locationd vs ubloxLocation position difference too large, kalman reset" << std::endl;
    this->reset_kalman(NAN, ecef_pos, initial_pose_ecef_quat);
  }

  this->update_kalman(current_time, KIND_ECEF_POS, { ecef_pos }, { ecef_pos_R });
  this->update_kalman(current_time, KIND_ECEF_VEL, { ecef_vel }, { ecef_vel_R });
}


void Localizer::handle_car_state(double current_time, const cereal::CarState::Reader& log) {
  this->speed_counter += 1;

  if (this->speed_counter % SENSOR_DECIMATION == 0) {
    this->update_kalman(current_time, KIND_ODOMETRIC_SPEED, { (VectorXd(1) << log.getVEgo()).finished() });
    this->car_speed = abs(log.getVEgo());
    if (log.getVEgo() == 0.0) {
      this->update_kalman(current_time, KIND_NO_ROT, { (VectorXd(3) << 0.0, 0.0, 0.0).finished() });
    }
  }
}

void Localizer::handle_cam_odo(double current_time, const cereal::CameraOdometry::Reader& log) {
  std::cout << "recv cam_odo" << std::endl;
  this->cam_counter += 1;

  if (this->cam_counter % VISION_DECIMATION == 0) {
    // TODO len of vectors is always 3
    VectorXd rot_device = this->device_from_calib * floatlist_to_vector(log.getRot());
    VectorXd rot_device_std = (this->device_from_calib * floatlist_to_vector(log.getRotStd())) * 10.0;
    this->update_kalman(current_time, KIND_CAMERA_ODO_ROTATION,
      { (VectorXd(rot_device.rows() + rot_device_std.rows()) << rot_device, rot_device_std).finished() });

    VectorXd trans_device = this->device_from_calib * floatlist_to_vector(log.getTrans());
    VectorXd trans_device_std = this->device_from_calib * floatlist_to_vector(log.getTransStd());

    this->posenet_stds[this->posenet_stds_i] = trans_device_std[0];
    this->posenet_stds_i = (this->posenet_stds_i + 1) % POSENET_STD_HIST;

    trans_device_std *= 10.0;
    this->update_kalman(current_time, KIND_CAMERA_ODO_TRANSLATION,
      { (VectorXd(trans_device.rows() + trans_device_std.rows()) << trans_device, trans_device_std).finished() });
  }
}

void Localizer::handle_live_calib(double current_time, const cereal::LiveCalibrationData::Reader& log) {
  if (log.getRpyCalib().size() > 0) {
    this->calib = floatlist_to_vector(log.getRpyCalib());
    this->device_from_calib = euler2rot(this->calib);
    this->calib_from_device = this->device_from_calib.transpose();
    this->calibrated = log.getCalStatus() == 1;
  }
}

void Localizer::reset_kalman(double current_time) {  // TODO nan ?
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

int Localizer::locationd_thread() {
  const std::initializer_list<const char *> service_list =
      { "gpsLocationExternal", "sensorEvents", "cameraOdometry", "liveCalibration", "carState" };
  SubMaster sm(service_list, nullptr, { "gpsLocationExternal" });
  PubMaster pm({ "liveLocationKalman" });

  Params params;

  while (!do_exit) {
    bool updatedCameraOdometry = false;
    sm.update(); // TODO timeout?
    for (const char* service : service_list) {
      if (sm.updated(service) && sm.valid(service)) {
        cereal::Event::Reader& log = sm[service];
        double t = sm.rcv_time(service) * 1e-9;

        if (log.isSensorEvents()) {
          this->handle_sensors(t, log.getSensorEvents());
        } else if (log.isGpsLocationExternal()) {
          this->handle_gps(t, log.getGpsLocationExternal());
        } else if (log.isCarState()) {
          this->handle_car_state(t, log.getCarState());
        } else if (log.isCameraOdometry()) {
          this->handle_cam_odo(t, log.getCameraOdometry());
          updatedCameraOdometry = true;
        } else if (log.isLiveCalibration()) {
          this->handle_live_calib(t, log.getLiveCalibration());
        } else {
          std::cout << "invalid event" << std::endl;
        }
      }
    }

    if (updatedCameraOdometry) {
      std::cout << "sending" << std::endl;
      double t = sm.rcv_time("cameraOdometry") * 1e-9;  // TODO rcv_frame?

      MessageBuilder msg_builder;
      auto liveLoc = msg_builder.initEvent().initLiveLocationKalman();
      //liveLoc.setLiveLocationMonoTime(t);
      //liveLoc.setLiveLocationKalman(this->liveLocationMsg());
      liveLoc.setInputsOK(sm.allAliveAndValid());
      liveLoc.setSensorsOK(sm.alive("sensorEvents") && sm.valid("sensorEvents"));
      liveLoc.setGpsOK((t / 1e9) - this->last_gps_fix < 1.0);
      pm.send("liveLocationKalman", msg_builder);
      std::cout << "sent" << std::endl;

// TODO:
//       if sm.frame % 1200 == 0 and msg.liveLocationKalman.gpsOK:  # once a minute
//         location = {
//           'latitude': msg.liveLocationKalman.positionGeodetic.value[0],
//           'longitude': msg.liveLocationKalman.positionGeodetic.value[1],
//           'altitude': msg.liveLocationKalman.positionGeodetic.value[2],
//         }
//         params.put("LastGPSPosition", json.dumps(location))

    }
  }
  return 0;
}

int main() {
  Localizer localizer;
  return localizer.locationd_thread();
}
