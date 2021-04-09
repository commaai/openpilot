#include "locationd.h"

using namespace EKFS;
using namespace Eigen;

Localizer::Localizer() {
  this->kf = std::make_shared<LiveKalman>();
  this->reset_kalman();

  this->calib = Eigen::VectorXd(3);
  this->calib << 0.0, 0.0, 0.0;
  this->device_from_calib = MatrixXdr::Identity(3, 3);
  this->calib_from_device = MatrixXdr::Identity(3, 3);

//     self.H = get_H()
//     self.converter = coord.LocalCoord.from_ecef(self.kf.x[States.ECEF_POS])
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

void Localizer::update_kalman(double t, int kind, std::vector<Eigen::VectorXd> meas, std::vector<MatrixXdr> R) {
  try {
    this->kf->predict_and_observe(t, kind, meas, R);
  }
  catch (std::exception e) {  // TODO specify exception
    std::cout << "Error in predict and observe, kalman reset" << std::endl;  // TODO cloudlog
    this->reset_kalman();
  }
}

void Localizer::handle_event(double current_time, cereal::Event::Reader& event) {

}

//   def handle_gps(self, current_time, log):
//     # ignore the message if the fix is invalid
//     if log.flags % 2 == 0:
//       return

//     self.last_gps_fix = current_time

//     self.converter = coord.LocalCoord.from_geodetic([log.latitude, log.longitude, log.altitude])
//     ecef_pos = self.converter.ned2ecef([0, 0, 0])
//     ecef_vel = self.converter.ned2ecef(np.array(log.vNED)) - ecef_pos
//     ecef_pos_R = np.diag([(3*log.verticalAccuracy)**2]*3)
//     ecef_vel_R = np.diag([(log.speedAccuracy)**2]*3)

//     #self.time = GPSTime.from_datetime(datetime.utcfromtimestamp(log.timestamp*1e-3))
//     self.unix_timestamp_millis = log.timestamp
//     gps_est_error = np.sqrt((self.kf.x[0] - ecef_pos[0])**2 +
//                             (self.kf.x[1] - ecef_pos[1])**2 +
//                             (self.kf.x[2] - ecef_pos[2])**2)

//     orientation_ecef = euler_from_quat(self.kf.x[States.ECEF_ORIENTATION])
//     orientation_ned = ned_euler_from_ecef(ecef_pos, orientation_ecef)
//     orientation_ned_gps = np.array([0, 0, np.radians(log.bearingDeg)])
//     orientation_error = np.mod(orientation_ned - orientation_ned_gps - np.pi, 2*np.pi) - np.pi
//     initial_pose_ecef_quat = quat_from_euler(ecef_euler_from_ned(ecef_pos, orientation_ned_gps))
//     if np.linalg.norm(ecef_vel) > 5 and np.linalg.norm(orientation_error) > 1:
//       cloudlog.error("Locationd vs ubloxLocation orientation difference too large, kalman reset")
//       self.reset_kalman(init_pos=ecef_pos, init_orient=initial_pose_ecef_quat)
//       self.update_kalman(current_time, ObservationKind.ECEF_ORIENTATION_FROM_GPS, initial_pose_ecef_quat)
//     elif gps_est_error > 50:
//       cloudlog.error("Locationd vs ubloxLocation position difference too large, kalman reset")
//       self.reset_kalman(init_pos=ecef_pos, init_orient=initial_pose_ecef_quat)

//     self.update_kalman(current_time, ObservationKind.ECEF_POS, ecef_pos, R=ecef_pos_R)
//     self.update_kalman(current_time, ObservationKind.ECEF_VEL, ecef_vel, R=ecef_vel_R)

//   def handle_car_state(self, current_time, log):
//     self.speed_counter += 1

//     if self.speed_counter % SENSOR_DECIMATION == 0:
//       self.update_kalman(current_time, ObservationKind.ODOMETRIC_SPEED, [log.vEgo])
//       self.car_speed = abs(log.vEgo)
//       if log.vEgo == 0:
//         self.update_kalman(current_time, ObservationKind.NO_ROT, [0, 0, 0])

//   def handle_cam_odo(self, current_time, log):
//     self.cam_counter += 1

//     if self.cam_counter % VISION_DECIMATION == 0:
//       rot_device = self.device_from_calib.dot(log.rot)
//       rot_device_std = self.device_from_calib.dot(log.rotStd)
//       self.update_kalman(current_time,
//                          ObservationKind.CAMERA_ODO_ROTATION,
//                          np.concatenate([rot_device, 10*rot_device_std]))
//       trans_device = self.device_from_calib.dot(log.trans)
//       trans_device_std = self.device_from_calib.dot(log.transStd)
//       self.posenet_speed = np.linalg.norm(trans_device)
//       self.posenet_stds[:-1] = self.posenet_stds[1:]
//       self.posenet_stds[-1] = trans_device_std[0]
//       self.update_kalman(current_time,
//                          ObservationKind.CAMERA_ODO_TRANSLATION,
//                          np.concatenate([trans_device, 10*trans_device_std]))

//   def handle_sensors(self, current_time, log):
//     # TODO does not yet account for double sensor readings in the log
//     for sensor_reading in log:
//       sensor_time = 1e-9 * sensor_reading.timestamp
//       # TODO: handle messages from two IMUs at the same time
//       if sensor_reading.source == SensorSource.lsm6ds3:
//         continue

//       # Gyro Uncalibrated
//       if sensor_reading.sensor == 5 and sensor_reading.type == 16:
//         self.gyro_counter += 1
//         if self.gyro_counter % SENSOR_DECIMATION == 0:
//           v = sensor_reading.gyroUncalibrated.v
//           self.update_kalman(sensor_time, ObservationKind.PHONE_GYRO, [-v[2], -v[1], -v[0]])

//       # Accelerometer
//       if sensor_reading.sensor == 1 and sensor_reading.type == 1:
//         # check if device fell, estimate 10 for g
//         # 40m/s**2 is a good filter for falling detection, no false positives in 20k minutes of driving
//         self.device_fell = self.device_fell or (np.linalg.norm(np.array(sensor_reading.acceleration.v) - np.array([10, 0, 0])) > 40)

//         self.acc_counter += 1
//         if self.acc_counter % SENSOR_DECIMATION == 0:
//           v = sensor_reading.acceleration.v
//           self.update_kalman(sensor_time, ObservationKind.PHONE_ACCEL, [-v[2], -v[1], -v[0]])

//   def handle_live_calib(self, current_time, log):
//     if len(log.rpyCalib):
//       self.calib = log.rpyCalib
//       self.device_from_calib = rot_from_euler(self.calib)
//       self.calib_from_device = self.device_from_calib.T
//       self.calibrated = log.calStatus == 1

void Localizer::reset_kalman(double current_time) {  // TODO nan ?
  VectorXd init_x = this->kf->get_initial_x();
  this->reset_kalman(current_time, init_x.segment<4>(3), init_x.head(3));
}

void Localizer::reset_kalman(double current_time, Eigen::VectorXd init_orient, Eigen::VectorXd init_pos) {
  // too nonlinear to init on completely wrong
  VectorXd init_x = this->kf->get_initial_x();
  VectorXd init_P = this->kf->get_initial_P();
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

  while (true) {
    sm.update();
    for (const char* service : service_list) {
      if (sm.updated(service) && sm.valid(service)) {
        this->handle_event(sm.rcv_time(service) * 1e-9, sm[service]);  // TODO rcv_frame?
      }
    }

    if (sm.updated("cameraOdometry")) {
      double t = sm.rcv_time("cameraOdometry") * 1e-9;  // TODO rcv_frame?

      MessageBuilder msg_builder;
      auto evnt = msg_builder.initEvent();
      evnt.setLogMonoTime(t);
      auto liveLoc = msg_builder.initEvent().initLiveLocationKalman();
      //liveLoc.setLiveLocationKalman(this->liveLocationMsg());
      liveLoc.setInputsOK(sm.allAliveAndValid());
      liveLoc.setSensorsOK(sm.alive("sensorEvents") && sm.valid("sensorEvents"));
      liveLoc.setGpsOK((t / 1e9) - this->last_gps_fix < 1.0);
      pm.send("liveLocationKalman", msg_builder);

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
}

int main() {
  Localizer localizer;
  return localizer.locationd_thread();
}
