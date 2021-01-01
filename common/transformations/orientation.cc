#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "orientation.hpp"
#include "coordinates.hpp"

Eigen::Quaterniond ensure_unique(Eigen::Quaterniond quat){
  if (quat.w() > 0){
    return quat;
  } else {
    return Eigen::Quaterniond(-quat.w(), -quat.x(), -quat.y(), -quat.z());
  }
}

Eigen::Quaterniond euler2quat(Eigen::Vector3d euler){
  Eigen::Quaterniond q;

  q = Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitZ())
    * Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitX());
  return ensure_unique(q);
}


Eigen::Vector3d quat2euler(Eigen::Quaterniond quat){
  // TODO: switch to eigen implementation if the range of the Euler angles doesn't matter anymore
  // Eigen::Vector3d euler = quat.toRotationMatrix().eulerAngles(2, 1, 0);
  // return {euler(2), euler(1), euler(0)};
  double gamma = atan2(2 * (quat.w() * quat.x() + quat.y() * quat.z()), 1 - 2 * (quat.x()*quat.x() + quat.y()*quat.y()));
  double theta = asin(2 * (quat.w() * quat.y() - quat.z() * quat.x()));
  double psi = atan2(2 * (quat.w() * quat.z() + quat.x() * quat.y()), 1 - 2 * (quat.y()*quat.y() + quat.z()*quat.z()));
  return {gamma, theta, psi};
}

Eigen::Matrix3d quat2rot(Eigen::Quaterniond quat){
  return quat.toRotationMatrix();
}

Eigen::Quaterniond rot2quat(const Eigen::Matrix3d &rot){
  return ensure_unique(Eigen::Quaterniond(rot));
}

Eigen::Matrix3d euler2rot(Eigen::Vector3d euler){
  return quat2rot(euler2quat(euler));
}

Eigen::Vector3d rot2euler(const Eigen::Matrix3d &rot){
  return quat2euler(rot2quat(rot));
}

Eigen::Matrix3d rot_matrix(double roll, double pitch, double yaw){
  return euler2rot({roll, pitch, yaw});
}

Eigen::Matrix3d rot(Eigen::Vector3d axis, double angle){
  Eigen::Quaterniond q;
  q = Eigen::AngleAxisd(angle, axis);
  return q.toRotationMatrix();
}


Eigen::Vector3d ecef_euler_from_ned(ECEF ecef_init, Eigen::Vector3d ned_pose) {
  /*
    Using Rotations to Build Aerospace Coordinate Systems
    Don Koks
    https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf
  */
  LocalCoord converter = LocalCoord(ecef_init);
  Eigen::Vector3d zero = ecef_init.to_vector();

  Eigen::Vector3d x0 = converter.ned2ecef({1, 0, 0}).to_vector() - zero;
  Eigen::Vector3d y0 = converter.ned2ecef({0, 1, 0}).to_vector() - zero;
  Eigen::Vector3d z0 = converter.ned2ecef({0, 0, 1}).to_vector() - zero;

  Eigen::Vector3d x1 = rot(z0, ned_pose(2)) * x0;
  Eigen::Vector3d y1 = rot(z0, ned_pose(2)) * y0;
  Eigen::Vector3d z1 = rot(z0, ned_pose(2)) * z0;

  Eigen::Vector3d x2 = rot(y1, ned_pose(1)) * x1;
  Eigen::Vector3d y2 = rot(y1, ned_pose(1)) * y1;
  Eigen::Vector3d z2 = rot(y1, ned_pose(1)) * z1;

  Eigen::Vector3d x3 = rot(x2, ned_pose(0)) * x2;
  Eigen::Vector3d y3 = rot(x2, ned_pose(0)) * y2;


  x0 = Eigen::Vector3d(1, 0, 0);
  y0 = Eigen::Vector3d(0, 1, 0);
  z0 = Eigen::Vector3d(0, 0, 1);

  double psi = atan2(x3.dot(y0), x3.dot(x0));
  double theta = atan2(-x3.dot(z0), sqrt(pow(x3.dot(x0), 2) + pow(x3.dot(y0), 2)));

  y2 = rot(z0, psi) * y0;
  z2 = rot(y2, theta) * z0;

  double phi = atan2(y3.dot(z2), y3.dot(y2));

  return {phi, theta, psi};
}

Eigen::Vector3d ned_euler_from_ecef(ECEF ecef_init, Eigen::Vector3d ecef_pose){
  /*
    Using Rotations to Build Aerospace Coordinate Systems
    Don Koks
    https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf
  */
  LocalCoord converter = LocalCoord(ecef_init);

  Eigen::Vector3d x0 = Eigen::Vector3d(1, 0, 0);
  Eigen::Vector3d y0 = Eigen::Vector3d(0, 1, 0);
  Eigen::Vector3d z0 = Eigen::Vector3d(0, 0, 1);

  Eigen::Vector3d x1 = rot(z0, ecef_pose(2)) * x0;
  Eigen::Vector3d y1 = rot(z0, ecef_pose(2)) * y0;
  Eigen::Vector3d z1 = rot(z0, ecef_pose(2)) * z0;

  Eigen::Vector3d x2 = rot(y1, ecef_pose(1)) * x1;
  Eigen::Vector3d y2 = rot(y1, ecef_pose(1)) * y1;
  Eigen::Vector3d z2 = rot(y1, ecef_pose(1)) * z1;

  Eigen::Vector3d x3 = rot(x2, ecef_pose(0)) * x2;
  Eigen::Vector3d y3 = rot(x2, ecef_pose(0)) * y2;

  Eigen::Vector3d zero = ecef_init.to_vector();
  x0 = converter.ned2ecef({1, 0, 0}).to_vector() - zero;
  y0 = converter.ned2ecef({0, 1, 0}).to_vector() - zero;
  z0 = converter.ned2ecef({0, 0, 1}).to_vector() - zero;

  double psi = atan2(x3.dot(y0), x3.dot(x0));
  double theta = atan2(-x3.dot(z0), sqrt(pow(x3.dot(x0), 2) + pow(x3.dot(y0), 2)));

  y2 = rot(z0, psi) * y0;
  z2 = rot(y2, theta) * z0;

  double phi = atan2(y3.dot(z2), y3.dot(y2));

  return {phi, theta, psi};
}



int main(void){
}
