#pragma once
#include <eigen3/Eigen/Dense>
#include "coordinates.hpp"


Eigen::Quaterniond ensure_unique(Eigen::Quaterniond quat);

Eigen::Quaterniond euler2quat(Eigen::Vector3d euler);
Eigen::Vector3d quat2euler(Eigen::Quaterniond quat);
Eigen::Matrix3d quat2rot(Eigen::Quaterniond quat);
Eigen::Quaterniond rot2quat(Eigen::Matrix3d rot);
Eigen::Matrix3d euler2rot(Eigen::Vector3d euler);
Eigen::Vector3d rot2euler(Eigen::Matrix3d rot);
Eigen::Matrix3d rot_matrix(double roll, double pitch, double yaw);
Eigen::Matrix3d rot(Eigen::Vector3d axis, double angle);
Eigen::Vector3d ecef_euler_from_ned(ECEF ecef_init, Eigen::Vector3d ned_pose);
Eigen::Vector3d ned_euler_from_ecef(ECEF ecef_init, Eigen::Vector3d ecef_pose);
