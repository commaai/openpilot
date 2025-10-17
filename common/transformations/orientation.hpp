#pragma once
#include <eigen3/Eigen/Dense>
#include "common/transformations/coordinates.hpp"


Eigen::Quaterniond ensure_unique(const Eigen::Quaterniond &quat);

Eigen::Quaterniond euler2quat(const Eigen::Vector3d &euler);
Eigen::Vector3d quat2euler(const Eigen::Quaterniond &quat);
Eigen::Matrix3d quat2rot(const Eigen::Quaterniond &quat);
Eigen::Quaterniond rot2quat(const Eigen::Matrix3d &rot);
Eigen::Matrix3d euler2rot(const Eigen::Vector3d &euler);
Eigen::Vector3d rot2euler(const Eigen::Matrix3d &rot);
Eigen::Matrix3d rot_matrix(double roll, double pitch, double yaw);
Eigen::Matrix3d rot(const Eigen::Vector3d &axis, double angle);
Eigen::Vector3d ecef_euler_from_ned(const ECEF &ecef_init, const Eigen::Vector3d &ned_pose);
Eigen::Vector3d ned_euler_from_ecef(const ECEF &ecef_init, const Eigen::Vector3d &ecef_pose);
