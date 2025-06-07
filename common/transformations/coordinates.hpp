#pragma once

#include <eigen3/Eigen/Dense>

#define DEG2RAD(x) ((x) * M_PI / 180.0)
#define RAD2DEG(x) ((x) * 180.0 / M_PI)

struct ECEF {
  double x, y, z;
  Eigen::Vector3d to_vector() const {
    return Eigen::Vector3d(x, y, z);
  }
};

struct NED {
  double n, e, d;
  Eigen::Vector3d to_vector() const {
    return Eigen::Vector3d(n, e, d);
  }
};

struct Geodetic {
  double lat, lon, alt;
  bool radians=false;
};

ECEF geodetic2ecef(const Geodetic &g);
Geodetic ecef2geodetic(const ECEF &e);

class LocalCoord {
public:
  Eigen::Matrix3d ned2ecef_matrix;
  Eigen::Matrix3d ecef2ned_matrix;
  Eigen::Vector3d init_ecef;
  LocalCoord(const Geodetic &g, const ECEF &e);
  LocalCoord(const Geodetic &g) : LocalCoord(g, ::geodetic2ecef(g)) {}
  LocalCoord(const ECEF &e) : LocalCoord(::ecef2geodetic(e), e) {}

  NED ecef2ned(const ECEF &e);
  ECEF ned2ecef(const NED &n);
  NED geodetic2ned(const Geodetic &g);
  Geodetic ned2geodetic(const NED &n);
};
