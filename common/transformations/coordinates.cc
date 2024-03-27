#define _USE_MATH_DEFINES

#include "common/transformations/coordinates.hpp"

#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Dense>

double a = 6378137; // lgtm [cpp/short-global-name]
double b = 6356752.3142; // lgtm [cpp/short-global-name]
double esq = 6.69437999014 * 0.001; // lgtm [cpp/short-global-name]
double e1sq = 6.73949674228 * 0.001;


static Geodetic to_degrees(Geodetic geodetic){
  geodetic.lat = RAD2DEG(geodetic.lat);
  geodetic.lon = RAD2DEG(geodetic.lon);
  return geodetic;
}

static Geodetic to_radians(Geodetic geodetic){
  geodetic.lat = DEG2RAD(geodetic.lat);
  geodetic.lon = DEG2RAD(geodetic.lon);
  return geodetic;
}


ECEF geodetic2ecef(Geodetic g){
  g = to_radians(g);
  double xi = sqrt(1.0 - esq * pow(sin(g.lat), 2));
  double x = (a / xi + g.alt) * cos(g.lat) * cos(g.lon);
  double y = (a / xi + g.alt) * cos(g.lat) * sin(g.lon);
  double z = (a / xi * (1.0 - esq) + g.alt) * sin(g.lat);
  return {x, y, z};
}

Geodetic ecef2geodetic(ECEF e){
  // Convert from ECEF to geodetic using Ferrari's methods
  // https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Ferrari.27s_solution
  double x = e.x;
  double y = e.y;
  double z = e.z;

  double r = sqrt(x * x + y * y);
  double Esq = a * a - b * b;
  double F = 54 * b * b * z * z;
  double G = r * r + (1 - esq) * z * z - esq * Esq;
  double C = (esq * esq * F * r * r) / (pow(G, 3));
  double S = cbrt(1 + C + sqrt(C * C + 2 * C));
  double P = F / (3 * pow((S + 1 / S + 1), 2) * G * G);
  double Q = sqrt(1 + 2 * esq * esq * P);
  double r_0 = -(P * esq * r) / (1 + Q) + sqrt(0.5 * a * a*(1 + 1.0 / Q) - P * (1 - esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r);
  double U = sqrt(pow((r - esq * r_0), 2) + z * z);
  double V = sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z);
  double Z_0 = b * b * z / (a * V);
  double h = U * (1 - b * b / (a * V));

  double lat = atan((z + e1sq * Z_0) / r);
  double lon = atan2(y, x);

  return to_degrees({lat, lon, h});
}

LocalCoord::LocalCoord(Geodetic g, ECEF e){
  init_ecef <<  e.x, e.y, e.z;

  g = to_radians(g);

  ned2ecef_matrix <<
    -sin(g.lat)*cos(g.lon), -sin(g.lon), -cos(g.lat)*cos(g.lon),
    -sin(g.lat)*sin(g.lon), cos(g.lon), -cos(g.lat)*sin(g.lon),
    cos(g.lat), 0, -sin(g.lat);
  ecef2ned_matrix = ned2ecef_matrix.transpose();
}

NED LocalCoord::ecef2ned(ECEF e) {
  Eigen::Vector3d ecef;
  ecef << e.x, e.y, e.z;

  Eigen::Vector3d ned = (ecef2ned_matrix * (ecef - init_ecef));
  return {ned[0], ned[1], ned[2]};
}

ECEF LocalCoord::ned2ecef(NED n) {
  Eigen::Vector3d ned;
  ned << n.n, n.e, n.d;

  Eigen::Vector3d ecef = (ned2ecef_matrix * ned) + init_ecef;
  return {ecef[0], ecef[1], ecef[2]};
}

NED LocalCoord::geodetic2ned(Geodetic g) {
  ECEF e = ::geodetic2ecef(g);
  return ecef2ned(e);
}

Geodetic LocalCoord::ned2geodetic(NED n){
  ECEF e = ned2ecef(n);
  return ::ecef2geodetic(e);
}
