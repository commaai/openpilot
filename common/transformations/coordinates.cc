#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>

#define DEG2RAD(x) ((x) * M_PI / 180.0)
#define RAD2DEG(x) ((x) * 180.0 / M_PI)


double a = 6378137;
double b = 6356752.3142;
double esq = 6.69437999014 * 0.001;
double e1sq = 6.73949674228 * 0.001;

struct ECEF {
  double x, y, z;
};

struct NED {
  double n, e, d;
};

struct Geodetic {
  double lat, lon, alt;
  bool radians=false;
};

Geodetic ensure_degrees(Geodetic geodetic){
  if (geodetic.radians) {
    geodetic.lat = RAD2DEG(geodetic.lat);
    geodetic.lon = RAD2DEG(geodetic.lon);
  }
  return geodetic;
}

Geodetic ensure_radians(Geodetic geodetic){
  if (!geodetic.radians) {
    geodetic.lat = DEG2RAD(geodetic.lat);
    geodetic.lon = DEG2RAD(geodetic.lon);
  }
  return geodetic;
}


ECEF geodetic2ecef(Geodetic g){
  g = ensure_radians(g);
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

  return ensure_degrees({lat, lon, h, true});
}


int main(void){
  std::cout << "Hello, Transformations" << std::endl;
  ECEF e = geodetic2ecef({37.7610403, -122.4778699, 115});
  Geodetic g = ecef2geodetic(e);
  std::cout << e.x << "\t" << e.y << "\t" << e.z << std::endl;
  std::cout << g.lat << "\t" << g.lon << "\t" << g.alt << std::endl;

  e = geodetic2ecef({0.65905448, -2.13764209, 115, true});
  std::cout << e.x << "\t" << e.y << "\t" << e.z << std::endl;
}
