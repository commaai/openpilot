#include <cstdlib>
#include <cstring>
#include "common/transformations/coordinates.hpp"
#include "common/transformations/orientation.hpp"

extern "C" {

typedef struct {
  double x, y, z;
} ECEF_C;

typedef struct {
  double n, e, d;
} NED_C;

typedef struct {
  double lat, lon, alt;
  bool radians;
} Geodetic_C;

typedef struct {
  double w, x, y, z;
} Quaternion_C;

// Helper conversions
static ECEF to_eigen(const ECEF_C& c) { return {c.x, c.y, c.z}; }
static ECEF_C from_eigen(const ECEF& c) { return {c.x, c.y, c.z}; }
static NED to_eigen(const NED_C& c) { return {c.n, c.e, c.d}; }
static NED_C from_eigen(const NED& c) { return {c.n, c.e, c.d}; }
static Geodetic to_geod(const Geodetic_C& c) { return {c.lat, c.lon, c.alt, c.radians}; }
static Geodetic_C from_geod(const Geodetic& c) { return {c.lat, c.lon, c.alt, c.radians}; }

static Eigen::Vector3d vec3_from_arr(double* arr) {
  return Eigen::Vector3d(arr[0], arr[1], arr[2]);
}

static void vec3_to_arr(const Eigen::Vector3d& v, double* arr) {
  arr[0] = v.x(); arr[1] = v.y(); arr[2] = v.z();
}

static Eigen::Quaterniond quat_from_c(const Quaternion_C& q) {
  return Eigen::Quaterniond(q.w, q.x, q.y, q.z);
}

static Quaternion_C quat_to_c(const Eigen::Quaterniond& q) {
  return {q.w(), q.x(), q.y(), q.z()};
}

static void mat3_to_arr(const Eigen::Matrix3d& m, double* arr) {
  // Row-major for numpy compatibility/readability? Or Col-major?
  // Eigen defaults to Col-major. Numpy defaults to Row-major.
  // The old module implementation iterated: m(i, j) => data[i*3 + j] -> Row-major.
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      arr[i * 3 + j] = m(i, j);
    }
  }
}

static Eigen::Matrix3d mat3_from_arr(double* arr) {
  Eigen::Matrix3d m;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m(i, j) = arr[i * 3 + j];
    }
  }
  return m;
}

// Coordinates
void geodetic2ecef(Geodetic_C g, ECEF_C* out) {
  *out = from_eigen(geodetic2ecef(to_geod(g)));
}

void ecef2geodetic(ECEF_C e, Geodetic_C* out) {
  *out = from_geod(ecef2geodetic(to_eigen(e)));
}

// Orientation
void euler2quat(double* euler, Quaternion_C* out) {
  *out = quat_to_c(euler2quat(vec3_from_arr(euler)));
}

void quat2euler(Quaternion_C q, double* out) {
  vec3_to_arr(quat2euler(quat_from_c(q)), out);
}

void quat2rot(Quaternion_C q, double* out) {
  mat3_to_arr(quat2rot(quat_from_c(q)), out);
}

void rot2quat(double* rot, Quaternion_C* out) {
  *out = quat_to_c(rot2quat(mat3_from_arr(rot)));
}

void euler2rot(double* euler, double* out) {
  mat3_to_arr(euler2rot(vec3_from_arr(euler)), out);
}

void rot2euler(double* rot, double* out) {
  vec3_to_arr(rot2euler(mat3_from_arr(rot)), out);
}

void rot_matrix(double roll, double pitch, double yaw, double* out) {
  mat3_to_arr(rot_matrix(roll, pitch, yaw), out);
}

void ecef_euler_from_ned(ECEF_C ecef_init, double* ned_pose, double* out) {
  vec3_to_arr(ecef_euler_from_ned(to_eigen(ecef_init), vec3_from_arr(ned_pose)), out);
}

void ned_euler_from_ecef(ECEF_C ecef_init, double* ecef_pose, double* out) {
  vec3_to_arr(ned_euler_from_ecef(to_eigen(ecef_init), vec3_from_arr(ecef_pose)), out);
}

// LocalCoord
void* localcoord_create(Geodetic_C g) {
  return new LocalCoord(to_geod(g));
}

void* localcoord_create_from_ecef(ECEF_C e) {
  return new LocalCoord(to_eigen(e));
}

void localcoord_destroy(void* lc) {
  delete (LocalCoord*)lc;
}

void localcoord_ecef2ned(void* lc, ECEF_C e, NED_C* out) {
  *out = from_eigen(((LocalCoord*)lc)->ecef2ned(to_eigen(e)));
}

void localcoord_ned2ecef(void* lc, NED_C n, ECEF_C* out) {
  *out = from_eigen(((LocalCoord*)lc)->ned2ecef(to_eigen(n)));
}

void localcoord_geodetic2ned(void* lc, Geodetic_C g, NED_C* out) {
  *out = from_eigen(((LocalCoord*)lc)->geodetic2ned(to_geod(g)));
}

void localcoord_ned2geodetic(void* lc, NED_C n, Geodetic_C* out) {
  *out = from_geod(((LocalCoord*)lc)->ned2geodetic(to_eigen(n)));
}

void localcoord_get_ned2ecef_matrix(void* lc, double* out) {
  mat3_to_arr(((LocalCoord*)lc)->ned2ecef_matrix, out);
}

void localcoord_get_ecef2ned_matrix(void* lc, double* out) {
  mat3_to_arr(((LocalCoord*)lc)->ecef2ned_matrix, out);
}

}
