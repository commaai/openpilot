# cython: language_level=3
from libcpp cimport bool

cdef extern from "orientation.cc":
  pass

cdef extern from "orientation.hpp":
  cdef cppclass Quaternion "Eigen::Quaterniond":
    Quaternion()
    Quaternion(double, double, double, double)
    double w()
    double x()
    double y()
    double z()

  cdef cppclass Vector3 "Eigen::Vector3d":
    Vector3()
    Vector3(double, double, double)
    double operator()(int)

  cdef cppclass Matrix3 "Eigen::Matrix3d":
    Matrix3()
    Matrix3(double*)

    double operator()(int, int)

  Quaternion euler2quat(Vector3)
  Vector3 quat2euler(Quaternion)
  Matrix3 quat2rot(Quaternion)
  Quaternion rot2quat(Matrix3)
  Vector3 rot2euler(Matrix3)
  Matrix3 euler2rot(Vector3)
  Matrix3 rot_matrix(double, double, double)
  Vector3 ecef_euler_from_ned(ECEF, Vector3)
  Vector3 ned_euler_from_ecef(ECEF, Vector3)


cdef extern from "coordinates.cc":
  cdef struct ECEF:
    double x
    double y
    double z

  cdef struct NED:
    double n
    double e
    double d

  cdef struct Geodetic:
    double lat
    double lon
    double alt
    bool radians

  ECEF geodetic2ecef(Geodetic)
  Geodetic ecef2geodetic(ECEF)

  cdef cppclass LocalCoord_c "LocalCoord":
    Matrix3 ned2ecef_matrix
    Matrix3 ecef2ned_matrix

    LocalCoord_c(Geodetic, ECEF)
    LocalCoord_c(Geodetic)
    LocalCoord_c(ECEF)

    NED ecef2ned(ECEF)
    ECEF ned2ecef(NED)
    NED geodetic2ned(Geodetic)
    Geodetic ned2geodetic(NED)

cdef extern from "coordinates.hpp":
  pass
