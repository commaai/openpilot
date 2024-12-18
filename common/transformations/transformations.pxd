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

  Quaternion euler2quat(const Vector3 &)
  Vector3 quat2euler(const Quaternion &)
  Matrix3 quat2rot(const Quaternion &)
  Quaternion rot2quat(const Matrix3 &)
  Vector3 rot2euler(const Matrix3 &)
  Matrix3 euler2rot(const Vector3 &)
  Matrix3 rot_matrix(double, double, double)
  Vector3 ecef_euler_from_ned(const ECEF &, const Vector3 &)
  Vector3 ned_euler_from_ecef(const ECEF &, const Vector3 &)


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

  ECEF geodetic2ecef(const Geodetic &)
  Geodetic ecef2geodetic(const ECEF &)

  cdef cppclass LocalCoord_c "LocalCoord":
    Matrix3 ned2ecef_matrix
    Matrix3 ecef2ned_matrix

    LocalCoord_c(const Geodetic &, const ECEF &)
    LocalCoord_c(const Geodetic &)
    LocalCoord_c(const ECEF &)

    NED ecef2ned(const ECEF &)
    ECEF ned2ecef(const NED &)
    NED geodetic2ned(const Geodetic &)
    Geodetic ned2geodetic(const NED &)

cdef extern from "coordinates.hpp":
  pass
