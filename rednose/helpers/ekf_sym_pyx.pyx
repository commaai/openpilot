# cython: language_level=3
# cython: profile=True
# distutils: language = c++

cimport cython

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
cimport numpy as np

import numpy as np


cdef extern from "<optional>" namespace "std" nogil:
  cdef cppclass optional[T]:
    ctypedef T value_type
    bool has_value()
    T& value()

cdef extern from "rednose/helpers/ekf_load.h":
  cdef void ekf_load_and_register(string directory, string name)

cdef extern from "rednose/helpers/ekf_sym.h" namespace "EKFS":
  cdef cppclass MapVectorXd "Eigen::Map<Eigen::VectorXd>":
    MapVectorXd(double*, int)

  cdef cppclass MapMatrixXdr "Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >":
    MapMatrixXdr(double*, int, int)

  cdef cppclass VectorXd "Eigen::VectorXd":
    VectorXd()
    double* data()
    int rows()

  cdef cppclass MatrixXdr "Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>":
    MatrixXdr()
    double* data()
    int rows()
    int cols()

  ctypedef struct Estimate:
    VectorXd xk1
    VectorXd xk
    MatrixXdr Pk1
    MatrixXdr Pk
    double t
    int kind
    vector[VectorXd] y
    vector[VectorXd] z
    vector[vector[double]] extra_args

  cdef cppclass EKFSym:
    EKFSym(string name, MapMatrixXdr Q, MapVectorXd x_initial, MapMatrixXdr P_initial, int dim_main,
        int dim_main_err, int N, int dim_augment, int dim_augment_err, vector[int] maha_test_kinds,
        vector[int] quaternion_idxs, vector[string] global_vars, double max_rewind_age)
    void init_state(MapVectorXd state, MapMatrixXdr covs, double filter_time)

    VectorXd state()
    MatrixXdr covs()
    void set_filter_time(double t)
    double get_filter_time()
    void set_global(string name, double val)
    void reset_rewind()

    void predict(double t)
    optional[Estimate] predict_and_update_batch(double t, int kind, vector[MapVectorXd] z, vector[MapMatrixXdr] z,
        vector[vector[double]] extra_args, bool augment)

# Functions like `numpy_to_matrix` are not possible, cython requires default
# constructor for return variable types which aren't available with Eigen::Map

@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray[np.float64_t, ndim=2, mode="c"] matrix_to_numpy(MatrixXdr arr):
  cdef double[:,:] mem_view = <double[:arr.rows(),:arr.cols()]>arr.data()
  return np.copy(np.asarray(mem_view, dtype=np.double, order="C"))

@cython.wraparound(False)
@cython.boundscheck(False)
cdef np.ndarray[np.float64_t, ndim=1, mode="c"] vector_to_numpy(VectorXd arr):
  cdef double[:] mem_view = <double[:arr.rows()]>arr.data()
  return np.copy(np.asarray(mem_view, dtype=np.double, order="C"))

cdef class EKF_sym_pyx:
  cdef EKFSym* ekf
  def __cinit__(self, str gen_dir, str name, np.ndarray[np.float64_t, ndim=2] Q,
      np.ndarray[np.float64_t, ndim=1] x_initial, np.ndarray[np.float64_t, ndim=2] P_initial, int dim_main,
      int dim_main_err, int N=0, int dim_augment=0, int dim_augment_err=0, list maha_test_kinds=[],
      list quaternion_idxs=[], list global_vars=[], double max_rewind_age=1.0, logger=None):
    # TODO logger
    ekf_load_and_register(gen_dir.encode('utf8'), name.encode('utf8'))

    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Q_b = np.ascontiguousarray(Q, dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] x_initial_b = np.ascontiguousarray(x_initial, dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] P_initial_b = np.ascontiguousarray(P_initial, dtype=np.double)
    self.ekf = new EKFSym(
      name.encode('utf8'),
      MapMatrixXdr(<double*> Q_b.data, Q.shape[0], Q.shape[1]),
      MapVectorXd(<double*> x_initial_b.data, x_initial.shape[0]),
      MapMatrixXdr(<double*> P_initial_b.data, P_initial.shape[0], P_initial.shape[1]),
      dim_main,
      dim_main_err,
      N,
      dim_augment,
      dim_augment_err,
      maha_test_kinds,
      quaternion_idxs,
      [x.encode('utf8') for x in global_vars],
      max_rewind_age
    )

  def init_state(self, np.ndarray[np.float64_t, ndim=1] state, np.ndarray[np.float64_t, ndim=2] covs, filter_time):
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] state_b = np.ascontiguousarray(state, dtype=np.double)
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] covs_b = np.ascontiguousarray(covs, dtype=np.double)
    self.ekf.init_state(
      MapVectorXd(<double*> state_b.data, state.shape[0]),
      MapMatrixXdr(<double*> covs_b.data, covs.shape[0], covs.shape[1]),
      np.nan if filter_time is None else filter_time
    )

  def state(self):
    cdef np.ndarray res = vector_to_numpy(self.ekf.state())
    return res

  def covs(self):
    return matrix_to_numpy(self.ekf.covs())

  def set_filter_time(self, double t):
    self.ekf.set_filter_time(t)

  def get_filter_time(self):
    return self.ekf.get_filter_time()

  def set_global(self, str global_var, double val):
    self.ekf.set_global(global_var.encode('utf8'), val)

  def reset_rewind(self):
    self.ekf.reset_rewind()

  def predict(self, double t):
    self.ekf.predict(t)

  def predict_and_update_batch(self, double t, int kind, z, R, extra_args=[[]], bool augment=False):
    cdef vector[MapVectorXd] z_map
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] zi_b
    for zi in z:
      zi_b = np.ascontiguousarray(zi, dtype=np.double)
      z_map.push_back(MapVectorXd(<double*> zi_b.data, zi.shape[0]))

    cdef vector[MapMatrixXdr] R_map
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Ri_b
    for Ri in R:
      Ri_b = np.ascontiguousarray(Ri, dtype=np.double)
      R_map.push_back(MapMatrixXdr(<double*> Ri_b.data, Ri.shape[0], Ri.shape[1]))

    cdef vector[vector[double]] extra_args_map
    cdef vector[double] args_map
    for args in extra_args:
      args_map.clear()
      for a in args:
        args_map.push_back(a)
      extra_args_map.push_back(args_map)

    cdef optional[Estimate] res = self.ekf.predict_and_update_batch(t, kind, z_map, R_map, extra_args_map, augment)
    if not res.has_value():
      return None

    cdef VectorXd tmpvec
    return (
      vector_to_numpy(res.value().xk1),
      vector_to_numpy(res.value().xk),
      matrix_to_numpy(res.value().Pk1),
      matrix_to_numpy(res.value().Pk),
      res.value().t,
      res.value().kind,
      [vector_to_numpy(tmpvec) for tmpvec in res.value().y],
      z,  # TODO: take return values?
      extra_args,
    )

  def augment(self):
    raise NotImplementedError()  # TODO

  def get_augment_times(self):
    raise NotImplementedError()  # TODO

  def rts_smooth(self, estimates, norm_quats=False):
    raise NotImplementedError()  # TODO

  def maha_test(self, x, P, kind, z, R, extra_args=[], maha_thresh=0.95):
    raise NotImplementedError()  # TODO

  def __dealloc__(self):
    del self.ekf
