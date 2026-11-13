#include <limits>
#include <new>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "rednose/helpers/ekf_sym.h"

namespace nb = nanobind;
using namespace nb::literals;
using EKFS::EKFSym;
using EKFS::MatrixXdr;
using Eigen::VectorXd;

NB_MODULE(ekf_sym_pyx, m) {
  // Keep the historical name for compatibility with existing filter clients.
  nb::class_<EKFSym>(m, "EKF_sym_pyx")
    .def("__init__", [](
      EKFSym *self,
      const std::string &gen_dir,
      const std::string &name,
      MatrixXdr Q,
      VectorXd x_initial,
      MatrixXdr P_initial,
      int dim_main,
      int dim_main_err,
      int N,
      int dim_augment,
      int dim_augment_err,
      std::vector<int> maha_test_kinds,
      std::vector<int> quaternion_idxs,
      std::vector<std::string> global_vars,
      double max_rewind_age,
      nb::object logger
    ) {
      ekf_load_and_register(gen_dir, name);
      new (self) EKFSym(
        name,
        Eigen::Map<MatrixXdr>(Q.data(), Q.rows(), Q.cols()),
        Eigen::Map<VectorXd>(x_initial.data(), x_initial.size()),
        Eigen::Map<MatrixXdr>(P_initial.data(), P_initial.rows(), P_initial.cols()),
        dim_main,
        dim_main_err,
        N,
        dim_augment,
        dim_augment_err,
        maha_test_kinds,
        quaternion_idxs,
        global_vars,
        max_rewind_age
      );
    },
    "gen_dir"_a,
    "name"_a,
    "Q"_a,
    "x_initial"_a,
    "P_initial"_a,
    "dim_main"_a,
    "dim_main_err"_a,
    "N"_a = 0,
    "dim_augment"_a = 0,
    "dim_augment_err"_a = 0,
    "maha_test_kinds"_a = std::vector<int>{},
    "quaternion_idxs"_a = std::vector<int>{},
    "global_vars"_a = std::vector<std::string>{},
    "max_rewind_age"_a = 1.0,
    "logger"_a.none() = nb::none())
    .def("init_state", [](EKFSym &self, VectorXd state, MatrixXdr covs, nb::object filter_time) {
      self.init_state(
        Eigen::Map<VectorXd>(state.data(), state.size()),
        Eigen::Map<MatrixXdr>(covs.data(), covs.rows(), covs.cols()),
        filter_time.is_none() ? std::numeric_limits<double>::quiet_NaN() : nb::cast<double>(filter_time)
      );
    }, "state"_a, "covs"_a, "filter_time"_a.none())
    .def("state", &EKFSym::state)
    .def("covs", &EKFSym::covs)
    .def("set_filter_time", &EKFSym::set_filter_time, "t"_a)
    .def("get_filter_time", &EKFSym::get_filter_time)
    .def("set_global", &EKFSym::set_global, "global_var"_a, "val"_a)
    .def("normalize_slice", &EKFSym::normalize_slice, "slice_start"_a, "slice_end_ex"_a)
    .def("reset_rewind", &EKFSym::reset_rewind)
    .def("predict", &EKFSym::predict, "t"_a)
    .def("predict_and_update_batch", [](
      EKFSym &self,
      double t,
      int kind,
      nb::object z,
      nb::object R,
      nb::object extra_args,
      bool augment
    ) -> std::optional<nb::tuple> {
      // Own all converted buffers until the filter has finished using the maps.
      auto zs = nb::cast<std::vector<VectorXd>>(z);
      auto Rs = nb::cast<std::vector<MatrixXdr>>(R);
      std::vector<Eigen::Map<VectorXd>> z_maps;
      std::vector<Eigen::Map<MatrixXdr>> R_maps;
      for (auto &zi : zs) {
        z_maps.emplace_back(zi.data(), zi.size());
      }
      for (auto &Ri : Rs) {
        R_maps.emplace_back(Ri.data(), Ri.rows(), Ri.cols());
      }
      auto result = self.predict_and_update_batch(
        t, kind, z_maps, R_maps,
        nb::cast<std::vector<std::vector<double>>>(extra_args), augment
      );
      if (!result) {
        return std::nullopt;
      }
      return nb::make_tuple(
        result->xk1, result->xk, result->Pk1, result->Pk, result->t,
        result->kind, result->y, z, extra_args
      );
    }, "t"_a, "kind"_a, "z"_a, "R"_a,
       "extra_args"_a = std::vector<std::vector<double>>{{}}, "augment"_a = false)
    .def("augment", [](EKFSym &) {
      PyErr_SetNone(PyExc_NotImplementedError);
      throw nb::python_error();
    })
    .def("get_augment_times", [](EKFSym &) {
      PyErr_SetNone(PyExc_NotImplementedError);
      throw nb::python_error();
    })
    .def("rts_smooth", [](EKFSym &, nb::object, bool) {
      PyErr_SetNone(PyExc_NotImplementedError);
      throw nb::python_error();
    }, "estimates"_a, "norm_quats"_a = false)
    .def("maha_test", [](EKFSym &, nb::object, nb::object, int, nb::object, nb::object, nb::object, double) {
      PyErr_SetNone(PyExc_NotImplementedError);
      throw nb::python_error();
    }, "x"_a, "P"_a, "kind"_a, "z"_a, "R"_a, "extra_args"_a = nb::list(), "maha_thresh"_a = 0.95);
}
