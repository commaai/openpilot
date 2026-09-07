#include <Python.h>

#include "tools/cabana/analysis/equations.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace cabana {
namespace {
struct PyDeleter { void operator()(PyObject *object) const { Py_XDECREF(object); } };
using PyPtr = std::unique_ptr<PyObject, PyDeleter>;

[[noreturn]] void pythonError() {
  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  PyPtr owned_type(type), owned_value(value), owned_traceback(traceback);
  PyPtr message(value ? PyObject_Str(value) : nullptr);
  const char *text = message ? PyUnicode_AsUTF8(message.get()) : nullptr;
  const std::string error = text ? text : "Python equation failed";
  PyErr_Clear();
  throw std::runtime_error(error);
}

PyPtr checked(PyObject *object) {
  if (!object) pythonError();
  return PyPtr(object);
}

void initializePython() {
  static const bool initialized = []() {
    if (!Py_IsInitialized()) {
      PyConfig config;
      PyConfig_InitIsolatedConfig(&config);
      config.install_signal_handlers = 0;
      config.site_import = 0;  // Do not execute site customizations or .pth files.
      auto status = PyConfig_SetBytesString(&config, &config.home, CABANA_PYTHON_HOME);
      if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
      const std::string error = PyStatus_Exception(status) ? (status.err_msg ? status.err_msg : "Python initialization failed") : "";
      PyConfig_Clear(&config);
      if (!error.empty()) throw std::runtime_error(error);
      PyEval_SaveThread();
    }
    return true;
  }();
  (void)initialized;
}

struct PythonLock {
  PythonLock() { initializePython(); state = PyGILState_Ensure(); }
  ~PythonLock() { PyGILState_Release(state); }
  PyGILState_STATE state;
};

PyObject *runtimeModule() {
  // Kept alive with the interpreter; all access is under the GIL.
  static PyObject *module = []() {
    auto *path = PySys_GetObject("path");
    auto analysis = checked(PyUnicode_FromString(CABANA_ANALYSIS_DIR));
    if (PyList_Insert(path, 0, analysis.get())) pythonError();
    return checked(PyImport_ImportModule("cabana_equations")).release();
  }();
  return module;
}

}  // namespace

double nearestValue(const std::vector<Sample> &samples, double time) {
  if (samples.empty()) return std::numeric_limits<double>::quiet_NaN();
  auto it = std::lower_bound(samples.begin(), samples.end(), time, [](const auto &p, double x) { return p.x < x; });
  if (it == samples.end()) return samples.back().y;
  if (it != samples.begin() && time - (it - 1)->x < it->x - time) --it;
  return it->y;
}

std::vector<Sample> evaluateEquation(const Equation &equation, const FieldsSnapshot &data) {
  auto source = data.find(equation.source);
  if (source == data.end() || source->second->empty()) throw std::runtime_error("Waiting for " + equation.source);
  std::vector<const std::vector<Sample> *> inputs;
  for (const auto &path : equation.additional) {
    auto it = data.find(path);
    if (it == data.end() || it->second->empty()) throw std::runtime_error("Waiting for " + path);
    inputs.push_back(it->second.get());
  }
  PythonLock lock;
  auto compile = checked(PyObject_GetAttrString(runtimeModule(), "compile_numeric_equation"));
  auto function = checked(PyObject_CallFunction(compile.get(), "ssi", equation.globals.c_str(), equation.function.c_str(), (int)inputs.size()));
  std::vector<Sample> result;
  result.reserve(source->second->size());
  // The AST compiler still validates every equation. Native inputs are already floats;
  // call the compiled function directly without a Python conversion wrapper or argument tuple.
  std::vector<PyPtr> args(inputs.size() + 2);
  std::vector<PyObject *> argv(args.size());
  auto setArg = [&](size_t index, double value) {
    args[index] = checked(PyFloat_FromDouble(value));
    argv[index] = args[index].get();
  };
  for (const auto &sample : *source->second) {
    setArg(0, sample.x);
    setArg(1, sample.y);
    for (size_t i = 0; i < inputs.size(); ++i) {
      setArg(i + 2, nearestValue(*inputs[i], sample.x));
    }
    auto output = checked(PyObject_Vectorcall(function.get(), argv.data(), argv.size(), nullptr));
    double time = sample.x;
    PyObject *value = output.get();
    if (PyTuple_Check(value)) {
      if (PyTuple_Size(value) != 2) throw std::runtime_error("Equation must return value or (time, value)");
      time = PyFloat_AsDouble(PyTuple_GetItem(value, 0));
      value = PyTuple_GetItem(value, 1);
    }
    double y = PyFloat_AsDouble(value);
    if (PyErr_Occurred()) pythonError();
    if (std::isfinite(time) && std::isfinite(y)) result.emplace_back(time, y);
  }
  std::stable_sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.x < b.x; });
  return result;
}
}  // namespace cabana
