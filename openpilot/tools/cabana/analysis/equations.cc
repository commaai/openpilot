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
    auto site = checked(PyUnicode_FromString(CABANA_PYTHON_SITE));
    auto analysis = checked(PyUnicode_FromString(CABANA_ANALYSIS_DIR));
    if (PyList_Insert(path, 0, site.get()) || PyList_Insert(path, 0, analysis.get())) pythonError();
    return checked(PyImport_ImportModule("cabana_equations")).release();
  }();
  return module;
}

thread_local int remaining_steps;
// Lines cover Python loops and C calls cover call-heavy code. A loop that never leaves C, like
// max(iter(int, 1)), cannot be interrupted, so the thread pool detaches stuck workers at exit.
int traceEquation(PyObject *, PyFrameObject *, int event, PyObject *) {
  if ((event == PyTrace_LINE || event == PyTrace_C_CALL) && --remaining_steps <= 0) {
    PyErr_SetString(PyExc_RuntimeError, "Equation exceeded its execution limit");
    return -1;
  }
  return 0;
}
struct ExecutionLimit {
  ExecutionLimit() { reset(); PyEval_SetTrace(traceEquation, nullptr); PyEval_SetProfile(traceEquation, nullptr); }
  ~ExecutionLimit() { PyEval_SetTrace(nullptr, nullptr); PyEval_SetProfile(nullptr, nullptr); }
  void reset() { remaining_steps = 100000; }
};
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
  auto compile = checked(PyObject_GetAttrString(runtimeModule(), "compile_equation"));
  ExecutionLimit limit;
  auto function = checked(PyObject_CallFunction(compile.get(), "ssi", equation.globals.c_str(), equation.function.c_str(), (int)inputs.size()));
  std::vector<Sample> result;
  result.reserve(source->second->size());
  for (const auto &sample : *source->second) {
    limit.reset();
    auto args = checked(PyTuple_New(inputs.size() + 2));
    PyTuple_SET_ITEM(args.get(), 0, checked(PyFloat_FromDouble(sample.x)).release());
    PyTuple_SET_ITEM(args.get(), 1, checked(PyFloat_FromDouble(sample.y)).release());
    for (size_t i = 0; i < inputs.size(); ++i) {
      PyTuple_SET_ITEM(args.get(), i + 2, checked(PyFloat_FromDouble(nearestValue(*inputs[i], sample.x))).release());
    }
    auto output = checked(PyObject_CallObject(function.get(), args.get()));
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
