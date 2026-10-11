#include "tools/cabana/lua/python_evaluator.h"

#include <Python.h>
#include <algorithm>

static double interpAt(const std::vector<CerealPoint> &pts, double t) {
  if (pts.empty()) return 0.0;
  auto it = std::lower_bound(pts.begin(), pts.end(), t, [](const CerealPoint &p, double v) { return p.t < v; });
  if (it == pts.end()) return pts.back().v;
  if (it == pts.begin()) return pts.front().v;
  auto prev = std::prev(it);
  double dt = it->t - prev->t;
  if (dt < 1e-9) return prev->v;
  return prev->v + (it->v - prev->v) * (t - prev->t) / dt;
}

static std::string pyErrString() {
  PyObject *ptype, *pvalue, *ptb;
  PyErr_Fetch(&ptype, &pvalue, &ptb);
  PyErr_NormalizeException(&ptype, &pvalue, &ptb);
  std::string msg = "Python error";
  if (pvalue) {
    PyObject *s = PyObject_Str(pvalue);
    if (s) { msg = PyUnicode_AsUTF8(s); Py_DECREF(s); }
  }
  Py_XDECREF(ptype); Py_XDECREF(pvalue); Py_XDECREF(ptb);
  return msg;
}

std::string pythonEvaluate(const PythonSnippet &snippet, const CerealSeriesMap &map, CerealSeriesMap &out) {
  if (!Py_IsInitialized()) Py_InitializeEx(0);

  auto src_it = map.find(snippet.linked_source);
  if (src_it == map.end() || src_it->second.empty())
    return "linked source '" + snippet.linked_source + "' not found in loaded data";

  std::vector<const std::vector<CerealPoint> *> extra;
  for (const auto &path : snippet.additional_sources) {
    auto it = map.find(path);
    extra.push_back(it != map.end() ? &it->second : nullptr);
  }

  // each evaluation gets its own namespace so snippets don't interfere
  PyObject *builtins = PyImport_ImportModule("builtins");
  PyObject *globals = PyDict_New();
  PyDict_SetItemString(globals, "__builtins__", builtins);
  Py_DECREF(builtins);

  PyObject *compiled = PyRun_String(snippet.code.c_str(), Py_file_input, globals, globals);
  if (!compiled) {
    std::string err = pyErrString();
    Py_DECREF(globals);
    return err;
  }
  Py_DECREF(compiled);

  PyObject *func = PyDict_GetItemString(globals, "derive");
  if (!func || !PyCallable_Check(func)) {
    Py_DECREF(globals);
    return "define a function named 'derive(value, v1, v2, v3, v4, time)'";
  }

  auto &out_series = out[snippet.name];
  out_series.reserve(src_it->second.size());

  for (const auto &pt : src_it->second) {
    double ev[4] = {};
    for (int i = 0; i < 4; i++)
      if (i < (int)extra.size() && extra[i]) ev[i] = interpAt(*extra[i], pt.t);

    PyObject *args = PyTuple_Pack(6,
      PyFloat_FromDouble(pt.v),
      PyFloat_FromDouble(ev[0]), PyFloat_FromDouble(ev[1]),
      PyFloat_FromDouble(ev[2]), PyFloat_FromDouble(ev[3]),
      PyFloat_FromDouble(pt.t));

    PyObject *ret = PyObject_CallObject(func, args);
    Py_DECREF(args);
    if (!ret) { PyErr_Clear(); continue; }
    if (PyFloat_Check(ret) || PyLong_Check(ret))
      out_series.push_back({pt.t, PyFloat_AsDouble(ret)});
    Py_DECREF(ret);
  }

  Py_DECREF(globals);
  return {};
}
