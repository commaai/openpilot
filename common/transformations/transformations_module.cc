#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <iostream>
#include <vector>

#include "common/transformations/coordinates.hpp"
#include "common/transformations/orientation.hpp"


static bool list2ecef(PyObject *obj, ECEF &out) {
  if (!PySequence_Check(obj) || PySequence_Size(obj) != 3) {
    PyErr_SetString(PyExc_ValueError, "ECEF must be a sequence of length 3");
    return false;
  }
  PyObject *item;
  item = PySequence_GetItem(obj, 0); out.x = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(obj, 1); out.y = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(obj, 2); out.z = PyFloat_AsDouble(item); Py_DECREF(item);
  if (PyErr_Occurred()) return false;
  return true;
}

static bool list2ned(PyObject *obj, NED &out) {
  if (!PySequence_Check(obj) || PySequence_Size(obj) != 3) {
    PyErr_SetString(PyExc_ValueError, "NED must be a sequence of length 3");
    return false;
  }
  PyObject *item;
  item = PySequence_GetItem(obj, 0); out.n = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(obj, 1); out.e = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(obj, 2); out.d = PyFloat_AsDouble(item); Py_DECREF(item);
  if (PyErr_Occurred()) return false;
  return true;
}

static bool list2geodetic(PyObject *obj, Geodetic &out) {
  if (!PySequence_Check(obj) || PySequence_Size(obj) != 3) {
    PyErr_SetString(PyExc_ValueError, "Geodetic must be a sequence of length 3");
    return false;
  }
  PyObject *item;
  item = PySequence_GetItem(obj, 0); out.lat = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(obj, 1); out.lon = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(obj, 2); out.alt = PyFloat_AsDouble(item); Py_DECREF(item);
  out.radians = false;
  if (PyErr_Occurred()) return false;
  return true;
}

static PyObject* ecef2list(const ECEF &e) {
  PyObject *lst = PyList_New(3);
  PyList_SetItem(lst, 0, PyFloat_FromDouble(e.x));
  PyList_SetItem(lst, 1, PyFloat_FromDouble(e.y));
  PyList_SetItem(lst, 2, PyFloat_FromDouble(e.z));
  return lst;
}

static PyObject* ned2list(const NED &n) {
  PyObject *lst = PyList_New(3);
  PyList_SetItem(lst, 0, PyFloat_FromDouble(n.n));
  PyList_SetItem(lst, 1, PyFloat_FromDouble(n.e));
  PyList_SetItem(lst, 2, PyFloat_FromDouble(n.d));
  return lst;
}

static PyObject* geodetic2list(const Geodetic &g) {
  PyObject *lst = PyList_New(3);
  PyList_SetItem(lst, 0, PyFloat_FromDouble(g.lat));
  PyList_SetItem(lst, 1, PyFloat_FromDouble(g.lon));
  PyList_SetItem(lst, 2, PyFloat_FromDouble(g.alt));
  return lst;
}

static PyObject* vector3_to_list(const Eigen::Vector3d &v) {
  PyObject *lst = PyList_New(3);
  PyList_SetItem(lst, 0, PyFloat_FromDouble(v(0)));
  PyList_SetItem(lst, 1, PyFloat_FromDouble(v(1)));
  PyList_SetItem(lst, 2, PyFloat_FromDouble(v(2)));
  return lst;
}

static PyObject* quat_to_list(const Eigen::Quaterniond &q) {
  PyObject *lst = PyList_New(4);
  PyList_SetItem(lst, 0, PyFloat_FromDouble(q.w()));
  PyList_SetItem(lst, 1, PyFloat_FromDouble(q.x()));
  PyList_SetItem(lst, 2, PyFloat_FromDouble(q.y()));
  PyList_SetItem(lst, 3, PyFloat_FromDouble(q.z()));
  return lst;
}

static PyObject* matrix3_to_numpy(const Eigen::Matrix3d &m) {
  npy_intp dims[2] = {3, 3};
  PyObject *arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (!arr) return NULL;


  double *data = (double*)PyArray_DATA((PyArrayObject*)arr);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      data[i*3 + j] = m(i, j);
    }
  }
  return arr;
}

static bool numpy_to_matrix3(PyObject *arr, Eigen::Matrix3d &out) {
  PyArrayObject *arr_np = (PyArrayObject*)PyArray_ContiguousFromAny(arr, NPY_DOUBLE, 2, 2);
  if (!arr_np) return false;

  if (PyArray_DIM(arr_np, 0) != 3 || PyArray_DIM(arr_np, 1) != 3) {
    PyErr_SetString(PyExc_ValueError, "Matrix must be 3x3");
    Py_DECREF(arr_np);
    return false;
  }

  double *data = (double*)PyArray_DATA(arr_np);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = data[i*3 + j];
    }
  }
  Py_DECREF(arr_np);
  return true;
}

// --- Module Functions ---

static PyObject* meth_euler2quat_single(PyObject *self, PyObject *args) {
  PyObject *euler_obj;
  if (!PyArg_ParseTuple(args, "O", &euler_obj)) return NULL;

  if (!PySequence_Check(euler_obj) || PySequence_Size(euler_obj) != 3) {
     PyErr_SetString(PyExc_ValueError, "Euler sequence must be size 3");
     return NULL;
  }

  Eigen::Vector3d e;
  PyObject *item;
  item = PySequence_GetItem(euler_obj, 0); e(0) = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(euler_obj, 1); e(1) = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(euler_obj, 2); e(2) = PyFloat_AsDouble(item); Py_DECREF(item);

  if (PyErr_Occurred()) return NULL;

  return quat_to_list(euler2quat(e));
}

static PyObject* meth_quat2euler_single(PyObject *self, PyObject *args) {
  PyObject *quat_obj;
  if (!PyArg_ParseTuple(args, "O", &quat_obj)) return NULL;

  if (!PySequence_Check(quat_obj) || PySequence_Size(quat_obj) != 4) {
    PyErr_SetString(PyExc_ValueError, "Quaternion sequence must be size 4");
    return NULL;
  }

  double q_vals[4];
  for(int i=0; i<4; ++i) {
     PyObject *item = PySequence_GetItem(quat_obj, i);
     q_vals[i] = PyFloat_AsDouble(item);
     Py_DECREF(item);
  }
  if (PyErr_Occurred()) return NULL;

  Eigen::Quaterniond q(q_vals[0], q_vals[1], q_vals[2], q_vals[3]);
  return vector3_to_list(quat2euler(q));
}

static PyObject* meth_quat2rot_single(PyObject *self, PyObject *args) {
  PyObject *quat_obj;
  if (!PyArg_ParseTuple(args, "O", &quat_obj)) return NULL;

  if (!PySequence_Check(quat_obj) || PySequence_Size(quat_obj) != 4) {
    PyErr_SetString(PyExc_ValueError, "Quaternion sequence must be size 4");
    return NULL;
  }
  double q_vals[4];
  for(int i=0; i<4; ++i) {
     PyObject *item = PySequence_GetItem(quat_obj, i);
     q_vals[i] = PyFloat_AsDouble(item);
     Py_DECREF(item);
  }
  if (PyErr_Occurred()) return NULL;

  Eigen::Quaterniond q(q_vals[0], q_vals[1], q_vals[2], q_vals[3]);
  return matrix3_to_numpy(quat2rot(q));
}

static PyObject* meth_rot2quat_single(PyObject *self, PyObject *args) {
  PyObject *rot_obj;
  if (!PyArg_ParseTuple(args, "O", &rot_obj)) return NULL;

  Eigen::Matrix3d m;
  if (!numpy_to_matrix3(rot_obj, m)) return NULL;

  return quat_to_list(rot2quat(m));
}

static PyObject* meth_euler2rot_single(PyObject *self, PyObject *args) {
  PyObject *euler_obj;
  if (!PyArg_ParseTuple(args, "O", &euler_obj)) return NULL;

  if (!PySequence_Check(euler_obj) || PySequence_Size(euler_obj) != 3) {
     PyErr_SetString(PyExc_ValueError, "Euler sequence must be size 3");
     return NULL;
  }

  Eigen::Vector3d e;
  PyObject *item;
  item = PySequence_GetItem(euler_obj, 0); e(0) = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(euler_obj, 1); e(1) = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(euler_obj, 2); e(2) = PyFloat_AsDouble(item); Py_DECREF(item);
  if (PyErr_Occurred()) return NULL;

  return matrix3_to_numpy(euler2rot(e));
}

static PyObject* meth_rot2euler_single(PyObject *self, PyObject *args) {
  PyObject *rot_obj;
  if (!PyArg_ParseTuple(args, "O", &rot_obj)) return NULL;

  Eigen::Matrix3d m;
  if (!numpy_to_matrix3(rot_obj, m)) return NULL;

  return vector3_to_list(rot2euler(m));
}

static PyObject* meth_rot_matrix(PyObject *self, PyObject *args) {
  double r, p, y;
  if (!PyArg_ParseTuple(args, "ddd", &r, &p, &y)) return NULL;
  return matrix3_to_numpy(rot_matrix(r, p, y));
}

static PyObject* meth_ecef_euler_from_ned_single(PyObject *self, PyObject *args) {
  PyObject *ecef_init_obj, *ned_pose_obj;
  if (!PyArg_ParseTuple(args, "OO", &ecef_init_obj, &ned_pose_obj)) return NULL;

  ECEF init;
  if (!list2ecef(ecef_init_obj, init)) return NULL;

  if (!PySequence_Check(ned_pose_obj) || PySequence_Size(ned_pose_obj) != 3) {
    PyErr_SetString(PyExc_ValueError, "NED pose must be size 3");
    return NULL;
  }
  Eigen::Vector3d pose;
  PyObject *item;
  item = PySequence_GetItem(ned_pose_obj, 0); pose(0) = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(ned_pose_obj, 1); pose(1) = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(ned_pose_obj, 2); pose(2) = PyFloat_AsDouble(item); Py_DECREF(item);
  if (PyErr_Occurred()) return NULL;

  return vector3_to_list(ecef_euler_from_ned(init, pose));
}

static PyObject* meth_ned_euler_from_ecef_single(PyObject *self, PyObject *args) {
  PyObject *ecef_init_obj, *ecef_pose_obj;
  if (!PyArg_ParseTuple(args, "OO", &ecef_init_obj, &ecef_pose_obj)) return NULL;

  ECEF init;
  if (!list2ecef(ecef_init_obj, init)) return NULL;

  if (!PySequence_Check(ecef_pose_obj) || PySequence_Size(ecef_pose_obj) != 3) {
    PyErr_SetString(PyExc_ValueError, "ECEF pose must be size 3");
    return NULL;
  }
  Eigen::Vector3d pose;
  PyObject *item;
  item = PySequence_GetItem(ecef_pose_obj, 0); pose(0) = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(ecef_pose_obj, 1); pose(1) = PyFloat_AsDouble(item); Py_DECREF(item);
  item = PySequence_GetItem(ecef_pose_obj, 2); pose(2) = PyFloat_AsDouble(item); Py_DECREF(item);
  if (PyErr_Occurred()) return NULL;

  return vector3_to_list(ned_euler_from_ecef(init, pose));
}

static PyObject* meth_geodetic2ecef_single(PyObject *self, PyObject *args) {
  PyObject *g_obj;
  if (!PyArg_ParseTuple(args, "O", &g_obj)) return NULL;
  Geodetic g;
  if (!list2geodetic(g_obj, g)) return NULL;
  return ecef2list(geodetic2ecef(g));
}

static PyObject* meth_ecef2geodetic_single(PyObject *self, PyObject *args) {
  PyObject *e_obj;
  if (!PyArg_ParseTuple(args, "O", &e_obj)) return NULL;
  ECEF e;
  if (!list2ecef(e_obj, e)) return NULL;
  return geodetic2list(ecef2geodetic(e));
}

// --- LocalCoord Class ---

typedef struct {
  PyObject_HEAD
  LocalCoord *lc;
} LocalCoordObject;

static void LocalCoord_dealloc(LocalCoordObject *self) {
  if (self->lc) delete self->lc;
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static int LocalCoord_init(LocalCoordObject *self, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = {(char *)"geodetic", (char *)"ecef", NULL};
  PyObject *geodetic_obj = NULL;
  PyObject *ecef_obj = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &geodetic_obj, &ecef_obj)) {
    return -1;
  }

  if (geodetic_obj && geodetic_obj != Py_None) {
    Geodetic g;
    if (!list2geodetic(geodetic_obj, g)) return -1;
    self->lc = new LocalCoord(g);
  } else if (ecef_obj && ecef_obj != Py_None) {
    ECEF e;
    if (!list2ecef(ecef_obj, e)) return -1;
    self->lc = new LocalCoord(e);
  } else {
    PyErr_SetString(PyExc_ValueError, "Must provide geodetic or ecef");
    return -1;
  }
  return 0;
}

static PyObject* LocalCoord_get_ned2ecef_matrix(LocalCoordObject *self, void *closure) {
  if (!self->lc) {
     PyErr_SetString(PyExc_RuntimeError, "LocalCoord not initialized");
     return NULL;
  }
  return matrix3_to_numpy(self->lc->ned2ecef_matrix);
}

static PyObject* LocalCoord_get_ecef2ned_matrix(LocalCoordObject *self, void *closure) {
  if (!self->lc) {
     PyErr_SetString(PyExc_RuntimeError, "LocalCoord not initialized");
     return NULL;
  }
  return matrix3_to_numpy(self->lc->ecef2ned_matrix);
}

static PyObject* LocalCoord_from_geodetic(PyObject *cls, PyObject *g_obj) {
  PyObject *kwds = PyDict_New();
  PyDict_SetItemString(kwds, "geodetic", g_obj);

  PyObject *empty_args = PyTuple_New(0);
  PyObject *result = PyObject_Call(cls, empty_args, kwds);
  Py_DECREF(empty_args);
  Py_DECREF(kwds);
  return result;
}

static PyObject* LocalCoord_from_ecef(PyObject *cls, PyObject *e_obj) {
  PyObject *kwds = PyDict_New();
  PyDict_SetItemString(kwds, "ecef", e_obj);

  PyObject *empty_args = PyTuple_New(0);
  PyObject *result = PyObject_Call(cls, empty_args, kwds);
  Py_DECREF(empty_args);
  Py_DECREF(kwds);
  return result;
}

static PyObject* LocalCoord_ecef2ned_single(LocalCoordObject *self, PyObject *e_obj) {
  if (!self->lc) { PyErr_SetString(PyExc_RuntimeError, "Uninitialized"); return NULL; }
  ECEF e;
  if (!list2ecef(e_obj, e)) return NULL;
  return ned2list(self->lc->ecef2ned(e));
}

static PyObject* LocalCoord_ned2ecef_single(LocalCoordObject *self, PyObject *n_obj) {
  if (!self->lc) { PyErr_SetString(PyExc_RuntimeError, "Uninitialized"); return NULL; }
  NED n;
  if (!list2ned(n_obj, n)) return NULL;
  return ecef2list(self->lc->ned2ecef(n));
}

static PyObject* LocalCoord_geodetic2ned_single(LocalCoordObject *self, PyObject *g_obj) {
  if (!self->lc) { PyErr_SetString(PyExc_RuntimeError, "Uninitialized"); return NULL; }
  Geodetic g;
  if (!list2geodetic(g_obj, g)) return NULL;
  return ned2list(self->lc->geodetic2ned(g));
}

static PyObject* LocalCoord_ned2geodetic_single(LocalCoordObject *self, PyObject *n_obj) {
  if (!self->lc) { PyErr_SetString(PyExc_RuntimeError, "Uninitialized"); return NULL; }
  NED n;
  if (!list2ned(n_obj, n)) return NULL;
  return geodetic2list(self->lc->ned2geodetic(n));
}

static PyMethodDef LocalCoord_methods[] = {
  {"from_geodetic", (PyCFunction)LocalCoord_from_geodetic, METH_O | METH_CLASS, "Create from geodetic"},
  {"from_ecef", (PyCFunction)LocalCoord_from_ecef, METH_O | METH_CLASS, "Create from ecef"},
  {"ecef2ned_single", (PyCFunction)LocalCoord_ecef2ned_single, METH_O, "Convert ecef to ned"},
  {"ned2ecef_single", (PyCFunction)LocalCoord_ned2ecef_single, METH_O, "Convert ned to ecef"},
  {"geodetic2ned_single", (PyCFunction)LocalCoord_geodetic2ned_single, METH_O, "Convert geodetic to ned"},
  {"ned2geodetic_single", (PyCFunction)LocalCoord_ned2geodetic_single, METH_O, "Convert ned to geodetic"},
  {NULL}
};

static PyGetSetDef LocalCoord_getset[] = {
    {"ned2ecef_matrix", (getter)LocalCoord_get_ned2ecef_matrix, NULL, "NED to ECEF matrix", NULL},
    {"ecef2ned_matrix", (getter)LocalCoord_get_ecef2ned_matrix, NULL, "ECEF to NED matrix", NULL},
    {"ned_from_ecef_matrix", (getter)LocalCoord_get_ecef2ned_matrix, NULL, "Alias for ecef2ned_matrix", NULL},
    {"ecef_from_ned_matrix", (getter)LocalCoord_get_ned2ecef_matrix, NULL, "Alias for ned2ecef_matrix", NULL},
    {NULL}
};

static PyTypeObject LocalCoordType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "transformations.LocalCoord",
  .tp_basicsize = sizeof(LocalCoordObject),
  .tp_dealloc = (destructor)LocalCoord_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_methods = LocalCoord_methods,
  .tp_getset = LocalCoord_getset,
  .tp_init = (initproc)LocalCoord_init,
  .tp_new = PyType_GenericNew,
};

static PyMethodDef transformations_methods[] = {
  {"euler2quat_single", (PyCFunction)meth_euler2quat_single, METH_VARARGS, ""},
  {"quat2euler_single", (PyCFunction)meth_quat2euler_single, METH_VARARGS, ""},
  {"quat2rot_single", (PyCFunction)meth_quat2rot_single, METH_VARARGS, ""},
  {"rot2quat_single", (PyCFunction)meth_rot2quat_single, METH_VARARGS, ""},
  {"euler2rot_single", (PyCFunction)meth_euler2rot_single, METH_VARARGS, ""},
  {"rot2euler_single", (PyCFunction)meth_rot2euler_single, METH_VARARGS, ""},
  {"rot_matrix", (PyCFunction)meth_rot_matrix, METH_VARARGS, ""},
  {"ecef_euler_from_ned_single", (PyCFunction)meth_ecef_euler_from_ned_single, METH_VARARGS, ""},
  {"ned_euler_from_ecef_single", (PyCFunction)meth_ned_euler_from_ecef_single, METH_VARARGS, ""},
  {"geodetic2ecef_single", (PyCFunction)meth_geodetic2ecef_single, METH_VARARGS, ""},
  {"ecef2geodetic_single", (PyCFunction)meth_ecef2geodetic_single, METH_VARARGS, ""},
  {NULL}
};

static struct PyModuleDef transformations_module = {
  PyModuleDef_HEAD_INIT,
  "transformations",
  NULL,
  -1,
  transformations_methods
};

PyMODINIT_FUNC PyInit_transformations(void) {
  import_array();
  PyObject *m;
  if (PyType_Ready(&LocalCoordType) < 0)
    return NULL;

  m = PyModule_Create(&transformations_module);
  if (m == NULL)
    return NULL;

  Py_INCREF(&LocalCoordType);
  if (PyModule_AddObject(m, "LocalCoord", (PyObject *)&LocalCoordType) < 0) {
    Py_DECREF(&LocalCoordType);
    Py_DECREF(m);
    return NULL;
  }

  return m;
}
