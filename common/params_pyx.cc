#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <datetime.h>
#include <structmember.h>

#include "common/params.h"

static PyObject *UnknownKeyName;
static PyObject *json_module;
static PyObject *json_dumps;
static PyObject *json_loads;

static PyTypeObject ParamKeyFlagType = {
  PyVarObject_HEAD_INIT(NULL, 0)
};

static PyTypeObject ParamKeyTypeType = {
  PyVarObject_HEAD_INIT(NULL, 0)
};

typedef struct {
  PyObject_HEAD
  Params *p;
  PyObject *d;
} ParamsObject;

static void Params_dealloc(ParamsObject *self) {
  if (self->p) {
    delete self->p;
  }
  Py_XDECREF(self->d);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static int Params_init(ParamsObject *self, PyObject *args, PyObject *kwds) {
  const char *d = "";
  if (!PyArg_ParseTuple(args, "|s", &d)) {
    return -1;
  }
  self->d = PyUnicode_FromString(d);
  if (!self->d) return -1;

  std::string path(d);
  Py_BEGIN_ALLOW_THREADS
  self->p = new Params(path);
  Py_END_ALLOW_THREADS
  return 0;
}

static PyObject* PyUnicode_FromStringAndSize_Bytes(const std::string& s) {
  return PyUnicode_FromStringAndSize(s.c_str(), s.size());
}

static PyObject* cpp2python(ParamKeyType type, const std::string& value) {
  switch (type) {
    case ParamKeyType::STRING:
      return PyUnicode_FromStringAndSize_Bytes(value);
    case ParamKeyType::BOOL:
      return PyBool_FromLong(value == "1");
    case ParamKeyType::INT:
      try {
        return PyLong_FromLong(std::stoi(value));
      } catch (...) {
        return NULL;
      }
    case ParamKeyType::FLOAT:
      try {
        return PyFloat_FromDouble(std::stof(value));
      } catch (...) {
        return NULL;
      }
    case ParamKeyType::TIME: {
      PyObject *dt_module = PyImport_ImportModule("datetime");
      if (!dt_module) return NULL;
      PyObject *dt_class = PyObject_GetAttrString(dt_module, "datetime");
      Py_DECREF(dt_module);
      if (!dt_class) return NULL;
      PyObject *val_str = PyUnicode_FromStringAndSize_Bytes(value);
      PyObject *res = PyObject_CallMethod(dt_class, "fromisoformat", "O", val_str);
      Py_DECREF(dt_class);
      Py_DECREF(val_str);
      if (!res) PyErr_Clear(); // Clear error to allow fallback
      return res;
    }
    case ParamKeyType::JSON: {
      PyObject *val_str = PyUnicode_FromStringAndSize_Bytes(value);
      PyObject *res = PyObject_CallFunctionObjArgs(json_loads, val_str, NULL);
      Py_DECREF(val_str);
      if (!res) PyErr_Clear();
      return res;
    }
    case ParamKeyType::BYTES:
      return PyBytes_FromStringAndSize(value.c_str(), value.size());
  }
  Py_RETURN_NONE;
}

static std::string python2cpp(ParamKeyType type, PyObject* value) {
  switch (type) {
    case ParamKeyType::STRING:
      if (PyUnicode_Check(value)) {
        return PyUnicode_AsUTF8(value);
      }
      break;
    case ParamKeyType::BOOL:
      return PyObject_IsTrue(value) ? "1" : "0";
    case ParamKeyType::INT:
    case ParamKeyType::FLOAT: {
       PyObject* str_val = PyObject_Str(value);
       if (str_val) {
         std::string s = PyUnicode_AsUTF8(str_val);
         Py_DECREF(str_val);
         return s;
       }
       break;
    }
    case ParamKeyType::TIME: {
      if (PyObject_HasAttrString(value, "isoformat")) {
        PyObject* iso = PyObject_CallMethod(value, "isoformat", NULL);
        if (iso) {
           std::string s = PyUnicode_AsUTF8(iso);
           Py_DECREF(iso);
           return s;
        }
      }
      break;
    }
    case ParamKeyType::JSON: {
      PyObject* dumped = PyObject_CallFunctionObjArgs(json_dumps, value, NULL);
      if (dumped) {
         std::string s = PyUnicode_AsUTF8(dumped);
         Py_DECREF(dumped);
         return s;
      }
      break;
    }
    case ParamKeyType::BYTES:
       if (PyBytes_Check(value)) {
         return std::string(PyBytes_AS_STRING(value), PyBytes_GET_SIZE(value));
       }
       if (PyUnicode_Check(value)) {
          return PyUnicode_AsUTF8(value);
       }
       break;
  }

  if (type == ParamKeyType::BYTES) {
      if (PyBytes_Check(value)) return std::string(PyBytes_AS_STRING(value), PyBytes_GET_SIZE(value));
  }

  return "";
}

static std::string ensure_bytes(PyObject* v) {
  if (PyUnicode_Check(v)) {
    return PyUnicode_AsUTF8(v);
  } else if (PyBytes_Check(v)) {
    return std::string(PyBytes_AS_STRING(v), PyBytes_GET_SIZE(v));
  }
  return "";
}

static std::string check_key(ParamsObject* self, PyObject* key) {
  std::string k = ensure_bytes(key);
  if (!self->p->checkKey(k)) {
    PyErr_SetString(UnknownKeyName, k.c_str());
    return "";
  }
  return k;
}

static PyObject* Params_check_key(ParamsObject* self, PyObject* key) {
  std::string k = check_key(self, key);
  if (PyErr_Occurred()) return NULL;
  return PyBytes_FromStringAndSize(k.c_str(), k.size());
}

static PyObject* Params_clear_all(ParamsObject* self, PyObject* args, PyObject* kwds) {
  static char *kwlist[] = {(char *)"tx_flag", NULL};
  int tx_flag = ParamKeyFlag::ALL;
  PyObject* tx_flag_obj = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &tx_flag_obj)) {
    return NULL;
  }
  if (tx_flag_obj) {
    tx_flag = PyLong_AsLong(tx_flag_obj);
  }

  self->p->clearAll((ParamKeyFlag)tx_flag);
  Py_RETURN_NONE;
}

static PyObject* Params_get(ParamsObject* self, PyObject* args, PyObject* kwds) {
  static char *kwlist[] = {(char *)"key", (char *)"block", (char *)"return_default", NULL};
  PyObject* key_obj;
  int block = 0;
  int return_default = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|pp", kwlist, &key_obj, &block, &return_default)) {
    return NULL;
  }

  std::string k = check_key(self, key_obj);
  if (PyErr_Occurred()) return NULL;

  ParamKeyType t = self->p->getKeyType(k);
  std::optional<std::string> default_val_opt = self->p->getKeyDefaultValue(k);

  std::string val;
  {
    Py_BEGIN_ALLOW_THREADS
    val = self->p->get(k, block);
    Py_END_ALLOW_THREADS
  }

  if (val.empty()) {
    if (block) {
       PyErr_SetNone(PyExc_KeyboardInterrupt);
       return NULL;
    } else {
       if (return_default && default_val_opt.has_value()) {
         return cpp2python(t, default_val_opt.value());
       }
       Py_RETURN_NONE;
    }
  }

  PyObject* ret = cpp2python(t, val);
  if (!ret) {
    if (default_val_opt.has_value()) {
       return cpp2python(t, default_val_opt.value());
    }
    Py_RETURN_NONE;
  }
  return ret;
}

static PyObject* Params_get_bool(ParamsObject* self, PyObject* args, PyObject* kwds) {
  static char *kwlist[] = {(char *)"key", (char *)"block", NULL};
  PyObject* key_obj;
  int block = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|p", kwlist, &key_obj, &block)) {
    return NULL;
  }

  std::string k = check_key(self, key_obj);
  if (PyErr_Occurred()) return NULL;

  bool r;
  Py_BEGIN_ALLOW_THREADS
  r = self->p->getBool(k, block);
  Py_END_ALLOW_THREADS

  if (r) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyObject* Params_put(ParamsObject* self, PyObject* args, PyObject* kwds) {
  static char *kwlist[] = {(char *)"key", (char *)"dat", NULL};
  PyObject* key_obj;
  PyObject* dat_obj;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &key_obj, &dat_obj)) {
    return NULL;
  }

  std::string k = check_key(self, key_obj);
  if (PyErr_Occurred()) return NULL;

  ParamKeyType t = self->p->getKeyType(k);
  std::string dat_bytes = python2cpp(t, dat_obj);

  Py_BEGIN_ALLOW_THREADS
  self->p->put(k, dat_bytes);
  Py_END_ALLOW_THREADS

  Py_RETURN_NONE;
}

static PyObject* Params_put_bool(ParamsObject* self, PyObject* args, PyObject* kwds) {
  static char *kwlist[] = {(char *)"key", (char *)"val", NULL};
  PyObject* key_obj;
  int val;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Op", kwlist, &key_obj, &val)) {
    return NULL;
  }

  std::string k = check_key(self, key_obj);
  if (PyErr_Occurred()) return NULL;

  Py_BEGIN_ALLOW_THREADS
  self->p->putBool(k, val);
  Py_END_ALLOW_THREADS

  Py_RETURN_NONE;
}


static PyObject* Params_put_nonblocking(ParamsObject* self, PyObject* args, PyObject* kwds) {
  static char *kwlist[] = {(char *)"key", (char *)"dat", NULL};
  PyObject* key_obj;
  PyObject* dat_obj;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &key_obj, &dat_obj)) {
    return NULL;
  }

  std::string k = check_key(self, key_obj);
  if (PyErr_Occurred()) return NULL;

  ParamKeyType t = self->p->getKeyType(k);
  std::string dat_bytes = python2cpp(t, dat_obj);

  Py_BEGIN_ALLOW_THREADS
  self->p->putNonBlocking(k, dat_bytes);
  Py_END_ALLOW_THREADS

  Py_RETURN_NONE;
}

static PyObject* Params_put_bool_nonblocking(ParamsObject* self, PyObject* args, PyObject* kwds) {
  static char *kwlist[] = {(char *)"key", (char *)"val", NULL};
  PyObject* key_obj;
  int val;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Op", kwlist, &key_obj, &val)) {
    return NULL;
  }

  std::string k = check_key(self, key_obj);
  if (PyErr_Occurred()) return NULL;

  Py_BEGIN_ALLOW_THREADS
  self->p->putBoolNonBlocking(k, val);
  Py_END_ALLOW_THREADS

  Py_RETURN_NONE;
}

static PyObject* Params_remove(ParamsObject* self, PyObject* key) {
  std::string k = check_key(self, key);
  if (PyErr_Occurred()) return NULL;

  Py_BEGIN_ALLOW_THREADS
  self->p->remove(k);
  Py_END_ALLOW_THREADS

  Py_RETURN_NONE;
}

static PyObject* Params_get_param_path(ParamsObject* self, PyObject* args, PyObject* kwds) {
  static char *kwlist[] = {(char *)"key", NULL};
  PyObject* key_obj = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &key_obj)) {
    return NULL;
  }

  std::string k;
  if (key_obj && key_obj != Py_None) {
      k = ensure_bytes(key_obj);
  }

  std::string path = self->p->getParamPath(k);
  return PyUnicode_FromString(path.c_str());
}

static PyObject* Params_get_type(ParamsObject* self, PyObject* key) {
  std::string k = check_key(self, key);
  if (PyErr_Occurred()) return NULL;

  ParamKeyType t = self->p->getKeyType(k);
  return PyLong_FromLong((long)t);
}

static PyObject* Params_all_keys(ParamsObject* self) {
  std::vector<std::string> keys = self->p->allKeys();
  PyObject* list = PyList_New(keys.size());
  for (size_t i = 0; i < keys.size(); i++) {
    PyList_SetItem(list, i, PyBytes_FromStringAndSize(keys[i].c_str(), keys[i].size()));
  }
  return list;
}

static PyObject* Params_get_default_value(ParamsObject* self, PyObject* key) {
  std::string k = check_key(self, key);
  if (PyErr_Occurred()) return NULL;

  ParamKeyType t = self->p->getKeyType(k);
  std::optional<std::string> default_val = self->p->getKeyDefaultValue(k);
  if (default_val.has_value()) {
    return cpp2python(t, default_val.value());
  }
  Py_RETURN_NONE;
}

static PyObject* Params_reduce(ParamsObject* self) {
  return Py_BuildValue("(O(O))", Py_TYPE(self), self->d);
}

static PyMethodDef Params_methods[] = {
  {"__reduce__", (PyCFunction)Params_reduce, METH_NOARGS, "Pickle support"},
  {"clear_all", (PyCFunction)Params_clear_all, METH_VARARGS | METH_KEYWORDS, "Clear all params"},
  {"check_key", (PyCFunction)Params_check_key, METH_O, "Check if key exists"},
  {"get", (PyCFunction)Params_get, METH_VARARGS | METH_KEYWORDS, "Get param value"},
  {"get_bool", (PyCFunction)Params_get_bool, METH_VARARGS | METH_KEYWORDS, "Get param bool value"},
  {"put", (PyCFunction)Params_put, METH_VARARGS | METH_KEYWORDS, "Put param value"},
  {"put_bool", (PyCFunction)Params_put_bool, METH_VARARGS | METH_KEYWORDS, "Put param bool value"},
  {"put_nonblocking", (PyCFunction)Params_put_nonblocking, METH_VARARGS | METH_KEYWORDS, "Put param value non-blocking"},
  {"put_bool_nonblocking", (PyCFunction)Params_put_bool_nonblocking, METH_VARARGS | METH_KEYWORDS, "Put param bool value non-blocking"},
  {"remove", (PyCFunction)Params_remove, METH_O, "Remove param"},
  {"get_param_path", (PyCFunction)Params_get_param_path, METH_VARARGS | METH_KEYWORDS, "Get param path"},
  {"get_type", (PyCFunction)Params_get_type, METH_O, "Get param type"},
  {"all_keys", (PyCFunction)Params_all_keys, METH_NOARGS, "Get all keys"},
  {"get_default_value", (PyCFunction)Params_get_default_value, METH_O, "Get default value"},
  {NULL}
};

static PyTypeObject ParamsType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  .tp_name = "params_pyx.Params",
  .tp_basicsize = sizeof(ParamsObject),
  .tp_dealloc = (destructor)Params_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_methods = Params_methods,
  .tp_init = (initproc)Params_init,
  .tp_new = PyType_GenericNew,
};

static PyMethodDef params_functions[] = {
  {NULL}
};

static struct PyModuleDef params_module = {
  PyModuleDef_HEAD_INIT,
  "params_pyx",
  NULL,
  -1,
  params_functions
};

PyMODINIT_FUNC PyInit_params_pyx(void) {
  PyObject *m;
  if (PyType_Ready(&ParamsType) < 0)
    return NULL;

  m = PyModule_Create(&params_module);
  if (m == NULL)
    return NULL;

  Py_INCREF(&ParamsType);
  if (PyModule_AddObject(m, "Params", (PyObject *)&ParamsType) < 0) {
    Py_DECREF(&ParamsType);
    Py_DECREF(m);
    return NULL;
  }

  // Initialize ParamKeyFlagType
  ParamKeyFlagType.tp_name = "params_pyx.ParamKeyFlag";
  ParamKeyFlagType.tp_basicsize = sizeof(PyObject);
  ParamKeyFlagType.tp_flags = Py_TPFLAGS_DEFAULT;
  ParamKeyFlagType.tp_doc = "ParamKeyFlag";

  if (PyType_Ready(&ParamKeyFlagType) < 0) return NULL;

  // Add constants to ParamKeyFlag
  PyObject *dict = ParamKeyFlagType.tp_dict;
  PyDict_SetItemString(dict, "PERSISTENT", PyLong_FromLong(ParamKeyFlag::PERSISTENT));
  PyDict_SetItemString(dict, "CLEAR_ON_MANAGER_START", PyLong_FromLong(ParamKeyFlag::CLEAR_ON_MANAGER_START));
  PyDict_SetItemString(dict, "CLEAR_ON_ONROAD_TRANSITION", PyLong_FromLong(ParamKeyFlag::CLEAR_ON_ONROAD_TRANSITION));
  PyDict_SetItemString(dict, "CLEAR_ON_OFFROAD_TRANSITION", PyLong_FromLong(ParamKeyFlag::CLEAR_ON_OFFROAD_TRANSITION));
  PyDict_SetItemString(dict, "DEVELOPMENT_ONLY", PyLong_FromLong(ParamKeyFlag::DEVELOPMENT_ONLY));
  PyDict_SetItemString(dict, "CLEAR_ON_IGNITION_ON", PyLong_FromLong(ParamKeyFlag::CLEAR_ON_IGNITION_ON));
  PyDict_SetItemString(dict, "ALL", PyLong_FromUnsignedLong(ParamKeyFlag::ALL));

  Py_INCREF(&ParamKeyFlagType);
  if (PyModule_AddObject(m, "ParamKeyFlag", (PyObject *)&ParamKeyFlagType) < 0) {
      Py_DECREF(&ParamKeyFlagType);
      return NULL;
  }

  // Initialize ParamKeyTypeType
  ParamKeyTypeType.tp_name = "params_pyx.ParamKeyType";
  ParamKeyTypeType.tp_basicsize = sizeof(PyObject);
  ParamKeyTypeType.tp_flags = Py_TPFLAGS_DEFAULT;
  ParamKeyTypeType.tp_doc = "ParamKeyType";

  if (PyType_Ready(&ParamKeyTypeType) < 0) return NULL;

  // Add constants to ParamKeyType
  dict = ParamKeyTypeType.tp_dict;
  PyDict_SetItemString(dict, "STRING", PyLong_FromLong(ParamKeyType::STRING));
  PyDict_SetItemString(dict, "BOOL", PyLong_FromLong(ParamKeyType::BOOL));
  PyDict_SetItemString(dict, "INT", PyLong_FromLong(ParamKeyType::INT));
  PyDict_SetItemString(dict, "FLOAT", PyLong_FromLong(ParamKeyType::FLOAT));
  PyDict_SetItemString(dict, "TIME", PyLong_FromLong(ParamKeyType::TIME));
  PyDict_SetItemString(dict, "JSON", PyLong_FromLong(ParamKeyType::JSON));
  PyDict_SetItemString(dict, "BYTES", PyLong_FromLong(ParamKeyType::BYTES));

  Py_INCREF(&ParamKeyTypeType);
  if (PyModule_AddObject(m, "ParamKeyType", (PyObject *)&ParamKeyTypeType) < 0) {
      Py_DECREF(&ParamKeyTypeType);
      return NULL;
  }

  UnknownKeyName = PyErr_NewException("params_pyx.UnknownKeyName", NULL, NULL);
  Py_INCREF(UnknownKeyName);
  PyModule_AddObject(m, "UnknownKeyName", UnknownKeyName);

  // Initialize standard modules
  json_module = PyImport_ImportModule("json");
  if (json_module) {
     json_dumps = PyObject_GetAttrString(json_module, "dumps");
     json_loads = PyObject_GetAttrString(json_module, "loads");
  }

  PyDateTime_IMPORT;

  return m;
}
