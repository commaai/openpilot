#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <string>
#include <cstring>
#include "selfdrive/pandad/can_types.h"

void can_list_to_can_capnp_cpp(const std::vector<CanFrame> &can_list, std::string &out, bool sendcan, bool valid);
void can_capnp_to_can_list_cpp(const std::vector<std::string> &strings, std::vector<CanData> &can_list, bool sendcan);

static bool parse_can_msgs(PyObject *can_msgs, std::vector<CanFrame> &frames) {
  if (!PyList_Check(can_msgs)) {
    PyErr_SetString(PyExc_TypeError, "can_msgs must be a list");
    return false;
  }

  Py_ssize_t len = PyList_Size(can_msgs);
  frames.reserve(len);

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject *item = PyList_GetItem(can_msgs, i);
    if (!PySequence_Check(item) || PySequence_Size(item) < 3) {
       PyErr_SetString(PyExc_ValueError, "Each CAN message must be a sequence of length at least 3 (address, data, src)");
       return false;
    }

    PyObject *addr_obj = PySequence_GetItem(item, 0);
    PyObject *dat_obj = PySequence_GetItem(item, 1);
    PyObject *src_obj = PySequence_GetItem(item, 2);

    CanFrame frame;
    frame.address = (uint32_t)PyLong_AsUnsignedLong(addr_obj);

    char *buffer = NULL;
    Py_ssize_t length = 0;
    if (PyBytes_Check(dat_obj)) {
        if (PyBytes_AsStringAndSize(dat_obj, &buffer, &length) < 0) {
            Py_DECREF(addr_obj); Py_DECREF(dat_obj); Py_DECREF(src_obj);
            return false;
        }
        frame.dat.assign((uint8_t*)buffer, (uint8_t*)buffer + length);
    } else {
        Py_DECREF(addr_obj); Py_DECREF(dat_obj); Py_DECREF(src_obj);
        PyErr_SetString(PyExc_TypeError, "CAN data must be bytes");
        return false;
    }

    frame.src = PyLong_AsLong(src_obj);

    Py_DECREF(addr_obj);
    Py_DECREF(dat_obj);
    Py_DECREF(src_obj);

    if (PyErr_Occurred()) return false;

    frames.push_back(frame);
  }
  return true;
}

static PyObject* method_can_list_to_can_capnp(PyObject *self, PyObject *args, PyObject *kwds) {
  PyObject *can_msgs;
  char *msgtype_str = (char*)"can";
  int valid = 1;
  static char *kwlist[] = {(char*)"can_msgs", (char*)"msgtype", (char*)"valid", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|sp", kwlist, &can_msgs, &msgtype_str, &valid)) {
    return NULL;
  }

  std::vector<CanFrame> frames;
  if (!parse_can_msgs(can_msgs, frames)) return NULL;

  bool sendcan = (strcmp(msgtype_str, "sendcan") == 0);

  std::string out;
  {
    Py_BEGIN_ALLOW_THREADS
    can_list_to_can_capnp_cpp(frames, out, sendcan, valid);
    Py_END_ALLOW_THREADS
  }

  return PyBytes_FromStringAndSize(out.data(), out.size());
}

static PyObject* method_can_capnp_to_list(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *strings_obj;
    char *msgtype_str = (char*)"can";
    static char *kwlist[] = {(char*)"strings", (char*)"msgtype", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|s", kwlist, &strings_obj, &msgtype_str)) {
        return NULL;
    }

    std::vector<std::string> strings;
    if (!PySequence_Check(strings_obj)) {
         PyErr_SetString(PyExc_TypeError, "strings must be a list or tuple");
         return NULL;
    }
    Py_ssize_t len = PySequence_Size(strings_obj);
    strings.reserve(len);
    for (Py_ssize_t i=0; i<len; ++i) {
        PyObject *item = PySequence_GetItem(strings_obj, i);
        if (!PyBytes_Check(item)) {
             PyErr_SetString(PyExc_TypeError, "items in strings must be bytes");
             Py_DECREF(item);
             return NULL;
        }
        char *ptr;
        Py_ssize_t size;
        if (PyBytes_AsStringAndSize(item, &ptr, &size) < 0) {
             Py_DECREF(item);
             return NULL;
        }
        strings.emplace_back(ptr, size);
        Py_DECREF(item);
    }

    bool sendcan = (strcmp(msgtype_str, "sendcan") == 0);
    std::vector<CanData> can_data;

    {
        Py_BEGIN_ALLOW_THREADS
        can_capnp_to_can_list_cpp(strings, can_data, sendcan);
        Py_END_ALLOW_THREADS
    }

    PyObject *result = PyList_New(can_data.size());
    if (!result) return NULL;

    for (size_t i=0; i < can_data.size(); ++i) {
        const auto &cd = can_data[i];
        PyObject *frames_list = PyList_New(cd.frames.size());
        if (!frames_list) {
            Py_DECREF(result);
            return NULL;
        }

        for (size_t j=0; j < cd.frames.size(); ++j) {
            const auto &f = cd.frames[j];
            PyObject *frame_tuple = PyTuple_New(3);
            if (!frame_tuple) {
                Py_DECREF(frames_list);
                Py_DECREF(result);
                return NULL;
            }

            PyTuple_SetItem(frame_tuple, 0, PyLong_FromUnsignedLong(f.address));
            PyTuple_SetItem(frame_tuple, 1, PyBytes_FromStringAndSize((char*)f.dat.data(), f.dat.size()));
            PyTuple_SetItem(frame_tuple, 2, PyLong_FromLong(f.src));
            PyList_SetItem(frames_list, j, frame_tuple);
        }

        PyObject *entry = PyTuple_New(2);
        if (!entry) {
             Py_DECREF(frames_list);
             Py_DECREF(result);
             return NULL;
        }
        PyTuple_SetItem(entry, 0, PyLong_FromUnsignedLongLong(cd.nanos));
        PyTuple_SetItem(entry, 1, frames_list);

        PyList_SetItem(result, i, entry);
    }
    return result;
}

static PyMethodDef methods[] = {
  {"can_list_to_can_capnp", (PyCFunction)method_can_list_to_can_capnp, METH_VARARGS | METH_KEYWORDS, "Convert list of can messages to capnp"},
  {"can_capnp_to_list", (PyCFunction)method_can_capnp_to_list, METH_VARARGS | METH_KEYWORDS, "Convert capnp messages to list"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "_pandad_api_impl",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit__pandad_api_impl(void) {
  return PyModule_Create(&module);
}
