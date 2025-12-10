
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "selfdrive/modeld/models/commonmodel.h"


// Global Type Pointers
static PyTypeObject *CLContextType = NULL;
static PyTypeObject *CLMemType = NULL;
static PyTypeObject *ModelFrameType = NULL;
static PyTypeObject *DrivingModelFrameType = NULL;
static PyTypeObject *MonitoringModelFrameType = NULL;

// --- CLContext ---
typedef struct {
  PyObject_HEAD
  cl_device_id device_id;
  cl_context context;
} CLContext;

static void CLContext_dealloc(CLContext *self) {
  if (self->context) {
    clReleaseContext(self->context);
  }
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static int CLContext_init(CLContext *self, PyObject *args, PyObject *kwds) {
  self->device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
  self->context = cl_create_context(self->device_id);
  return 0;
}

static PyObject *CLContext_get_device_id(CLContext *self, void *closure) {
  return PyLong_FromUnsignedLongLong((unsigned long long)self->device_id);
}

static PyObject *CLContext_get_context(CLContext *self, void *closure) {
  return PyLong_FromUnsignedLongLong((unsigned long long)self->context);
}

static PyGetSetDef CLContext_getset[] = {
    {"device_id", (getter)CLContext_get_device_id, NULL, "OpenCL Device ID", NULL},
    {"context", (getter)CLContext_get_context, NULL, "OpenCL Context", NULL},
    {NULL}
};

static PyType_Slot CLContext_slots[] = {
    {Py_tp_dealloc, (void*)CLContext_dealloc},
    {Py_tp_init, (void*)CLContext_init},
    {Py_tp_getset, CLContext_getset},
    {Py_tp_new, (void*)PyType_GenericNew},
    {0, 0}
};

static PyType_Spec CLContext_spec = {
    "commonmodel_module.CLContext",
    sizeof(CLContext),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    CLContext_slots
};

// --- CLMem ---
typedef struct {
  PyObject_HEAD
  cl_mem mem;
} CLMem;

static void CLMem_dealloc(CLMem *self) {
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *CLMem_get_mem_address(CLMem *self, void *closure) {
  return PyLong_FromUnsignedLongLong((unsigned long long)self->mem);
}

static PyGetSetDef CLMem_getset[] = {
    {"mem_address", (getter)CLMem_get_mem_address, NULL, "CL Mem Address", NULL},
    {NULL}
};

static PyObject *CLMem_create(PyTypeObject *cls, PyObject *args) {
    unsigned long long mem_ptr;
    if (!PyArg_ParseTuple(args, "K", &mem_ptr)) return NULL;

    CLMem *obj = (CLMem *)cls->tp_alloc(cls, 0);
    if (obj) {
        obj->mem = (cl_mem)mem_ptr;
    }
    return (PyObject *)obj;
}

static PyMethodDef CLMem_methods[] = {
    {"create", (PyCFunction)CLMem_create, METH_VARARGS | METH_CLASS, "Create CLMem from pointer"},
    {NULL}
};

static PyType_Slot CLMem_slots[] = {
    {Py_tp_dealloc, (void*)CLMem_dealloc},
    {Py_tp_getset, CLMem_getset},
    {Py_tp_methods, CLMem_methods},
    {Py_tp_new, (void*)PyType_GenericNew},
    {0, 0}
};

static PyType_Spec CLMem_spec = {
    "commonmodel_module.CLMem",
    sizeof(CLMem),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    CLMem_slots
};

static PyObject *cl_from_visionbuf(PyObject *self, PyObject *visionbuf) {
    PyObject *buf_cl_obj = PyObject_GetAttrString(visionbuf, "buf_cl");
    if (!buf_cl_obj) {
        PyErr_SetString(PyExc_TypeError, "VisionBuf object has no 'buf_cl' attribute");
        return NULL;
    }

    unsigned long long buf_cl = PyLong_AsUnsignedLongLong(buf_cl_obj);
    Py_DECREF(buf_cl_obj);
    if (PyErr_Occurred()) return NULL;

    PyObject *args = PyTuple_Pack(1, PyLong_FromUnsignedLongLong(buf_cl));
    PyObject *clmem = PyObject_CallMethod((PyObject*)CLMemType, "create", "K", buf_cl);
    Py_DECREF(args);
    return clmem;
}

// --- ModelFrame (Base) ---
typedef struct {
  PyObject_HEAD
  ModelFrame *frame;
  int buf_size;
} PyModelFrame;

static void ModelFrame_dealloc(PyModelFrame *self) {
  if (self->frame) {
      delete self->frame;
  }
  Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject *ModelFrame_prepare(PyModelFrame *self, PyObject *args) {
    PyObject *visionbuf;
    PyObject *projection_obj; // memoryview or buffer

    if (!PyArg_ParseTuple(args, "OO", &visionbuf, &projection_obj)) return NULL;

    auto get_int = [&](const char* name) -> size_t {
        PyObject *attr = PyObject_GetAttrString(visionbuf, name);
        if (!attr) return 0; // Error set
        size_t val = PyLong_AsSize_t(attr);
        Py_DECREF(attr);
        return val;
    };

    cl_mem buf_cl = (cl_mem)get_int("buf_cl");
    if (PyErr_Occurred()) return NULL;
    int width = (int)get_int("width");
    int height = (int)get_int("height");
    int stride = (int)get_int("stride");
    int uv_offset = (int)get_int("uv_offset");

    // Extract projection matrix
    Py_buffer view;
    if (PyObject_GetBuffer(projection_obj, &view, PyBUF_SIMPLE) < 0) return NULL;

    mat3 cprojection;
    if (view.len == 9 * sizeof(float)) {
        memcpy(cprojection.v, view.buf, 9 * sizeof(float));
    } else {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Projection matrix must be 9 floats");
        return NULL;
    }
    PyBuffer_Release(&view);

    cl_mem *data_ptr = self->frame->prepare(buf_cl, width, height, stride, uv_offset, cprojection);

    if (!data_ptr) Py_RETURN_NONE;

    PyObject *res = PyObject_CallMethod((PyObject*)CLMemType, "create", "K", (unsigned long long)*data_ptr);
    return res;
}

static PyObject *numpy_module = NULL;

static PyObject *ModelFrame_buffer_from_cl(PyModelFrame *self, PyObject *args) {
    PyObject *clmem_obj;
    if (!PyArg_ParseTuple(args, "O", &clmem_obj)) return NULL;

    if (!PyObject_TypeCheck(clmem_obj, CLMemType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be CLMem");
        return NULL;
    }

    CLMem* clmem = (CLMem*)clmem_obj;
    unsigned char* data2 = self->frame->buffer_from_cl(&clmem->mem, self->buf_size);

    if (!numpy_module) {
        numpy_module = PyImport_ImportModule("numpy");
        if (!numpy_module) return NULL;
    }

    PyObject *memview = PyMemoryView_FromMemory((char*)data2, self->buf_size, PyBUF_READ);
    if (!memview) return NULL;

    PyObject *dtype_str = PyUnicode_FromString("uint8");
    PyObject *args_np = PyTuple_Pack(2, memview, dtype_str);
    PyObject *ret = PyObject_CallMethod(numpy_module, "array", "OO", memview, dtype_str);

    Py_DECREF(memview);
    Py_DECREF(dtype_str);
    Py_DECREF(args_np);

    return ret;
}

static PyMethodDef ModelFrame_methods[] = {
    {"prepare", (PyCFunction)ModelFrame_prepare, METH_VARARGS, ""},
    {"buffer_from_cl", (PyCFunction)ModelFrame_buffer_from_cl, METH_VARARGS, ""},
    {NULL}
};

static PyType_Slot ModelFrame_slots[] = {
    {Py_tp_dealloc, (void*)ModelFrame_dealloc},
    {Py_tp_methods, ModelFrame_methods},
    {Py_tp_new, (void*)PyType_GenericNew},
    {0, 0}
};

static PyType_Spec ModelFrame_spec = {
    "commonmodel_module.ModelFrame",
    sizeof(PyModelFrame),
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    ModelFrame_slots
};

// --- DrivingModelFrame ---
static int DrivingModelFrame_init(PyModelFrame *self, PyObject *args, PyObject *kwds) {
    PyObject *ctx_obj;
    int temporal_skip;
    if (!PyArg_ParseTuple(args, "Oi", &ctx_obj, &temporal_skip)) return -1;

    if (!PyObject_TypeCheck(ctx_obj, CLContextType)) {
        PyErr_SetString(PyExc_TypeError, "Context must be CLContext");
        return -1;
    }
    CLContext *ctx = (CLContext*)ctx_obj;

    self->frame = new DrivingModelFrame(ctx->device_id, ctx->context, temporal_skip);
    self->buf_size = self->frame->buf_size;
    return 0;
}

static PyType_Slot DrivingModelFrame_slots[] = {
    {Py_tp_init, (void*)DrivingModelFrame_init},
    {0, 0}
};

static PyType_Spec DrivingModelFrame_spec = {
    "commonmodel_module.DrivingModelFrame",
    sizeof(PyModelFrame),
    0,
    Py_TPFLAGS_DEFAULT,
    DrivingModelFrame_slots
};

// --- MonitoringModelFrame ---
static int MonitoringModelFrame_init(PyModelFrame *self, PyObject *args) {
    PyObject *ctx_obj;
    if (!PyArg_ParseTuple(args, "O", &ctx_obj)) return -1;

     if (!PyObject_TypeCheck(ctx_obj, CLContextType)) {
        PyErr_SetString(PyExc_TypeError, "Context must be CLContext");
        return -1;
    }
    CLContext *ctx = (CLContext*)ctx_obj;

    self->frame = new MonitoringModelFrame(ctx->device_id, ctx->context);
    self->buf_size = self->frame->buf_size;
    return 0;
}

static PyType_Slot MonitoringModelFrame_slots[] = {
    {Py_tp_init, (void*)MonitoringModelFrame_init},
    {0, 0}
};

static PyType_Spec MonitoringModelFrame_spec = {
    "commonmodel_module.MonitoringModelFrame",
    sizeof(PyModelFrame),
    0,
    Py_TPFLAGS_DEFAULT,
    MonitoringModelFrame_slots
};

static PyMethodDef module_methods[] = {
    {"cl_from_visionbuf", (PyCFunction)cl_from_visionbuf, METH_O, ""},
    {NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_commonmodel_module",
    NULL,
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__commonmodel_module(void) {
    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;

    CLContextType = (PyTypeObject*)PyType_FromSpec(&CLContext_spec);
    if (PyModule_AddObject(m, "CLContext", (PyObject *)CLContextType) < 0) return NULL;
    Py_INCREF(CLContextType);

    CLMemType = (PyTypeObject*)PyType_FromSpec(&CLMem_spec);
    if (PyModule_AddObject(m, "CLMem", (PyObject *)CLMemType) < 0) return NULL;
    Py_INCREF(CLMemType);

    ModelFrameType = (PyTypeObject*)PyType_FromSpec(&ModelFrame_spec);
    if (PyModule_AddObject(m, "ModelFrame", (PyObject *)ModelFrameType) < 0) return NULL;
    Py_INCREF(ModelFrameType);

    PyObject *bases = PyTuple_Pack(1, ModelFrameType);

    DrivingModelFrameType = (PyTypeObject*)PyType_FromSpecWithBases(&DrivingModelFrame_spec, bases);
    if (PyModule_AddObject(m, "DrivingModelFrame", (PyObject *)DrivingModelFrameType) < 0) {
        Py_DECREF(bases);
        return NULL;
    }
    Py_INCREF(DrivingModelFrameType);

    MonitoringModelFrameType = (PyTypeObject*)PyType_FromSpecWithBases(&MonitoringModelFrame_spec, bases);
    if (PyModule_AddObject(m, "MonitoringModelFrame", (PyObject *)MonitoringModelFrameType) < 0) {
        Py_DECREF(bases);
        return NULL;
    }
    Py_INCREF(MonitoringModelFrameType);

    Py_DECREF(bases);

    return m;
}
