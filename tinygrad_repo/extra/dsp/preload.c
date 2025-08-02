__attribute__((constructor))
void preload_init() {
  Py_Initialize();
  PyRun_SimpleString("print('hello from c'); import extra.dsp.hook");
}

#define _GNU_SOURCE  // Must be defined before any includes for RTLD_NEXT
#include <stdio.h>
#include <dlfcn.h>
#include <Python.h>  // Include Python header
//#include <sys/ioctl.h>

// Define the original ioctl function pointer
static int (*real_ioctl)(int fd, unsigned long request, void *arg) = NULL;

// Our custom ioctl hook
int ioctl(int fd, unsigned long request, void *arg) {
	// Initialize the real ioctl function pointer on first call
	if (!real_ioctl) {
		real_ioctl = dlsym(RTLD_NEXT, "ioctl");
		if (!real_ioctl) {
			fprintf(stderr, "Error: Could not find real ioctl\n");
			return -1;
		}
	}

	// Log the call
	//printf("Hooked ioctl: tid=%d fd=%d, request=0x%lx, arg=%p\n", gettid(), fd, request, arg);

	// Call a Python function from extra.dsp.hook
	PyObject *pName, *pModule, *pFunc, *pArgs, *pValue;
	PyGILState_STATE gstate;

	// Ensure the GIL is held (required for Python calls in multi-threaded apps)
	//gstate = PyGILState_Ensure();

	// Import the module
	pName = PyUnicode_FromString("extra.dsp.hook");
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	// Call the original ioctl
	int ret = real_ioctl(fd, request, arg);

	if (pModule != NULL) {
		// Get the function (assume it’s called "handle_ioctl")
		pFunc = PyObject_GetAttrString(pModule, "handle_ioctl");

		if (pFunc && PyCallable_Check(pFunc)) {
			// Create arguments tuple (fd, request, arg, ret)
			pArgs = PyTuple_Pack(4,
													 PyLong_FromLong(fd),
													 PyLong_FromUnsignedLong(request),
													 PyLong_FromVoidPtr(arg),
													 PyLong_FromLong(ret));
			pValue = PyObject_CallObject(pFunc, pArgs);
			Py_DECREF(pArgs);

			if (pValue != NULL) {
				Py_DECREF(pValue);
			} else {
				PyErr_Print();  // Print Python error if call fails
			}
			Py_DECREF(pFunc);
		} else {
			if (PyErr_Occurred()) PyErr_Print();
			fprintf(stderr, "Cannot find function 'handle_ioctl'\n");
		}
		Py_DECREF(pModule);
	} else {
			PyErr_Print();
		fprintf(stderr, "Failed to load 'extra.dsp.hook'\n");
	}

	// Release the GIL
	//PyGILState_Release(gstate);
	return ret;
}

