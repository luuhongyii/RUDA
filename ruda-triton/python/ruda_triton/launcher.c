// Generic launcher for Ruda-compiled Triton kernels.
//
// launch(cu_launch_kernel, function, gx, gy, gz, threads, shared, stream, spec, offsets, info_size, args)
//
// `spec` has one code per Triton signature entry: 'c' constexpr (skipped), 'p' pointer, or a
// struct-module code for a scalar ('?', 'b', 'h', 'i', 'q', 'B', 'H', 'I', 'Q', 'e', 'E' (bf16),
// 'f', 'd'). Scalar `i` is written at byte offset offsets[i] (little-endian int32 array) of the
// kernel's `info` parameter, which is passed after one u64 parameter per pointer.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

typedef int (*LaunchKernel)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                            void **, void **);

#define MAX_POINTERS 128
#define INLINE_INFO 1024

static int pointer_of(PyObject *obj, uint64_t *out) {
  if (obj == Py_None) {
    *out = 0;
    return 0;
  }
  if (PyLong_Check(obj)) {
    *out = PyLong_AsUnsignedLongLong(obj);
    return PyErr_Occurred() ? -1 : 0;
  }
  PyObject *ptr = PyObject_CallMethod(obj, "data_ptr", NULL);
  if (!ptr)
    return -1;
  *out = PyLong_AsUnsignedLongLong(ptr);
  Py_DECREF(ptr);
  return PyErr_Occurred() ? -1 : 0;
}

static int store_scalar(char code, PyObject *obj, char *dst) {
  switch (code) {
  case '?': {
    int v = PyObject_IsTrue(obj);
    if (v < 0)
      return -1;
    *(uint8_t *)dst = (uint8_t)v;
    return 0;
  }
  case 'b': case 'B': case 'h': case 'H': case 'i': case 'I': case 'q': case 'Q': {
    long long v;
    if (PyBool_Check(obj) || PyLong_Check(obj)) {
      v = PyLong_AsLongLong(obj);
      if (v == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        v = (long long)PyLong_AsUnsignedLongLong(obj);
      }
    } else {
      PyObject *as_int = PyNumber_Long(obj);
      if (!as_int)
        return -1;
      v = PyLong_AsLongLong(as_int);
      Py_DECREF(as_int);
    }
    if (PyErr_Occurred())
      return -1;
    switch (code) {
    case 'b': case 'B': *(uint8_t *)dst = (uint8_t)v; break;
    case 'h': case 'H': { uint16_t x = (uint16_t)v; memcpy(dst, &x, 2); break; }
    case 'i': case 'I': { uint32_t x = (uint32_t)v; memcpy(dst, &x, 4); break; }
    default: memcpy(dst, &v, 8); break;
    }
    return 0;
  }
  case 'e': case 'E': case 'f': case 'd': {
    double v = PyFloat_AsDouble(obj);
    if (v == -1.0 && PyErr_Occurred())
      return -1;
    if (code == 'd') {
      memcpy(dst, &v, 8);
    } else if (code == 'f') {
      float f = (float)v;
      memcpy(dst, &f, 4);
    } else if (code == 'e') {
      return PyFloat_Pack2(v, dst, 1);
    } else {
      float f = (float)v;
      uint32_t bits;
      memcpy(&bits, &f, 4);
      uint16_t h = (bits & 0x7fffffff) > 0x7f800000 ? (uint16_t)((bits >> 16) | 0x40)
                                                     : (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
      memcpy(dst, &h, 2);
    }
    return 0;
  }
  }
  PyErr_Format(PyExc_TypeError, "unknown scalar code '%c'", code);
  return -1;
}

static PyObject *launch(PyObject *self, PyObject *args) {
  unsigned long long launch_fn, function, stream;
  unsigned int gx, gy, gz, threads, shared;
  const char *spec, *offsets;
  Py_ssize_t spec_len, offsets_len, info_size;
  PyObject *kernel_args;
  if (!PyArg_ParseTuple(args, "KKIIIIIKy#y#nO!", &launch_fn, &function, &gx, &gy, &gz, &threads, &shared, &stream,
                        &spec, &spec_len, &offsets, &offsets_len, &info_size, &PyTuple_Type, &kernel_args))
    return NULL;
  if (PyTuple_GET_SIZE(kernel_args) < spec_len) {
    PyErr_Format(PyExc_TypeError, "kernel expects %zd arguments, got %zd", spec_len, PyTuple_GET_SIZE(kernel_args));
    return NULL;
  }
  if ((Py_ssize_t)gx * gy * gz == 0)
    Py_RETURN_NONE;

  uint64_t pointers[MAX_POINTERS];
  void *params[MAX_POINTERS + 1];
  char inline_info[INLINE_INFO];
  char *info = info_size > INLINE_INFO ? (char *)PyMem_Calloc(1, info_size) : inline_info;
  if (!info)
    return PyErr_NoMemory();
  if (info == inline_info)
    memset(inline_info, 0, info_size > 0 ? (size_t)info_size : 0);
  int n_ptr = 0, n_scalar = 0;
  for (Py_ssize_t i = 0; i < spec_len; ++i) {
    char code = spec[i];
    PyObject *arg = PyTuple_GET_ITEM(kernel_args, i);
    if (code == 'c')
      continue;
    if (code == 'p') {
      if (n_ptr == MAX_POINTERS) {
        PyErr_SetString(PyExc_ValueError, "too many pointer arguments");
        goto fail;
      }
      if (pointer_of(arg, &pointers[n_ptr]) < 0)
        goto fail;
      params[n_ptr] = &pointers[n_ptr];
      ++n_ptr;
      continue;
    }
    if ((Py_ssize_t)(n_scalar + 1) * 4 > offsets_len) {
      PyErr_SetString(PyExc_ValueError, "scalar offset table too short");
      goto fail;
    }
    int32_t offset;
    memcpy(&offset, offsets + n_scalar * 4, 4);
    if (store_scalar(code, arg, info + offset) < 0)
      goto fail;
    ++n_scalar;
  }
  int n_params = n_ptr;
  if (info_size > 0)
    params[n_params++] = info;
  int result;
  Py_BEGIN_ALLOW_THREADS;
  result = ((LaunchKernel)(uintptr_t)launch_fn)((void *)(uintptr_t)function, gx, gy, gz, threads, 1, 1, shared,
                                                (void *)(uintptr_t)stream, params, NULL);
  Py_END_ALLOW_THREADS;
  if (info != inline_info)
    PyMem_Free(info);
  if (result != 0) {
    PyErr_Format(PyExc_RuntimeError, "cuLaunchKernel failed with CUDA error %d", result);
    return NULL;
  }
  Py_RETURN_NONE;
fail:
  if (info != inline_info)
    PyMem_Free(info);
  return NULL;
}

static PyMethodDef methods[] = {{"launch", launch, METH_VARARGS, "Launch a Ruda-compiled Triton kernel."},
                                {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "ruda_launcher", NULL, -1, methods};

PyMODINIT_FUNC PyInit_ruda_launcher(void) { return PyModule_Create(&module); }
