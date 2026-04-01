#define PY_SSIZE_T_CLEAN
#include "cutechronos_bridge.h"

#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef CUTECHRONOS_DEFAULT_PROGRAM
#define CUTECHRONOS_DEFAULT_PROGRAM ""
#endif

#ifndef CUTECHRONOS_DEFAULT_PYTHONHOME
#define CUTECHRONOS_DEFAULT_PYTHONHOME ""
#endif

#ifndef CUTECHRONOS_DEFAULT_PYTHONPATH
#define CUTECHRONOS_DEFAULT_PYTHONPATH ""
#endif

static int g_python_ready = 0;
static int g_embedded_python = 0;
static PyObject *g_module = NULL;
static PyObject *g_init_func = NULL;
static PyObject *g_predict_func = NULL;
static PyObject *g_destroy_func = NULL;

static void set_error(char *buffer, size_t buffer_size, const char *message) {
    if (buffer == NULL || buffer_size == 0) {
        return;
    }
    snprintf(buffer, buffer_size, "%s", message != NULL ? message : "unknown error");
}

static void set_python_exception(char *buffer, size_t buffer_size) {
    PyObject *ptype = NULL;
    PyObject *pvalue = NULL;
    PyObject *ptraceback = NULL;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

    if (pvalue != NULL) {
        PyObject *text = PyObject_Str(pvalue);
        if (text != NULL) {
            const char *message = PyUnicode_AsUTF8(text);
            set_error(buffer, buffer_size, message);
            Py_DECREF(text);
        } else {
            set_error(buffer, buffer_size, "python exception");
        }
    } else {
        set_error(buffer, buffer_size, "python exception");
    }

    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
}

static int add_python_path_entries(const char *paths, char *error_buffer, size_t error_buffer_size) {
    if (paths == NULL || paths[0] == '\0') {
        return 0;
    }

    PyObject *sys_path = PySys_GetObject("path");
    if (sys_path == NULL) {
        set_error(error_buffer, error_buffer_size, "failed to access sys.path");
        return -1;
    }

    char *copy = strdup(paths);
    if (copy == NULL) {
        set_error(error_buffer, error_buffer_size, "out of memory");
        return -1;
    }

    char *saveptr = NULL;
    char *token = strtok_r(copy, ":", &saveptr);
    while (token != NULL) {
        PyObject *item = PyUnicode_FromString(token);
        if (item == NULL || PyList_Insert(sys_path, 0, item) != 0) {
            Py_XDECREF(item);
            free(copy);
            set_python_exception(error_buffer, error_buffer_size);
            return -1;
        }
        Py_DECREF(item);
        token = strtok_r(NULL, ":", &saveptr);
    }

    free(copy);
    return 0;
}

static int ensure_python(char *error_buffer, size_t error_buffer_size) {
    if (g_python_ready) {
        return 0;
    }

    const char *program = getenv("CUTECHRONOS_PYTHON_PROGRAM");
    const char *pythonhome = getenv("CUTECHRONOS_PYTHONHOME");
    const char *pythonpath = getenv("CUTECHRONOS_PYTHONPATH");

    if (program == NULL || program[0] == '\0') {
        program = CUTECHRONOS_DEFAULT_PROGRAM;
    }
    if (pythonhome == NULL || pythonhome[0] == '\0') {
        pythonhome = CUTECHRONOS_DEFAULT_PYTHONHOME;
    }
    if (pythonpath == NULL || pythonpath[0] == '\0') {
        pythonpath = CUTECHRONOS_DEFAULT_PYTHONPATH;
    }

    if (Py_IsInitialized()) {
        PyGILState_STATE existing_gstate = PyGILState_Ensure();
        if (add_python_path_entries(pythonpath, error_buffer, error_buffer_size) != 0) {
            PyGILState_Release(existing_gstate);
            return -1;
        }

        g_module = PyImport_ImportModule("cutechronos.foreign");
        if (g_module == NULL) {
            set_python_exception(error_buffer, error_buffer_size);
            PyGILState_Release(existing_gstate);
            return -1;
        }

        g_init_func = PyObject_GetAttrString(g_module, "init_pipeline");
        g_predict_func = PyObject_GetAttrString(g_module, "predict_median");
        g_destroy_func = PyObject_GetAttrString(g_module, "destroy_pipeline");
        if (g_init_func == NULL || g_predict_func == NULL || g_destroy_func == NULL) {
            set_python_exception(error_buffer, error_buffer_size);
            PyGILState_Release(existing_gstate);
            return -1;
        }

        g_python_ready = 1;
        PyGILState_Release(existing_gstate);
        return 0;
    }

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;

    if (program[0] != '\0') {
        status = PyConfig_SetBytesString(&config, &config.program_name, program);
        if (PyStatus_Exception(status)) {
            set_error(error_buffer, error_buffer_size, status.err_msg);
            PyConfig_Clear(&config);
            return -1;
        }
    }
    if (pythonhome[0] != '\0') {
        status = PyConfig_SetBytesString(&config, &config.home, pythonhome);
        if (PyStatus_Exception(status)) {
            set_error(error_buffer, error_buffer_size, status.err_msg);
            PyConfig_Clear(&config);
            return -1;
        }
    }

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        set_error(error_buffer, error_buffer_size, status.err_msg);
        return -1;
    }

    if (add_python_path_entries(pythonpath, error_buffer, error_buffer_size) != 0) {
        return -1;
    }

    g_module = PyImport_ImportModule("cutechronos.foreign");
    if (g_module == NULL) {
        set_python_exception(error_buffer, error_buffer_size);
        return -1;
    }

    g_init_func = PyObject_GetAttrString(g_module, "init_pipeline");
    g_predict_func = PyObject_GetAttrString(g_module, "predict_median");
    g_destroy_func = PyObject_GetAttrString(g_module, "destroy_pipeline");
    if (g_init_func == NULL || g_predict_func == NULL || g_destroy_func == NULL) {
        set_python_exception(error_buffer, error_buffer_size);
        return -1;
    }

    g_python_ready = 1;
    g_embedded_python = 1;
    PyEval_SaveThread();
    return 0;
}

int cutechronos_init_pipeline(
    const char *model_id,
    const char *backend,
    const char *device,
    const char *dtype_name,
    const char *compile_mode,
    int *out_handle,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (out_handle == NULL) {
        set_error(error_buffer, error_buffer_size, "out_handle is required");
        return -1;
    }
    if (ensure_python(error_buffer, error_buffer_size) != 0) {
        return -1;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *kwargs = PyDict_New();
    if (kwargs == NULL) {
        set_python_exception(error_buffer, error_buffer_size);
        PyGILState_Release(gstate);
        return -1;
    }

    PyObject *result = NULL;
    PyObject *empty_args = PyTuple_New(0);
    int rc = -1;

    PyObject *py_model_id = PyUnicode_FromString(model_id);
    PyObject *py_backend = PyUnicode_FromString(backend);
    PyObject *py_device = PyUnicode_FromString(device);
    PyObject *py_dtype_name = PyUnicode_FromString(dtype_name);
    if (empty_args == NULL || py_model_id == NULL || py_backend == NULL ||
        py_device == NULL || py_dtype_name == NULL ||
        PyDict_SetItemString(kwargs, "model_id", py_model_id) != 0 ||
        PyDict_SetItemString(kwargs, "backend", py_backend) != 0 ||
        PyDict_SetItemString(kwargs, "device", py_device) != 0 ||
        PyDict_SetItemString(kwargs, "dtype_name", py_dtype_name) != 0) {
        set_python_exception(error_buffer, error_buffer_size);
        goto cleanup;
    }

    if (compile_mode != NULL && compile_mode[0] != '\0') {
        PyObject *py_compile_mode = PyUnicode_FromString(compile_mode);
        if (py_compile_mode == NULL || PyDict_SetItemString(kwargs, "compile_mode", py_compile_mode) != 0) {
            Py_XDECREF(py_compile_mode);
            set_python_exception(error_buffer, error_buffer_size);
            goto cleanup;
        }
        Py_DECREF(py_compile_mode);
    }

    result = PyObject_Call(g_init_func, empty_args, kwargs);
    if (result == NULL) {
        set_python_exception(error_buffer, error_buffer_size);
        goto cleanup;
    }

    *out_handle = (int)PyLong_AsLong(result);
    if (PyErr_Occurred()) {
        set_python_exception(error_buffer, error_buffer_size);
        goto cleanup;
    }

    rc = 0;

cleanup:
    Py_XDECREF(py_model_id);
    Py_XDECREF(py_backend);
    Py_XDECREF(py_device);
    Py_XDECREF(py_dtype_name);
    Py_XDECREF(empty_args);
    Py_XDECREF(result);
    Py_DECREF(kwargs);
    PyGILState_Release(gstate);
    return rc;
}

int cutechronos_predict_median(
    int handle,
    const float *context,
    int context_length,
    int prediction_length,
    float *out_values,
    int out_capacity,
    int *out_length,
    double *out_latency_ms,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (out_values == NULL || out_length == NULL || out_latency_ms == NULL) {
        set_error(error_buffer, error_buffer_size, "output buffers are required");
        return -1;
    }
    if (ensure_python(error_buffer, error_buffer_size) != 0) {
        return -1;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject *context_list = PyList_New((Py_ssize_t)context_length);
    if (context_list == NULL) {
        set_python_exception(error_buffer, error_buffer_size);
        PyGILState_Release(gstate);
        return -1;
    }

    for (int i = 0; i < context_length; ++i) {
        PyObject *value = PyFloat_FromDouble((double)context[i]);
        if (value == NULL) {
            Py_DECREF(context_list);
            set_python_exception(error_buffer, error_buffer_size);
            PyGILState_Release(gstate);
            return -1;
        }
        PyList_SET_ITEM(context_list, i, value);
    }

    PyObject *args = Py_BuildValue("(iOi)", handle, context_list, prediction_length);
    Py_DECREF(context_list);
    if (args == NULL) {
        set_python_exception(error_buffer, error_buffer_size);
        PyGILState_Release(gstate);
        return -1;
    }

    PyObject *result = PyObject_CallObject(g_predict_func, args);
    Py_DECREF(args);
    if (result == NULL) {
        set_python_exception(error_buffer, error_buffer_size);
        PyGILState_Release(gstate);
        return -1;
    }

    PyObject *forecast = NULL;
    PyObject *latency = NULL;
    if (!PyArg_ParseTuple(result, "OO", &forecast, &latency)) {
        Py_DECREF(result);
        set_python_exception(error_buffer, error_buffer_size);
        PyGILState_Release(gstate);
        return -1;
    }

    Py_ssize_t forecast_len = PySequence_Size(forecast);
    if (forecast_len < 0) {
        Py_DECREF(result);
        set_python_exception(error_buffer, error_buffer_size);
        PyGILState_Release(gstate);
        return -1;
    }
    if ((int)forecast_len > out_capacity) {
        Py_DECREF(result);
        set_error(error_buffer, error_buffer_size, "forecast buffer too small");
        PyGILState_Release(gstate);
        return -1;
    }

    for (Py_ssize_t i = 0; i < forecast_len; ++i) {
        PyObject *item = PySequence_GetItem(forecast, i);
        if (item == NULL) {
            Py_DECREF(result);
            set_python_exception(error_buffer, error_buffer_size);
            PyGILState_Release(gstate);
            return -1;
        }
        out_values[i] = (float)PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (PyErr_Occurred()) {
            Py_DECREF(result);
            set_python_exception(error_buffer, error_buffer_size);
            PyGILState_Release(gstate);
            return -1;
        }
    }

    *out_length = (int)forecast_len;
    *out_latency_ms = PyFloat_AsDouble(latency);
    Py_DECREF(result);

    if (PyErr_Occurred()) {
        set_python_exception(error_buffer, error_buffer_size);
        PyGILState_Release(gstate);
        return -1;
    }

    PyGILState_Release(gstate);
    return 0;
}

int cutechronos_destroy_pipeline(
    int handle,
    char *error_buffer,
    size_t error_buffer_size
) {
    if (ensure_python(error_buffer, error_buffer_size) != 0) {
        return -1;
    }

    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject *result = PyObject_CallFunction(g_destroy_func, "(i)", handle);
    if (result == NULL) {
        set_python_exception(error_buffer, error_buffer_size);
        PyGILState_Release(gstate);
        return -1;
    }
    Py_DECREF(result);
    PyGILState_Release(gstate);
    return 0;
}
