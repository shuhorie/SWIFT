/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2019 Loic Hausammann (loic.hausammann@epfl.ch)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#include "logger_header.h"
#include "logger_loader_io.h"
#include "logger_particle.h"
#include "logger_reader.h"
#include "logger_time.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <errno.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  PyObject_HEAD struct logger_particle part;
} PyLoggerParticle;

static PyTypeObject PyLoggerParticle_Type;

PyArray_Descr *logger_particle_descr;

/**
 * @brief load data from the index files.
 *
 * <b>basename</b> Base name of the logger files.
 *
 * <b>time</b> The time requested.
 *
 * <b>verbose</b> Verbose level.
 *
 * <b>returns</b> dictionnary containing the data read.
 */
static PyObject *loadSnapshotAtTime(__attribute__((unused)) PyObject *self,
                                    PyObject *args) {

  /* declare variables. */
  char *basename = NULL;

  double time = 0;
  int verbose = 2;

  /* parse arguments. */
  if (!PyArg_ParseTuple(args, "sd|i", &basename, &time, &verbose)) return NULL;

  /* initialize the reader. */
  struct logger_reader reader;
  logger_reader_init(&reader, basename, verbose);

  if (verbose > 1) message("Reading particles.");

  /* Number of particles in the index files */
  npy_intp n_tot = 0;

  /* Set the reading time */
  logger_reader_set_time(&reader, time);

  /* Get the number of particles */
  int n_type = 0;
  const uint64_t *n_parts =
      logger_reader_get_number_particles(&reader, &n_type);
  for (int i = 0; i < n_type; i++) {
    n_tot += n_parts[i];
  }

#ifdef SWIFT_DEBUG_CHECKS
  message("Found %lu particles", n_tot);
#endif  // SWIFT_DEBUG_CHECKS

  /* Allocate the output memory */
  PyArrayObject *out = (PyArrayObject *)PyArray_SimpleNewFromDescr(
      1, &n_tot, logger_particle_descr);

  /* Allows to use threads */
  Py_BEGIN_ALLOW_THREADS;

  /* Read the particle. */
  logger_reader_read_all_particles(&reader, time, logger_reader_const,
                                   PyArray_DATA(out), n_tot);

  /* No need of threads anymore */
  Py_END_ALLOW_THREADS;

  /* Free the memory. */
  logger_reader_free(&reader);

  return (PyObject *)out;
}

/**
 * @brief Read the minimal and maximal time.
 *
 * <b>basename</b> Base name of the logger files.
 *
 * <b>verbose</b> Verbose level.
 *
 * <b>returns</b> tuple containing min and max time.
 */
static PyObject *getTimeLimits(__attribute__((unused)) PyObject *self,
                               PyObject *args) {

  /* declare variables. */
  char *basename = NULL;

  int verbose = 2;

  /* parse arguments. */
  if (!PyArg_ParseTuple(args, "s|i", &basename, &verbose)) return NULL;

  /* initialize the reader. */
  struct logger_reader reader;
  logger_reader_init(&reader, basename, verbose);

  if (verbose > 1) message("Reading particles.");

  /* Get the time limits */
  double time_min = logger_reader_get_time_begin(&reader);
  double time_max = logger_reader_get_time_end(&reader);

  /* Free the memory. */
  logger_reader_free(&reader);

  /* Create the output */
  PyObject *out = PyTuple_New(2);
  PyTuple_SetItem(out, 0, PyFloat_FromDouble(time_min));
  PyTuple_SetItem(out, 1, PyFloat_FromDouble(time_max));

  return (PyObject *)out;
}

/**
 * @brief Reverse offset in log file
 *
 * <b>filename</b> string filename of the log file
 * <b>verbose</b> Verbose level
 */
static PyObject *pyReverseOffset(__attribute__((unused)) PyObject *self,
                                 PyObject *args) {
  /* input variables. */
  char *filename = NULL;

  int verbose = 0;

  /* parse the arguments. */
  if (!PyArg_ParseTuple(args, "s|i", &filename, &verbose)) return NULL;

  /* initialize the reader which reverse the offset if necessary. */
  struct logger_reader reader;
  logger_reader_init(&reader, filename, verbose);

  /* Free the reader. */
  logger_reader_free(&reader);

  return Py_BuildValue("");
}

/* definition of the method table. */

static PyMethodDef libloggerMethods[] = {
    {"loadSnapshotAtTime", loadSnapshotAtTime, METH_VARARGS,
     "Load a snapshot directly from the logger using the index files.\n\n"
     "Parameters\n"
     "----------\n\n"
     "basename: str\n"
     "  The basename of the index files.\n\n"
     "time: double\n"
     "  The (double) time of the snapshot.\n\n"
     "verbose: int, optional\n"
     "  The verbose level of the loader.\n\n"
     "Returns\n"
     "-------\n\n"
     "snapshot: dict\n"
     "  The full output generated for the whole file.\n"},
    {"reverseOffset", pyReverseOffset, METH_VARARGS,
     "Reverse the offset (from pointing backward to forward).\n\n"
     "Parameters\n"
     "----------\n\n"
     "filename: str\n"
     "  The filename of the log file.\n\n"
     "verbose: int, optional\n"
     "  The verbose level of the loader.\n"},
    {"getTimeLimits", getTimeLimits, METH_VARARGS,
     "Read the time limits of the simulation.\n\n"
     "Parameters\n"
     "----------\n\n"
     "basename: str\n"
     "  The basename of the index files.\n\n"
     "verbose: int, optional\n"
     "  The verbose level of the loader.\n\n"
     "Returns\n"
     "-------\n\n"
     "times: tuple\n"
     "  time min, time max\n"},

    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef libloggermodule = {
    PyModuleDef_HEAD_INIT,
    "liblogger",
    "Module reading a SWIFTsim logger snapshot",
    -1,
    libloggerMethods,
    NULL, /* m_slots */
    NULL, /* m_traverse */
    NULL, /* m_clear */
    NULL  /* m_free */
};

#define CREATE_FIELD(fields, name, field_name, type)                      \
  ({                                                                      \
    PyObject *tuple = PyTuple_New(2);                                     \
    PyTuple_SetItem(tuple, 0, (PyObject *)PyArray_DescrFromType(type));   \
    PyTuple_SetItem(                                                      \
        tuple, 1,                                                         \
        PyLong_FromSize_t(offsetof(struct logger_particle, field_name))); \
    PyDict_SetItem(fields, PyUnicode_FromString(name), tuple);            \
  })

#define CREATE_FIELD_3D(fields, name, field_name, type)                   \
  ({                                                                      \
    /* Create the 3D descriptor */                                        \
    PyArray_Descr *vec = PyArray_DescrNewFromType(type);                  \
    vec->subarray = malloc(sizeof(PyArray_ArrayDescr));                   \
    vec->subarray->base = PyArray_DescrFromType(type);                    \
    vec->subarray->shape = PyTuple_New(1);                                \
    PyTuple_SetItem(vec->subarray->shape, 0, PyLong_FromSize_t(3));       \
                                                                          \
    /* Create the field */                                                \
    PyObject *tuple = PyTuple_New(2);                                     \
    PyTuple_SetItem(tuple, 0, (PyObject *)vec);                           \
    PyTuple_SetItem(                                                      \
        tuple, 1,                                                         \
        PyLong_FromSize_t(offsetof(struct logger_particle, field_name))); \
    PyDict_SetItem(fields, PyUnicode_FromString(name), tuple);            \
  })

void pylogger_particle_define_descr(void) {
  /* Generate list of field names */
  PyObject *names = PyTuple_New(9);
  PyTuple_SetItem(names, 0, PyUnicode_FromString("positions"));
  PyTuple_SetItem(names, 1, PyUnicode_FromString("velocities"));
  PyTuple_SetItem(names, 2, PyUnicode_FromString("accelerations"));
  PyTuple_SetItem(names, 3, PyUnicode_FromString("entropies"));
  PyTuple_SetItem(names, 4, PyUnicode_FromString("smoothing_lengths"));
  PyTuple_SetItem(names, 5, PyUnicode_FromString("densities"));
  PyTuple_SetItem(names, 6, PyUnicode_FromString("masses"));
  PyTuple_SetItem(names, 7, PyUnicode_FromString("ids"));
  PyTuple_SetItem(names, 8, PyUnicode_FromString("times"));

  /* Generate list of fields */
  PyObject *fields = PyDict_New();
  CREATE_FIELD_3D(fields, "positions", pos, NPY_DOUBLE);
  CREATE_FIELD(fields, "velocities", vel, NPY_FLOAT32);
  CREATE_FIELD(fields, "accelerations", acc, NPY_FLOAT32);
  CREATE_FIELD(fields, "entropies", entropy, NPY_FLOAT32);
  CREATE_FIELD(fields, "smoothing_lenghts", h, NPY_FLOAT32);
  CREATE_FIELD(fields, "densities", density, NPY_FLOAT32);
  CREATE_FIELD(fields, "masses", mass, NPY_FLOAT32);
  CREATE_FIELD(fields, "ids", id, NPY_ULONGLONG);
  CREATE_FIELD(fields, "times", id, NPY_DOUBLE);

  /* Generate descriptor */
  logger_particle_descr = PyObject_New(PyArray_Descr, &PyArrayDescr_Type);
  logger_particle_descr->typeobj = &PyLoggerParticle_Type;
  // V if for an arbitrary kind of array
  logger_particle_descr->kind = 'V';
  // Not well documented (seems any value is fine)
  logger_particle_descr->type = 'p';
  // Native byte ordering
  logger_particle_descr->byteorder = '=';
  // Flags
  logger_particle_descr->flags = NPY_USE_GETITEM | NPY_USE_SETITEM;
  // id of the data type (assigned automatically)
  logger_particle_descr->type_num = 0;
  // Size of an element
  logger_particle_descr->elsize = sizeof(struct logger_particle);
  // alignment (doc magic)
  logger_particle_descr->alignment = offsetof(
      struct {
        char c;
        struct logger_particle v;
      },
      v);
  // no subarray
  logger_particle_descr->subarray = NULL;
  // functions
  logger_particle_descr->f = NULL;
  // Meta data
  logger_particle_descr->metadata = NULL;
  logger_particle_descr->c_metadata = NULL;
  logger_particle_descr->names = names;
  logger_particle_descr->fields = fields;
}

PyMODINIT_FUNC PyInit_liblogger(void) {
  PyObject *m;
  m = PyModule_Create(&libloggermodule);
  if (m == NULL) return NULL;

  /* Deal with SWIFT clock */
  clocks_set_cpufreq(0);

  import_array();

  /* Define the descr of the logger_particle */
  pylogger_particle_define_descr();

  return m;
}
