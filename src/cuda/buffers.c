/* ============================================================================
 * cuda/buffers.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * R vector <-> host staging buffer conversion. structurally identical to
 * metal/buffers.c -- same NA/NaN guarding (an integer-typed R argument
 * carrying NA_INTEGER, or a REALSXP argument declared as an integer
 * CudaType carrying NA_REAL, is caught here as a second, independent
 * layer; the primary guard is the front-loaded scan in cuda/runners.c,
 * which exists specifically so nothing has been allocated yet by the time
 * either guard could fire -- see that file for why the *order* matters,
 * not just the presence of the check), same explicit default: Rf_error()
 * on every switch.
 *
 * the one real difference from metal/buffers.c: this operates on plain
 * HOST memory (a malloc'd staging buffer), not a Metal Shared-storage
 * buffer directly. CUDA's classic allocation model needs an explicit
 * host-side intermediate before the explicit cudaMemcpyHostToDevice copy
 * -- see cuda/runners.c for where that copy happens; this file only
 * fills the host side of it.
 * ========================================================================= */

#include <Rinternals.h>
#include <R.h>
#include <stdlib.h>
#include <string.h>
#include "ardea.h"

void cuda_convert_r_numeric_to_host(const double *r_data,
                                    void *host_buffer,
                                    size_t length,
                                    CudaType type) {
  switch (type) {
  case CUDA_TYPE_FLOAT: {
    float *buf = (float *)host_buffer;
    for (size_t i = 0; i < length; i++) {
      if (ISNA(r_data[i]) || ISNAN(r_data[i])) {
        Rf_error("NA/NaN present in numeric argument at position %zu -- "
                 "CUDA has no representation for R's NA", i + 1);
      }
      buf[i] = (float)r_data[i];
    }
    break;
  }
  case CUDA_TYPE_DOUBLE: {
    for (size_t i = 0; i < length; i++) {
      if (ISNA(r_data[i])) {
        Rf_error("NA present in numeric argument at position %zu -- "
                 "CUDA has no representation for R's NA", i + 1);
      }
    }
    memcpy(host_buffer, r_data, length * sizeof(double));
    break;
  }
  case CUDA_TYPE_INT8:
  case CUDA_TYPE_UINT8:
  case CUDA_TYPE_INT16:
  case CUDA_TYPE_UINT16:
  case CUDA_TYPE_INT:
  case CUDA_TYPE_UINT:
  case CUDA_TYPE_INT64:
  case CUDA_TYPE_UINT64: {
    for (size_t i = 0; i < length; i++) {
      if (ISNA(r_data[i]) || ISNAN(r_data[i])) {
        Rf_error("NA/NaN present in numeric argument at position %zu, "
                 "declared as an integer type -- casting NaN to an "
                 "integer type is undefined behavior", i + 1);
      }
    }
    switch (type) {
    case CUDA_TYPE_INT8: {
      int8_t *buf = (int8_t *)host_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (int8_t)r_data[i];
      break;
    }
    case CUDA_TYPE_UINT8: {
      uint8_t *buf = (uint8_t *)host_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (uint8_t)r_data[i];
      break;
    }
    case CUDA_TYPE_INT16: {
      int16_t *buf = (int16_t *)host_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (int16_t)r_data[i];
      break;
    }
    case CUDA_TYPE_UINT16: {
      uint16_t *buf = (uint16_t *)host_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (uint16_t)r_data[i];
      break;
    }
    case CUDA_TYPE_INT: {
      int32_t *buf = (int32_t *)host_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (int32_t)r_data[i];
      break;
    }
    case CUDA_TYPE_UINT: {
      uint32_t *buf = (uint32_t *)host_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (uint32_t)r_data[i];
      break;
    }
    case CUDA_TYPE_INT64: {
      int64_t *buf = (int64_t *)host_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (int64_t)r_data[i];
      break;
    }
    case CUDA_TYPE_UINT64: {
      uint64_t *buf = (uint64_t *)host_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (uint64_t)r_data[i];
      break;
    }
    default:
      Rf_error("internal error: unreachable integer type in "
               "cuda_convert_r_numeric_to_host()");
    }
    break;
  }
  default:
    Rf_error("cuda_convert_r_numeric_to_host(): unrecognized CudaType "
             "value (%d)", (int)type);
  }
}

void cuda_convert_r_int_to_host(const int *r_data,
                                void *host_buffer,
                                size_t length,
                                CudaType type) {
  for (size_t i = 0; i < length; i++) {
    if (r_data[i] == NA_INTEGER) {
      Rf_error("NA present in integer argument at position %zu -- "
               "CUDA has no representation for R's NA", i + 1);
    }
  }
  
  double *temp = malloc(length * sizeof(double));
  if (temp == NULL) {
    Rf_error("failed to allocate %zu bytes for integer conversion",
             length * sizeof(double));
  }
  for (size_t i = 0; i < length; i++) {
    temp[i] = (double)r_data[i];
  }
  
  cuda_convert_r_numeric_to_host(temp, host_buffer, length, type);
  free(temp);
}

void cuda_convert_host_to_r(const void *host_buffer,
                            double *r_data,
                            size_t length,
                            CudaType type) {
  switch (type) {
  case CUDA_TYPE_FLOAT: {
    const float *buf = (const float *)host_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case CUDA_TYPE_DOUBLE: {
    memcpy(r_data, host_buffer, length * sizeof(double));
    break;
  }
  case CUDA_TYPE_INT8: {
    const int8_t *buf = (const int8_t *)host_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case CUDA_TYPE_UINT8: {
    const uint8_t *buf = (const uint8_t *)host_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case CUDA_TYPE_INT16: {
    const int16_t *buf = (const int16_t *)host_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case CUDA_TYPE_UINT16: {
    const uint16_t *buf = (const uint16_t *)host_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case CUDA_TYPE_INT: {
    const int32_t *buf = (const int32_t *)host_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case CUDA_TYPE_UINT: {
    const uint32_t *buf = (const uint32_t *)host_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case CUDA_TYPE_INT64: {
    const int64_t *buf = (const int64_t *)host_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case CUDA_TYPE_UINT64: {
    const uint64_t *buf = (const uint64_t *)host_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  default:
    Rf_error("cuda_convert_host_to_r(): unrecognized CudaType value (%d)",
             (int)type);
  }
}
