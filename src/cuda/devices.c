/* ============================================================================
 * cuda/devices.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * device count and property enumeration. pure Runtime API (cudaGetDevice
 * Count/cudaGetDeviceProperties) -- no driver API calls needed here, and
 * critically, no cudaSetDevice()/cuda_activate_context() either: device
 * enumeration is not scoped to "the current device" the way most other
 * CUDA operations are, so this file is exempt from the activation
 * discipline the rest of the CUDA subtree follows.
 * ========================================================================= */

#include <cuda_runtime.h>
#include <Rinternals.h>
#include "ardea.h"

int cuda_device_count(void) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    /* a real driver/toolkit problem (vs. simply "zero devices", which
     * cudaGetDeviceCount() reports as count == 0 with cudaSuccess, not as
     * an error) -- treated as zero devices rather than propagating the
     * error, matching metal_is_available()'s pattern of a clean FALSE
     * rather than a hard stop for an unavailable framework */
    return 0;
  }
  return count;
}

SEXP cuda_exposed_device_count(void) {
  return Rf_ScalarInteger(cuda_device_count());
}

/* ----------------------------------------------------------------------------
 * one named-list entry per device. field set intentionally parallels
 * metal_available_devices()'s shape (name/index-style fields first,
 * memory limits, then hardware-specific detail) rather than matching it
 * field-for-field -- CUDA's cudaDeviceProp exposes different things than
 * MTLDevice does, so forcing identical field NAMES across frameworks
 * would misrepresent what's actually queryable.
 *
 * no "is_default" field: unlike MTLCreateSystemDefaultDevice(), CUDA has
 * no inherent default-device concept -- cudaGetDevice() only reports
 * whatever happens to be current at the moment of the call, which is
 * arbitrary at device-enumeration time, not a hardware property worth
 * reporting.
 * ------------------------------------------------------------------------- */
SEXP cuda_available_devices(void) {
  int count = cuda_device_count();
  
  SEXP result = PROTECT(Rf_allocVector(VECSXP, count));
  
  for (int i = 0; i < count; i++) {
    struct cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, i);
    if (err != cudaSuccess) {
      UNPROTECT(1);
      Rf_error("failed to query properties for CUDA device %d: %s",
               i, cudaGetErrorString(err));
    }
    
    const char *names[] = {
      "index", "device_index", "name",
      "total_global_mem_bytes", "multiprocessor_count",
      "max_threads_per_block", "warp_size",
      "compute_capability_major", "compute_capability_minor"
    };
    int n_fields = 9;
    
    SEXP device_info = PROTECT(Rf_allocVector(VECSXP, n_fields));
    SEXP device_names = PROTECT(Rf_allocVector(STRSXP, n_fields));
    for (int f = 0; f < n_fields; f++) {
      SET_STRING_ELT(device_names, f, Rf_mkChar(names[f]));
    }
    
    SET_VECTOR_ELT(device_info, 0, Rf_ScalarInteger(i));
    SET_VECTOR_ELT(device_info, 1, Rf_ScalarInteger(i));
    SET_VECTOR_ELT(device_info, 2, Rf_mkString(prop.name));
    SET_VECTOR_ELT(device_info, 3, Rf_ScalarReal((double)prop.totalGlobalMem));
    SET_VECTOR_ELT(device_info, 4, Rf_ScalarInteger(prop.multiProcessorCount));
    SET_VECTOR_ELT(device_info, 5, Rf_ScalarInteger(prop.maxThreadsPerBlock));
    SET_VECTOR_ELT(device_info, 6, Rf_ScalarInteger(prop.warpSize));
    SET_VECTOR_ELT(device_info, 7, Rf_ScalarInteger(prop.major));
    SET_VECTOR_ELT(device_info, 8, Rf_ScalarInteger(prop.minor));
    
    Rf_setAttrib(device_info, R_NamesSymbol, device_names);
    SET_VECTOR_ELT(result, i, device_info);
    UNPROTECT(2); /* device_info, device_names -- now referenced by result */
  }
  
  UNPROTECT(1);
  return result;
}
