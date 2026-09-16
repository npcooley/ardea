/* ============================================================================
 * cuda/handles.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * CUDA handle construction, supporting:
 * cuda_make_context()
 * cuda_make_program()
 * cuda_make_kernelptr()
 * ========================================================================= */

#include <cuda_runtime.h>
#include <cuda.h>
#include <Rinternals.h>
#include <stdlib.h>
#include "ardea.h"

/* ============================================================================
 * context construction
 *
 * unlike metal_context_from_device()/opencl_context_from_device(), there
 * is no opaque device handle to validate against a tag here -- CUDA
 * identifies devices by plain integer ordinal. device_index is
 * re-validated against a LIVE cudaGetDeviceCount(), not trusted blindly:
 * it came from an 'alternative_device' object on the R side, but nothing
 * stops that object from being stale (built in an earlier session, a
 * device unplugged/reconfigured since).
 * ========================================================================= */
SEXP cuda_context_from_device(SEXP device_index_sexp,
                              SEXP use_default_stream_sexp) {
  if (TYPEOF(device_index_sexp) != INTSXP || LENGTH(device_index_sexp) != 1) {
    Rf_error("'device_index' must be a single integer value");
  }
  int device_index = INTEGER(device_index_sexp)[0];
  
  int device_count = 0;
  cudaError_t count_err = cudaGetDeviceCount(&device_count);
  if (count_err != cudaSuccess) {
    Rf_error("failed to query CUDA device count: %s",
             cudaGetErrorString(count_err));
  }
  if (device_index < 0 || device_index >= device_count) {
    Rf_error("'device_index' (%d) is out of range -- %d CUDA device(s) "
             "currently detected", device_index, device_count);
  }
  
  cudaError_t set_err = cudaSetDevice(device_index);
  if (set_err != cudaSuccess) {
    Rf_error("failed to set CUDA device %d as current: %s",
             device_index, cudaGetErrorString(set_err));
  }
  
  CudaContext *ctx = calloc(1, sizeof(*ctx));
  if (ctx == NULL) {
    Rf_error("failed to allocate CudaContext");
  }
  ctx->device_index = device_index;
  ctx->stream = NULL;
  
  if (TYPEOF(use_default_stream_sexp) != LGLSXP ||
      LENGTH(use_default_stream_sexp) != 1) {
    free(ctx);
    Rf_error("'use_default_stream' must be a single logical value");
  }
  
  if (LOGICAL(use_default_stream_sexp)[0] == FALSE) {
    cudaStream_t stream;
    cudaError_t stream_err = cudaStreamCreate(&stream);
    if (stream_err != cudaSuccess) {
      free(ctx);
      Rf_error("failed to create CUDA stream: %s",
               cudaGetErrorString(stream_err));
    }
    ctx->stream = (void *)stream;
  }
  
  SEXP context_ptr = PROTECT(R_MakeExternalPtr(ctx,
                                               cuda_context,
                                               R_NilValue));
  set_externalptr_class(context_ptr, "cuda_context");
  R_RegisterCFinalizerEx(context_ptr, cuda_context_finalizer, TRUE);
  UNPROTECT(1);
  return context_ptr;
}

/* ============================================================================
 * program (module) construction from a compiled .ptx file
 *
 * unlike Metal's dual-path metal_make_program() (runtime API compile OR
 * xcrun offline compile), CUDA support here is nvcc-only -- there is no
 * NVRTC (runtime-compilation) integration yet. cuda_make_program() (R
 * side) always shells out to nvcc first, then this function loads the
 * resulting .ptx file. cuModuleLoad() reads directly from a file path, so
 * there's no read-into-memory step the way Metal's source-string path
 * needed.
 * ========================================================================= */
SEXP cuda_program_from_ptx(SEXP ptx_file,
                           SEXP context_ptr) {
  CudaContext *ctx = get_checked_external_ptr(context_ptr,
                                              cuda_context,
                                              "cuda_context");
  cuda_activate_context(ctx);
  cuda_ensure_driver_init();
  
  if (TYPEOF(ptx_file) != STRSXP || LENGTH(ptx_file) != 1) {
    Rf_error("'ptx_file' must be a character vector of length 1");
  }
  
  CUmodule module;
  CUresult res = cuModuleLoad(&module, CHAR(STRING_ELT(ptx_file, 0)));
  if (res != CUDA_SUCCESS) {
    Rf_error("failed to load '%s': %s",
             CHAR(STRING_ELT(ptx_file, 0)),
             cuda_driver_error_string((int)res));
  }
  
  SEXP program_ptr = PROTECT(R_MakeExternalPtr((void *)module,
                                               cuda_module,
                                               R_NilValue));
  set_externalptr_class(program_ptr, "cuda_module");
  R_RegisterCFinalizerEx(program_ptr, cuda_module_finalizer, TRUE);
  UNPROTECT(1);
  return program_ptr;
}

/* ============================================================================
 * kernel extraction, with the lifetime anchor
 *
 * CUfunction has no reference counting -- unloading its parent CUmodule
 * (which happens when the program SEXP is garbage collected and
 * cuda_module_finalizer() fires) instantly invalidates every function
 * derived from it, with zero protection from the CUDA API itself. R's own
 * garbage collector is made to do this job instead: each kernel's
 * externalptr carries the parent program SEXP as an attribute, so R
 * cannot collect the module while any kernel built from it is still
 * reachable -- see the "ardea_parent_program" line below, and the
 * finalizer split (cuda_kernel_finalizer() frees only the small
 * CudaKernel struct; cuda_module_finalizer() is the one that actually
 * calls cuModuleUnload()).
 * ========================================================================= */
SEXP cuda_kernels_from_module(SEXP program_ptr,
                              SEXP context_ptr,
                              SEXP kernel_names) {
  void *module = get_checked_external_ptr(program_ptr,
                                          cuda_module,
                                          "cuda_module");
  CudaContext *ctx = get_checked_external_ptr(context_ptr,
                                              cuda_context,
                                              "cuda_context");
  cuda_activate_context(ctx);
  cuda_ensure_driver_init();
  
  if (TYPEOF(kernel_names) != STRSXP || LENGTH(kernel_names) < 1) {
    Rf_error("'kernel_names' must be a character vector of length 1 or greater");
  }
  
  R_xlen_t n = XLENGTH(kernel_names);
  SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
  
  for (R_xlen_t i = 0; i < n; i++) {
    const char *name = CHAR(STRING_ELT(kernel_names, i));
    
    CUfunction function;
    CUresult res = cuModuleGetFunction(&function, (CUmodule)module, name);
    if (res != CUDA_SUCCESS) {
      UNPROTECT(1);
      Rf_error("failed to get function '%s': %s",
               name, cuda_driver_error_string((int)res));
    }
    
    int max_threads = 0;
    CUresult attr_res = cuFuncGetAttribute(&max_threads,
                                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           function);
    if (attr_res != CUDA_SUCCESS) {
      UNPROTECT(1);
      Rf_error("failed to query max threads per block for '%s': %s",
               name, cuda_driver_error_string((int)attr_res));
    }
    
    CudaKernel *kern = malloc(sizeof(*kern));
    if (kern == NULL) {
      UNPROTECT(1);
      Rf_error("failed to allocate CudaKernel for '%s'", name);
    }
    kern->function = (void *)function;
    kern->max_threads_per_block = max_threads;
    kern->device_index = ctx->device_index;
    
    SEXP kernel_ptr = PROTECT(R_MakeExternalPtr(kern,
                                                cuda_kernel,
                                                R_NilValue));
    /* the anchor -- see the function-level comment above */
    Rf_setAttrib(kernel_ptr,
                Rf_install("ardea_parent_program"),
                program_ptr);
    set_externalptr_class(kernel_ptr, "cuda_kernel");
    R_RegisterCFinalizerEx(kernel_ptr, cuda_kernel_finalizer, TRUE);
    SET_VECTOR_ELT(out, i, kernel_ptr);
    UNPROTECT(1);
    /* -- kernel_ptr now referenced by out -------------------------------- */
  }
  
  UNPROTECT(1);
  return out;
}
