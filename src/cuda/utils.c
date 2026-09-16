/* ============================================================================
 * cuda/utils.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * shared CUDA helpers: the context-activation discipline, type parsing,
 * finalizers, and the default block-dims helper (CUDA's own copy of the
 * same isotropic-doubling logic metal/utils.c's metal_default_threadgroup_
 * dims() uses -- kept as an independent copy rather than a cross-framework
 * call, since HAVE_CUDA and HAVE_METAL are independently optional and
 * CUDA code must not assume Metal's object file was even compiled).
 * ========================================================================= */

#include <cuda_runtime.h>
#include <cuda.h>
#include <Rinternals.h>
#include <stdlib.h>
#include <string.h>
#include "ardea.h"

/* ============================================================================
 * driver API one-time initialization
 * cuInit() must be called before ANY other driver API call. NVIDIA
 * documents repeated calls as safe/idempotent, so this is called
 * defensively at the top of every driver-API-touching entry point rather
 * than tracked with a package-load-time flag -- matches ACFcuda's own
 * lazy-per-call approach; a single call-once-at-package-load alternative
 * was considered but isn't obviously better given the documented
 * idempotency, and this way nothing has to coordinate with .onLoad.
 * ========================================================================= */
void cuda_ensure_driver_init(void) {
  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    Rf_error("cuInit() failed: %s", cuda_driver_error_string((int)res));
  }
}

/* ============================================================================
 * the context-activation discipline
 *
 * CUDA has no object that "is" a context the way cl_context/MTLDevice do.
 * cudaSetDevice(index) sets which GPU is current for the CALLING THREAD;
 * every subsequent call -- runtime or driver API -- implicitly targets
 * whatever was last set current. CudaContext is a proxy for that ambient
 * state, not a container that owns it.
 *
 * the rule this function exists to enforce: every C function that
 * receives a CudaContext* calls this as its literal first statement, no
 * exceptions. nothing in C's type system can force that -- forgetting it
 * compiles fine and just silently operates on whichever device happened
 * to be current from some earlier, unrelated call. the only real defense
 * available is failing loudly here if cudaSetDevice() itself reports a
 * problem, and consistent discipline everywhere else.
 * ========================================================================= */
void cuda_activate_context(CudaContext *ctx) {
  if (ctx == NULL) {
    Rf_error("internal error: NULL CudaContext passed to cuda_activate_context()");
  }
  cudaError_t err = cudaSetDevice(ctx->device_index);
  if (err != cudaSuccess) {
    Rf_error("failed to set CUDA device %d as current: %s",
             ctx->device_index,
             cudaGetErrorString(err));
  }
}

/* ============================================================================
 * driver API error strings
 * cuGetErrorString() (unlike cudaGetErrorString()) returns its result via
 * an out-parameter and can itself fail -- guard both.
 * ========================================================================= */
const char *cuda_driver_error_string(int cu_result) {
  const char *msg = NULL;
  CUresult lookup_result = cuGetErrorString((CUresult)cu_result, &msg);
  if (lookup_result != CUDA_SUCCESS || msg == NULL) {
    return "unknown CUDA driver error (cuGetErrorString itself failed)";
  }
  return msg;
}

/* ============================================================================
 * finalizers
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * a stream's owning device must be current before it can be destroyed --
 * NOT whatever device happens to be current when R's GC decides to run
 * this finalizer, which could easily be a different context's device if
 * more than one is open at once. this is the same activation discipline
 * as everywhere else, just applied at teardown time instead of use time.
 * ------------------------------------------------------------------------- */
void cuda_context_finalizer(SEXP context_exp) {
  CudaContext *ctx = (CudaContext *)R_ExternalPtrAddr(context_exp);
  if (ctx != NULL) {
    if (ctx->stream != NULL) {
      cudaSetDevice(ctx->device_index);
      cudaStreamDestroy((cudaStream_t)ctx->stream);
    }
    free(ctx);
    R_ClearExternalPtr(context_exp);
  }
}

/* ----------------------------------------------------------------------------
 * this is the finalizer that actually unloads the module -- and the whole
 * reason cuda_kernels_from_module() anchors the program SEXP as an
 * attribute on every kernel it builds: R's garbage collector will not run
 * THIS finalizer while any such attribute-holding kernel is still
 * reachable, which is what keeps every live CudaKernel.function valid.
 * ------------------------------------------------------------------------- */
void cuda_module_finalizer(SEXP module_exp) {
  void *module = R_ExternalPtrAddr(module_exp);
  if (module != NULL) {
    cuModuleUnload((CUmodule)module);
    R_ClearExternalPtr(module_exp);
  }
}

/* ----------------------------------------------------------------------------
 * only frees the small CudaKernel struct itself -- function/module are
 * NOT owned here, so nothing CUDA-side is released. see cuda_module_
 * finalizer() above for where that actually happens, and the R-level
 * attribute anchor in cuda_kernels_from_module() for why it's safe for
 * this finalizer to run in any order relative to that one.
 * ------------------------------------------------------------------------- */
void cuda_kernel_finalizer(SEXP kernel_exp) {
  CudaKernel *kern = (CudaKernel *)R_ExternalPtrAddr(kernel_exp);
  if (kern != NULL) {
    free(kern);
    R_ClearExternalPtr(kernel_exp);
  }
}

/* ============================================================================
 * type parsing / sizing
 * unrecognized strings Rf_error() rather than silently defaulting --
 * matches opencl_parse_type()'s behavior and metal_parse_type()'s
 * corrected behavior (see metal/utils.c's own note on why a silent
 * fallback here is worse than a hard stop).
 * ========================================================================= */
CudaType cuda_parse_type(const char *type_str) {
  if (type_str == NULL) {
    Rf_error("type string is NULL");
  }
  if (strcmp(type_str, "float") == 0) {
    return CUDA_TYPE_FLOAT;
  } else if (strcmp(type_str, "double") == 0) {
    return CUDA_TYPE_DOUBLE;
  } else if (strcmp(type_str, "int8") == 0 || strcmp(type_str, "byte") == 0) {
    return CUDA_TYPE_INT8;
  } else if (strcmp(type_str, "int16") == 0 || strcmp(type_str, "short") == 0) {
    return CUDA_TYPE_INT16;
  } else if (strcmp(type_str, "int") == 0 || strcmp(type_str, "int32") == 0) {
    return CUDA_TYPE_INT;
  } else if (strcmp(type_str, "long") == 0 || strcmp(type_str, "int64") == 0) {
    return CUDA_TYPE_INT64;
  } else if (strcmp(type_str, "uint8") == 0 || strcmp(type_str, "ubyte") == 0) {
    return CUDA_TYPE_UINT8;
  } else if (strcmp(type_str, "uint16") == 0 || strcmp(type_str, "ushort") == 0) {
    return CUDA_TYPE_UINT16;
  } else if (strcmp(type_str, "uint") == 0 || strcmp(type_str, "uint32") == 0) {
    return CUDA_TYPE_UINT;
  } else if (strcmp(type_str, "ulong") == 0 || strcmp(type_str, "uint64") == 0) {
    return CUDA_TYPE_UINT64;
  } else {
    Rf_error("unrecognized type string: '%s'", type_str);
  }
}

size_t cuda_get_element_size(CudaType type) {
  switch (type) {
  case CUDA_TYPE_FLOAT:   return sizeof(float);
  case CUDA_TYPE_DOUBLE:  return sizeof(double);
  case CUDA_TYPE_INT8:    return sizeof(int8_t);
  case CUDA_TYPE_INT16:   return sizeof(int16_t);
  case CUDA_TYPE_INT:     return sizeof(int32_t);
  case CUDA_TYPE_INT64:   return sizeof(int64_t);
  case CUDA_TYPE_UINT8:   return sizeof(uint8_t);
  case CUDA_TYPE_UINT16:  return sizeof(uint16_t);
  case CUDA_TYPE_UINT:    return sizeof(uint32_t);
  case CUDA_TYPE_UINT64:  return sizeof(uint64_t);
  default:
    Rf_error("cuda_get_element_size(): unrecognized CudaType value (%d)",
             (int)type);
  }
}

const char *cuda_type_name(CudaType type) {
  switch (type) {
  case CUDA_TYPE_FLOAT:   return "float";
  case CUDA_TYPE_DOUBLE:  return "double";
  case CUDA_TYPE_INT8:    return "int8";
  case CUDA_TYPE_INT16:   return "int16";
  case CUDA_TYPE_INT:     return "int32";
  case CUDA_TYPE_INT64:   return "int64";
  case CUDA_TYPE_UINT8:   return "uint8";
  case CUDA_TYPE_UINT16:  return "uint16";
  case CUDA_TYPE_UINT:    return "uint32";
  case CUDA_TYPE_UINT64:  return "uint64";
  default:                return "unknown";
  }
}

/* ============================================================================
 * default block dims -- CUDA's own copy of metal_default_threadgroup_
 * dims()'s isotropic-doubling scheme. see that function's comment
 * (metal/utils.c) for the full rationale; identical logic, duplicated
 * rather than shared, since HAVE_METAL/HAVE_CUDA are independently
 * optional.
 * ========================================================================= */
void cuda_default_block_dims(size_t target,
                             int active_dims,
                             size_t block[3]) {
  block[0] = 1;
  block[1] = 1;
  block[2] = 1;
  if (active_dims < 1) {
    active_dims = 1;
  }
  if (active_dims > 3) {
    active_dims = 3;
  }
  
  size_t prod = 1;
  while (prod * 2 <= target) {
    int smallest = 0;
    for (int d = 1; d < active_dims; d++) {
      if (block[d] < block[smallest]) {
        smallest = d;
      }
    }
    block[smallest] *= 2;
    prod *= 2;
  }
}
