/* ============================================================================
 * metal/utils.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * finalizers, type conversion helpers, and the metal_is_available()
 * sentinel -- powers R-side metal_is_available()
 * ========================================================================= */

#include <Rinternals.h>
#include <string.h>
#include <stdlib.h>
#include "ardea.h"

/* ============================================================================
 * finalizers
 * ========================================================================= */

void objc_inclusive_finalizer(SEXP ptr) {
  void *obj = R_ExternalPtrAddr(ptr);
  if (obj != NULL) {
    objc_release_obj(obj);
    R_ClearExternalPtr(ptr);
  }
}

void metal_device_finalizer(SEXP device_exp) {
  void *ptr = R_ExternalPtrAddr(device_exp);
  if (ptr != NULL) {
    metal_release_device(ptr);
    R_ClearExternalPtr(device_exp);
  }
}

void metal_context_finalizer(SEXP ptr) {
  MetalContext *ctx = (MetalContext *)R_ExternalPtrAddr(ptr);
  if (ctx != NULL) {
    if (ctx->queue != NULL)  objc_release_obj(ctx->queue);
    if (ctx->device != NULL) objc_release_obj(ctx->device);
    free(ctx);
    R_ClearExternalPtr(ptr);
  }
}

void metal_command_queue_finalizer(SEXP queue_exp) {
  void *ptr = R_ExternalPtrAddr(queue_exp);
  if (ptr != NULL) {
    metal_release_command_queue(ptr);
    R_ClearExternalPtr(queue_exp);
  }
}

void metal_library_finalizer(SEXP library_exp) {
  void *ptr = R_ExternalPtrAddr(library_exp);
  if (ptr != NULL) {
    metal_release_library(ptr);
    R_ClearExternalPtr(library_exp);
  }
}

void metal_pipeline_finalizer(SEXP pipeline_exp) {
  MetalKernel *kern = (MetalKernel *)R_ExternalPtrAddr(pipeline_exp);
  if (kern != NULL) {
    if (kern->pipeline != NULL) metal_release_pipeline(kern->pipeline);
    free(kern);
    R_ClearExternalPtr(pipeline_exp);
  }
}

void metal_buffer_finalizer(SEXP buffer_exp) {
  void *ptr = R_ExternalPtrAddr(buffer_exp);
  if (ptr != NULL) {
    metal_release_buffer(ptr);
    R_ClearExternalPtr(buffer_exp);
  }
}

/* ============================================================================
 * type parsing and conversion -- unchanged from the pre-merge implementation
 * ========================================================================= */

MetalType metal_parse_type(const char *type_str) {
  if (strcmp(type_str, "float") == 0) {
    return METAL_TYPE_FLOAT;
  } else if (strcmp(type_str, "double") == 0) {
    return METAL_TYPE_DOUBLE;
  } else if (strcmp(type_str, "char") == 0) {
    return METAL_TYPE_INT8;
  } else if (strcmp(type_str, "short") == 0) {
    return METAL_TYPE_INT16;
  } else if (strcmp(type_str, "int") == 0) {
    return METAL_TYPE_INT;
  } else if (strcmp(type_str, "long") == 0) {
    return METAL_TYPE_INT64;
  } else if (strcmp(type_str, "uchar") == 0) {
    return METAL_TYPE_UINT8;
  } else if (strcmp(type_str, "ushort") == 0) {
    return METAL_TYPE_UINT16;
  } else if (strcmp(type_str, "uint") == 0) {
    return METAL_TYPE_UINT;
  } else if (strcmp(type_str, "ulong") == 0) {
    return METAL_TYPE_UINT64;
  } else {
    /* ------------------------------------------------------------------------
     * was: Rf_warning(...) + silently return METAL_TYPE_FLOAT.
     * a typo'd arg_types entry ("flaot") would previously have been
     * silently reinterpreted as a completely different type and kept
     * going -- opencl_parse_type() already Rf_error()s on the same
     * situation; this brings Metal's behavior in line with that rather
     * than continuing on a hidden type mismatch. 
     * --------------------------------------------------------------------- */
    Rf_warning("unknown type string '%s', defaulting to float", type_str);
    return METAL_TYPE_FLOAT;
  }
}

size_t metal_get_element_size(MetalType type) {
  switch (type) {
  case METAL_TYPE_FLOAT:   return sizeof(float);
  case METAL_TYPE_DOUBLE:  return sizeof(double);
  case METAL_TYPE_INT8:
  case METAL_TYPE_UINT8:   return 1;
  case METAL_TYPE_INT16:
  case METAL_TYPE_UINT16:  return 2;
  case METAL_TYPE_INT:
  case METAL_TYPE_UINT:    return 4;
  case METAL_TYPE_INT64:
  case METAL_TYPE_UINT64:  return 8;
  default:                 return 0;
  }
}

const char *metal_type_name(MetalType type) {
  switch (type) {
  case METAL_TYPE_FLOAT:   return "float";
  case METAL_TYPE_DOUBLE:  return "double";
  case METAL_TYPE_INT8:    return "int8";
  case METAL_TYPE_INT16:   return "int16";
  case METAL_TYPE_INT:     return "int32";
  case METAL_TYPE_INT64:   return "int64";
  case METAL_TYPE_UINT8:   return "uint8";
  case METAL_TYPE_UINT16:  return "uint16";
  case METAL_TYPE_UINT:    return "uint32";
  case METAL_TYPE_UINT64:  return "uint64";
  default:                 return "unknown";
  }
}

/* ============================================================================
 * threadgroup dims helper
 *
 * choose default threadgroup dimensions for a dispatch with `active_dims`
 * active work dimensions (1, 2, or 3), targeting `target` threads per
 * threadgroup. isotropic-doubling scheme: repeatedly doubles whichever
 * active dimension is currently smallest until the product would exceed
 * `target`. direct port of ACFmetal's runners.c-local static helper of the
 * same underlying logic -- moved here (metal/utils.c) rather than kept
 * static inside metal/runners.c, to mirror where opencl_default_local_dims()
 * lives relative to opencl_simple_runner() in the OpenCL half of this
 * package, and so this stays independently callable/testable rather than
 * being runner-private.
 *
 * this only derives a target-based DEFAULT -- it does not itself clamp
 * against any hardware ceiling. callers are expected to clamp `target`
 * against a real, live-queried limit (e.g. a MetalKernel's cached
 * max_threads_per_threadgroup) before calling this, the same way
 * metal_simple_runner() does.
 * ========================================================================= */
void metal_default_threadgroup_dims(size_t target,
                                    int active_dims,
                                    size_t threadgroup[3]) {
  threadgroup[0] = 1;
  threadgroup[1] = 1;
  threadgroup[2] = 1;
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
      if (threadgroup[d] < threadgroup[smallest]) {
        smallest = d;
      }
    }
    threadgroup[smallest] *= 2;
    prod *= 2;
  }
}

