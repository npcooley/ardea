/* ============================================================================
 * opencl/contexts.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * OpenCL context construction, this supports the R side function:
 * opencl_make_context()
 *
 * a context binds one (or more) devices, on a specific platform, into a
 * single object that programs are built against and kernels are dispatched
 * into. CL_CONTEXT_PLATFORM is set explicitly rather than omitted, since
 * omitting it is only safe on systems with exactly one ICD installed --
 * see the discussion in devices.c about platform_id being required for
 * unambiguous context creation on multi-ICD systems (common on Linux).
 * 
 * currently supporting binding a single device to a context, though openCL
 * allows binding of multiple devices from the same platform to a single
 * context
 * 
 * we set the platform explicitly in the case of multi-ICD systems
 * ========================================================================= */

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif
#include <R.h>
#include <Rinternals.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ardea.h"

/* ============================================================================
 * non-static helpers
 * ========================================================================= */

void opencl_context_finalizer(SEXP ptr) {
  OpenCLContext *ctx = (OpenCLContext *)R_ExternalPtrAddr(ptr);
  if (ctx != NULL) {
    if (ctx->queue != NULL) clReleaseCommandQueue(ctx->queue);
    if (ctx->context != NULL) clReleaseContext(ctx->context);
    free(ctx);
    R_ClearExternalPtr(ptr);
  }
}

/* ============================================================================
 * context construction
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * clCreateContext for a single device, explicitly scoped to its platform via
 * CL_CONTEXT_PLATFORM. returns a tagged, finalized externalptr wrapping the
 * created cl_context.
 * ------------------------------------------------------------------------- */
SEXP opencl_context_from_device(SEXP device_ptr,
                                SEXP platform_ptr) {
  cl_device_id device = get_checked_external_ptr(device_ptr,
                                                 opencl_device_id,
                                                 "opencl_device_id");
  cl_platform_id platform = get_checked_external_ptr(platform_ptr,
                                                     opencl_platform_id,
                                                     "opencl_platform_id");
  
  cl_context_properties props[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platform,
    0
  };
  cl_int err = CL_SUCCESS;
  
  cl_context context = clCreateContext(props,
                                       1,
                                       &device,
                                       NULL,
                                       NULL,
                                       &err);
  if (err != CL_SUCCESS || context == NULL) {
    Rf_error("clCreateContext failed (CL error %d)",
             err);
  }
  
#ifdef CL_VERSION_2_0
  /* --------------------------------------------------------------------------
   * post 2.0 function path
   * 0-terminated property list;
   * empty here since no special
   * queue properties are needed 
   * ----------------------------------------------------------------------- */
  cl_queue_properties queue_props[] = { 0 };
  cl_command_queue queue = clCreateCommandQueueWithProperties(context,
                                                              device,
                                                              queue_props,
                                                              &err);
#else
  /* --------------------------------------------------------------------------
   * pre 2.0 function path, most likely an apple (1.2) system ...
   * ----------------------------------------------------------------------- */
  cl_command_queue queue = clCreateCommandQueue(context,
                                                device,
                                                0,
                                                &err);
#endif
  if (err != CL_SUCCESS || queue == NULL) {
    clReleaseContext(context);
    Rf_error("clCreateCommandQueue failed (CL error %d)",
             err);
  }
  
  OpenCLContext *ctx = calloc(1, sizeof(*ctx));
  if (ctx == NULL) {
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    Rf_error("failed to allocate OpenCLContext");
  }
  ctx->context = context;
  ctx->queue = queue;
  ctx->device = device;
  
  SEXP context_ptr = PROTECT(R_MakeExternalPtr(ctx,
                                               opencl_context,
                                               R_NilValue));
  set_externalptr_class(context_ptr,
                        "opencl_context");
  R_RegisterCFinalizerEx(context_ptr,
                         opencl_context_finalizer,
                         TRUE);
  UNPROTECT(1);
  return context_ptr;
}
