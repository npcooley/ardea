/* ============================================================================
 * opencl/runners.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * R-callable entry point for simple OpenCL kernel dispatch.
 *
 * only supports numeric ("double") and integer ("long") arguments and
 * results for now, matching buffers.c's currently implemented type pair.
 *
 * argument convention:
 *   arg_list[[1]] / arg_types[1] is the OUTPUT TEMPLATE -- its length and
 *   declared type define a write-only output buffer; its VALUES are never
 *   copied to the device.
 *   arg_list[[2:n]] are actual inputs. an argument of length 1 is bound
 *   directly as a scalar kernel argument (clSetKernelArg with no cl_mem);
 *   an argument of length > 1 is staged into a read-only device buffer.
 * ========================================================================= */

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <R.h>
#include <Rinternals.h>
#include <stdlib.h>
#include <string.h>
#include "ardea.h"

SEXP opencl_simple_runner(SEXP context_ptr,
                          SEXP kernel_ptr,
                          SEXP arg_types,
                          SEXP arg_list,
                          SEXP work_dims,
                          SEXP local_dims,
                          SEXP target_local_size) {
  
  /* -- validate context and kernel handles -------------------------------- */
  OpenCLContext *ctx = get_checked_external_ptr(context_ptr,
                                                opencl_context,
                                                "opencl_context");
  cl_kernel kernel = get_checked_external_ptr(kernel_ptr,
                                              opencl_kernel,
                                              "opencl_kernel");
  
  /* -- validate arg_types / arg_list shapes ------------------------------- */
  if (TYPEOF(arg_types) != STRSXP || LENGTH(arg_types) < 1) {
    Rf_error("'arg_types' must be a non-empty character vector");
  }
  if (TYPEOF(arg_list) != VECSXP) {
    Rf_error("'arg_list' must be an R list");
  }
  if (LENGTH(arg_types) != LENGTH(arg_list)) {
    Rf_error("length of 'arg_types' must equal length of 'arg_list'");
  }
  int total_args = LENGTH(arg_types);
  
  /* --------------------------------------------------------------------------
   * output template: arg_list[[0]] / arg_types[0] 
   * length and type only -- values are never copied to the device 
   * ----------------------------------------------------------------------- */
  SEXP output_template = VECTOR_ELT(arg_list, 0);
  if (TYPEOF(output_template) != REALSXP && TYPEOF(output_template) != INTSXP) {
    Rf_error("first element of 'arg_list' must be a numeric or integer "
               "vector representing the output template");
  }
  OpenCLType output_type = opencl_parse_type(CHAR(STRING_ELT(arg_types, 0)));
  R_xlen_t output_length = XLENGTH(output_template);
  size_t output_bytes = (size_t)output_length * opencl_type_size(output_type);
  
  /* --------------------------------------------------------------------------
   * resolve global (logical) work size 
   * explicit work_dims, else inferred from output_template's dim attribute,
   * else {output_length, 1, 1} -- same resolution order as the CUDA runner 
   * ----------------------------------------------------------------------- */
  size_t work_dims_curr[3] = {(size_t)output_length, 1, 1};
  if (work_dims != R_NilValue) {
    if ((TYPEOF(work_dims) != REALSXP && TYPEOF(work_dims) != INTSXP) ||
        LENGTH(work_dims) != 3) {
      Rf_error("'work_dims' must be a numeric or integer vector of length 3");
    }
    for (int d = 0; d < 3; d++) {
      double v = (TYPEOF(work_dims) == REALSXP) ?
      REAL(work_dims)[d] : (double)INTEGER(work_dims)[d];
      if (v < 1) {
        Rf_error("'work_dims' values must be positive");
      }
      work_dims_curr[d] = (size_t)v;
    }
  } else {
    SEXP dims = Rf_getAttrib(output_template,
                             R_DimSymbol);
    if (dims != R_NilValue) {
      int ndims = LENGTH(dims);
      work_dims_curr[0] = (ndims >= 1) ? (size_t)INTEGER(dims)[0] : (size_t)output_length;
      work_dims_curr[1] = (ndims >= 2) ? (size_t)INTEGER(dims)[1] : 1;
      work_dims_curr[2] = (ndims >= 3) ? (size_t)INTEGER(dims)[2] : 1;
    }
  }
  
  int active_dims = 1;
  if (work_dims_curr[2] > 1) {
    active_dims = 3;
  } else if (work_dims_curr[1] > 1) {
    active_dims = 2;
  }
  
  /* --------------------------------------------------------------------------
   * resolve local (work-group) size 
   * explicit local_dims, validated against this device's real limits, or
   * derived from target_local_size (default 256) via opencl_default_local_dims,
   * which already clamps to the live-queried device limits internally 
   * ----------------------------------------------------------------------- */
  size_t local[3] = {1, 1, 1};
  if (local_dims != R_NilValue) {
    if ((TYPEOF(local_dims) != REALSXP && TYPEOF(local_dims) != INTSXP) ||
        LENGTH(local_dims) != 3) {
      Rf_error("'local_dims' must be a numeric or integer vector of length 3");
    }
    size_t max_group_size = 0;
    query_device_units(ctx->device,
                       CL_DEVICE_MAX_WORK_GROUP_SIZE,
                       sizeof(max_group_size),
                       &max_group_size,
                       "CL_DEVICE_MAX_WORK_GROUP_SIZE");
    size_t product = 1;
    for (int d = 0; d < 3; d++) {
      double v = (TYPEOF(local_dims) == REALSXP) ?
      REAL(local_dims)[d] : (double)INTEGER(local_dims)[d];
      if (v < 1) {
        Rf_error("'local_dims' values must be positive");
      }
      local[d] = (size_t)v;
      product *= local[d];
    }
    if (product > max_group_size) {
      Rf_error("'local_dims' implies a work-group size of %zu, exceeding "
                 "this device's maximum of %zu", product, max_group_size);
    }
  } else {
    size_t target = 256;
    if (target_local_size != R_NilValue) {
      if ((TYPEOF(target_local_size) != REALSXP &&
          TYPEOF(target_local_size) != INTSXP) ||
          LENGTH(target_local_size) != 1) {
        Rf_error("'target_local_size' must be a single numeric or integer value");
      }
      double v = (TYPEOF(target_local_size) == REALSXP) ?
      REAL(target_local_size)[0] : (double)INTEGER(target_local_size)[0];
      if (v < 1) {
        Rf_error("'target_local_size' must be a positive value");
      }
      target = (size_t)v;
    }
    opencl_default_local_dims(ctx->device,
                              target,
                              active_dims,
                              local);
  }
  
  /* --------------------------------------------------------------------------
   * global size must be a multiple of local size per dimension 
   * more work-items may launch than output_length -- kernels are
   * responsible for bounds-checking get_global_id() against the true
   * output length, same convention as the CUDA runner's rounded-up grid 
   * ----------------------------------------------------------------------- */
  size_t global[3];
  for (int d = 0; d < 3; d++) {
    global[d] = ((work_dims_curr[d] + local[d] - 1) / local[d]) * local[d];
  }
  
  /* -- allocate buffer-tracking array, index 0 reserved for output -------- */
  cl_mem *buffers = calloc((size_t)total_args, sizeof(*buffers));
  if (buffers == NULL) {
    Rf_error("failed to allocate buffer-tracking array");
  }
  
  cl_int err = CL_SUCCESS;
  buffers[0] = clCreateBuffer(ctx->context,
                              CL_MEM_WRITE_ONLY,
                              output_bytes,
                              NULL, &err);
  if (err != CL_SUCCESS || buffers[0] == NULL) {
    release_runner_buffers(buffers, total_args);
    Rf_error("clCreateBuffer failed for output (CL error %d)", err);
  }
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffers[0]);
  if (err != CL_SUCCESS) {
    release_runner_buffers(buffers, total_args);
    Rf_error("clSetKernelArg failed for output argument (CL error %d)", err);
  }
  
  /* --------------------------------------------------------------------------
   * validate every argument's R type against its declared arg_types
   * string, AND scan for NA/NaN, BEFORE creating any device buffers. this
   * is a pure check, no allocation -- so by the time the loop below runs,
   * neither a type mismatch nor an NA can occur mid-way through resource
   * creation.
   *
   * the loop following this calling Rf_error can create a long jump that
   * would leak buffers created before an error in the loop, so we do an
   * overhead check of our args beforehand.
   *
   * the NA/NaN scan is not redundant with marshal_r_vector()'s own
   * NA_INTEGER check (buffers.c) -- that check only covers the LONG path,
   * fires from INSIDE the second loop below, and by the time it fires,
   * any cl_mem buffers already created for earlier arguments in that loop
   * are native resources with no R-tracked owner, so Rf_error()'s longjmp
   * would skip right past them: a silent, permanent leak on every
   * NA-triggered error, not merely a delayed one. checking here, before
   * any buffer exists, leaves nothing for that longjmp to skip past.
   * this also closes a second, previously-silent gap: the DOUBLE/FLOAT
   * paths in marshal_r_vector() have no NA/NaN check at all today, so an
   * NA_real_ argument would currently be sent to the GPU as an ordinary
   * NaN with no error raised anywhere -- this scan catches that case too,
   * not just the LONG/NA_INTEGER one.
   * ----------------------------------------------------------------------- */
  for (int i = 1; i < total_args; i++) {
    SEXP arg = VECTOR_ELT(arg_list,
                          i);
    OpenCLType arg_type = opencl_parse_type(CHAR(STRING_ELT(arg_types,
                                                            i)));
    int type_ok = (arg_type == OPENCL_TYPE_DOUBLE || arg_type == OPENCL_TYPE_FLOAT) ?
    (TYPEOF(arg) == REALSXP) :
      (arg_type == OPENCL_TYPE_INT || arg_type == OPENCL_TYPE_LONG) ?
    (TYPEOF(arg) == INTSXP) : 0;
    if (!type_ok) {
      Rf_error("argument %d's R type does not match its declared arg_types "
                 "value ('%s')",
                 i,
                 CHAR(STRING_ELT(arg_types,
                                 i)));
    }
    
    R_xlen_t arg_len = XLENGTH(arg);
    if (TYPEOF(arg) == REALSXP) {
      double *d = REAL(arg);
      for (R_xlen_t j = 0; j < arg_len; j++) {
        if (ISNA(d[j]) || ISNAN(d[j])) {
          Rf_error("NA/NaN present in argument %d at position %ld -- "
                     "OpenCL has no representation for R's NA",
                     i,
                     (long)(j + 1));
        }
      }
    } else {
      int *iv = INTEGER(arg);
      for (R_xlen_t j = 0; j < arg_len; j++) {
        if (iv[j] == NA_INTEGER) {
          Rf_error("NA present in argument %d at position %ld -- "
                     "OpenCL has no representation for R's NA",
                     i,
                     (long)(j + 1));
        }
      }
    }
  }
  
  for (int i = 1; i < total_args; i++) {
    SEXP arg = VECTOR_ELT(arg_list,
                          i);
    OpenCLType arg_type = opencl_parse_type(CHAR(STRING_ELT(arg_types,
                                                            i)));
    R_xlen_t arg_length = XLENGTH(arg);
    
    size_t bytes = 0;
    void *host_buf = marshal_r_vector(arg,
                                      arg_type,
                                      &bytes);
    
    if (arg_length == 1) {
      /* ----------------------------------------------------------------------
       * scalar: clSetKernelArg copies the value immediately (per spec),
       * so the host buffer can be freed right after binding 
       * ------------------------------------------------------------------- */
      err = clSetKernelArg(kernel,
                           (cl_uint)i,
                           bytes,
                           host_buf);
      free(host_buf);
      if (err != CL_SUCCESS) {
        release_runner_buffers(buffers, total_args);
        Rf_error("clSetKernelArg failed for argument %d (CL error %d)",
                 i,
                 err);
      }
    } else {
      /* ----------------------------------------------------------------------
       * buffer: read-only, written once, blocking (gives synchronization
       * for free, no separate clFinish needed before the write completes) 
       * ------------------------------------------------------------------- */
      buffers[i] = clCreateBuffer(ctx->context,
                                  CL_MEM_READ_ONLY,
                                  bytes,
                                  NULL,
                                  &err);
      if (err != CL_SUCCESS || buffers[i] == NULL) {
        free(host_buf);
        release_runner_buffers(buffers, total_args);
        Rf_error("clCreateBuffer failed for argument %d (CL error %d)",
                 i,
                 err);
      }
      err = clEnqueueWriteBuffer(ctx->queue,
                                 buffers[i],
                                 CL_TRUE,
                                 0,
                                 bytes,
                                 host_buf,
                                 0,
                                 NULL,
                                 NULL);
      free(host_buf);
      if (err != CL_SUCCESS) {
        release_runner_buffers(buffers,
                               total_args);
        Rf_error("clEnqueueWriteBuffer failed for argument %d (CL error %d)",
                 i,
                 err);
      }
      err = clSetKernelArg(kernel,
                           (cl_uint)i,
                           sizeof(cl_mem),
                           &buffers[i]);
      if (err != CL_SUCCESS) {
        release_runner_buffers(buffers,
                               total_args);
        Rf_error("clSetKernelArg failed for argument %d (CL error %d)",
                 i,
                 err);
      }
    }
  }
  
  /* -- dispatch ------------------------------------------------------------*/
  err = clEnqueueNDRangeKernel(ctx->queue,
                               kernel,
                               active_dims,
                               NULL,
                               global,
                               local,
                               0,
                               NULL,
                               NULL);
  if (err != CL_SUCCESS) {
    release_runner_buffers(buffers,
                           total_args);
    Rf_error("clEnqueueNDRangeKernel failed (CL error %d)",
             err);
  }
  
  /* -- blocking read-back -- CL_TRUE gives synchronization for free ------- */
  void *raw_output = malloc(output_bytes);
  if (raw_output == NULL) {
    release_runner_buffers(buffers,
                           total_args);
    Rf_error("failed to allocate %zu bytes for output read-back",
             output_bytes);
  }
  err = clEnqueueReadBuffer(ctx->queue,
                            buffers[0],
                            CL_TRUE,
                            0,
                            output_bytes,
                            raw_output,
                            0,
                            NULL,
                            NULL);
  if (err != CL_SUCCESS) {
    free(raw_output);
    release_runner_buffers(buffers,
                           total_args);
    Rf_error("clEnqueueReadBuffer failed (CL error %d)",
             err);
  }
  
  SEXP result = PROTECT(unmarshal_to_r_vector(raw_output,
                                              output_type,
                                              output_length));
  free(raw_output);
  release_runner_buffers(buffers,
                         total_args);
  
  UNPROTECT(1);
  return result;
}
