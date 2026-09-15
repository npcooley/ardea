/* ============================================================================
 * metal/runners.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * R-callable entry point for simple Metal kernel dispatch.
 *
 * argument convention (identical to opencl_simple_runner()):
 *   arg_list[[1]] / arg_types[1] is the OUTPUT TEMPLATE -- its length and
 *   declared type define a write-only output buffer; its VALUES are never
 *   copied to the device.
 *   arg_list[[2:n]] are actual inputs. an argument of length 1 is bound
 *   directly as a scalar kernel argument (setBytes:, no MTLBuffer); an
 *   argument of length > 1 is staged into a Shared-storage device buffer.
 *
 * typing is PERMISSIVE, unlike opencl_simple_runner()'s strict type-family
 * matching: any REALSXP or INTSXP argument is accepted regardless of its
 * declared arg_types entry, and narrowed/widened as needed by
 * metal_convert_r_numeric_to_buffer()/metal_convert_r_int_to_buffer(). this
 * was a deliberate choice (see prior design discussion) -- Metal's
 * MetalType covers 8 integer widths where OpenCL's OpenCLType only covers
 * 2, and requiring as.integer()/as.double() coercion before every
 * int8/uint16/etc. argument was judged worse than the convenience cost of
 * permissive coercion.
 *
 * three corrections relative to ACFmetal's metal_simple_runner(), the
 * source this was ported from:
 *
 * 1. no pipeline is built here. metal_make_kernelptr() already built and
 *    cached one (MetalKernel.pipeline), specifically so this function
 *    doesn't have to rebuild it on every single dispatch the way
 *    ACFmetal's runner did.
 * 2. kern->device_registry_id is checked against the context's actual
 *    device before anything else runs -- this field existed on MetalKernel
 *    before this file did, put there for exactly this check, which had
 *    never actually been wired up until now.
 * 3. after waiting for the command buffer, its status/error is checked
 *    BEFORE the output buffer is read. ACFmetal's command.m already
 *    exposed the means to do this (its own comment says as much) but its
 *    runner never called it -- a GPU-side failure would previously come
 *    back as silently-wrong output rather than an R error.
 * ========================================================================= */

#include <Rinternals.h>
#include <stdlib.h>
#include <string.h>
#include "ardea.h"

/* ============================================================================
 * cleanup helper
 *
 * every error path in the argument-staging loop below needs to release
 * whatever buffers/scalars have already been created before Rf_error()
 * unwinds the stack -- collapsed into one helper rather than repeating the
 * same four-way free()/metal_release_buffer() block at each call site
 * (the shape ACFmetal's runner used, copy-pasted at every error branch).
 * `buffers_created`/`scalars_created` are counts, not capacities -- only
 * the slots actually populated so far are touched.
 * ========================================================================= */
static void metal_runner_cleanup(void **input_buffers,
                                 int buffers_created,
                                 void **scalar_values,
                                 int scalars_created,
                                 size_t *scalar_sizes,
                                 int *is_scalar) {
  if (input_buffers != NULL) {
    for (int i = 0; i < buffers_created; i++) {
      if (input_buffers[i] != NULL) {
        metal_release_buffer(input_buffers[i]);
      }
    }
    free(input_buffers);
  }
  if (scalar_values != NULL) {
    for (int i = 0; i < scalars_created; i++) {
      if (scalar_values[i] != NULL) {
        free(scalar_values[i]);
      }
    }
    free(scalar_values);
  }
  free(scalar_sizes);
  free(is_scalar);
}

SEXP metal_simple_runner(SEXP context_ptr,
                         SEXP kernel_ptr,
                         SEXP arg_types,
                         SEXP arg_list,
                         SEXP work_dims,
                         SEXP threadgroup_dims,
                         SEXP threads_per_threadgroup) {
  
  /* -- validate context and kernel handles -------------------------------- */
  MetalContext *ctx = get_checked_external_ptr(context_ptr,
                                               metal_context,
                                               "metal_context");
  MetalKernel *kern = get_checked_external_ptr(kernel_ptr,
                                               metal_pipeline,
                                               "metal_pipeline");
  
  /* -- device-identity check -----------------------------------------------
   * a pipeline compiled against one physical device cannot be dispatched
   * through a different device's queue -- MetalKernel has carried
   * device_registry_id since metal_make_kernelptr() was written; this is
   * the first place that actually checks it */
  uint64_t ctx_registry_id = metal_device_registry_id(ctx->device);
  if (kern->device_registry_id != ctx_registry_id) {
    Rf_error("'kernel_ptr' was built against a different Metal device "
             "(registry id %llu) than 'context_ptr' is bound to "
             "(registry id %llu) -- a kernel must be dispatched through "
             "the same context it was built with",
             (unsigned long long)kern->device_registry_id,
             (unsigned long long)ctx_registry_id);
  }
  if (kern->max_threads_per_threadgroup == 0) {
    Rf_error("this kernel's cached maxTotalThreadsPerThreadgroup is 0 -- "
             "the pipeline may not have been built correctly");
  }
  
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
  MetalType output_type = metal_parse_type(CHAR(STRING_ELT(arg_types, 0)));
  R_xlen_t output_length = XLENGTH(output_template);
  size_t output_element_size = metal_get_element_size(output_type);
  size_t output_bytes = (size_t)output_length * output_element_size;
  
  /* --------------------------------------------------------------------------
   * resolve global (logical) work size
   * same resolution order as opencl_simple_runner(): explicit work_dims,
   * else inferred from output_template's dim attribute, else
   * {output_length, 1, 1}
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
    SEXP dims = Rf_getAttrib(output_template, R_DimSymbol);
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
   * resolve threadgroup (local) size
   * hardware ceiling is the MetalKernel's cached max_threads_per_threadgroup
   * -- queried once, at metal_make_kernelptr() time, against this specific
   * compiled pipeline. NOT a fresh per-dispatch query (unlike ACFmetal's
   * runner) and NOT metal_device_information()'s per-DEVICE number either
   * -- see the MetalKernel struct comment in ardea.h for why those two
   * numbers differ.
   * ----------------------------------------------------------------------- */
  size_t hw_max_threadgroup = kern->max_threads_per_threadgroup;
  
  size_t threadgroup_size[3];
  if (threadgroup_dims != R_NilValue) {
    if ((TYPEOF(threadgroup_dims) != REALSXP && TYPEOF(threadgroup_dims) != INTSXP) ||
        LENGTH(threadgroup_dims) != 3) {
      Rf_error("'threadgroup_dims' must be a numeric or integer vector of length 3");
    }
    size_t requested_total = 1;
    for (int d = 0; d < 3; d++) {
      double v = (TYPEOF(threadgroup_dims) == REALSXP) ?
        REAL(threadgroup_dims)[d] : (double)INTEGER(threadgroup_dims)[d];
      if (v < 1) {
        Rf_error("'threadgroup_dims' values must be positive");
      }
      threadgroup_size[d] = (size_t)v;
      requested_total *= threadgroup_size[d];
    }
    if (requested_total > hw_max_threadgroup) {
      Rf_error("'threadgroup_dims' implies a threadgroup size of %zu, "
               "exceeding this pipeline's maximum of %zu",
               requested_total, hw_max_threadgroup);
    }
  } else {
    size_t target = 256;
    if (threads_per_threadgroup != R_NilValue) {
      if ((TYPEOF(threads_per_threadgroup) != REALSXP &&
          TYPEOF(threads_per_threadgroup) != INTSXP) ||
          LENGTH(threads_per_threadgroup) != 1) {
        Rf_error("'threads_per_threadgroup' must be a single numeric or integer value");
      }
      double v = (TYPEOF(threads_per_threadgroup) == REALSXP) ?
        REAL(threads_per_threadgroup)[0] : (double)INTEGER(threads_per_threadgroup)[0];
      if (v < 1) {
        Rf_error("'threads_per_threadgroup' must be a positive value");
      }
      target = (size_t)v;
    }
    if (target > hw_max_threadgroup) {
      Rf_warning("'threads_per_threadgroup' (%zu) exceeds this pipeline's "
                "maximum of %zu; clamping", target, hw_max_threadgroup);
      target = hw_max_threadgroup;
    }
    metal_default_threadgroup_dims(target, active_dims, threadgroup_size);
  }
  
  size_t num_threadgroups[3];
  for (int d = 0; d < 3; d++) {
    num_threadgroups[d] = (work_dims_curr[d] + threadgroup_size[d] - 1) / threadgroup_size[d];
  }
  
  /* --------------------------------------------------------------------------
   * front validation pass: every non-output argument must be a plain
   * numeric/integer R vector, and must contain no NA/NaN.
   *
   * the NA/NaN scan matters more than it looks: metal_convert_r_numeric_
   * to_buffer()/metal_convert_r_int_to_buffer() (metal/buffers.c) already
   * Rf_error() on NA themselves, as a second, independent safety layer --
   * but if that were the ONLY place NA got caught, it would fire in the
   * MIDDLE of the buffer-staging loop below, after some number of device
   * buffers have already been malloc'd/created. Rf_error() longjmps; none
   * of those already-created buffers are R-tracked externalptrs, so
   * nothing would ever free them -- a silent, permanent native memory
   * leak on every NA-triggered error, not just a delayed one. scanning
   * for NA here, before any buffer exists, means there is nothing left to
   * leak by the time the staging loop's own Rf_error() calls could fire.
   * (this same failure mode exists in opencl_simple_runner()'s equivalent
   * loop, calling marshal_r_vector() mid-loop -- out of scope for this
   * change, but worth knowing it's there too.)
   * ----------------------------------------------------------------------- */
  for (int i = 1; i < total_args; i++) {
    SEXP arg = VECTOR_ELT(arg_list, i);
    if (TYPEOF(arg) != REALSXP && TYPEOF(arg) != INTSXP) {
      Rf_error("element %d of 'arg_list' must be a numeric or integer vector",
               i + 1);
    }
    R_xlen_t arg_len = XLENGTH(arg);
    if (TYPEOF(arg) == REALSXP) {
      double *d = REAL(arg);
      for (R_xlen_t j = 0; j < arg_len; j++) {
        if (ISNA(d[j]) || ISNAN(d[j])) {
          Rf_error("NA/NaN present in element %d of 'arg_list' at position "
                   "%ld -- Metal has no representation for R's NA",
                   i + 1, (long)(j + 1));
        }
      }
    } else {
      int *iv = INTEGER(arg);
      for (R_xlen_t j = 0; j < arg_len; j++) {
        if (iv[j] == NA_INTEGER) {
          Rf_error("NA present in element %d of 'arg_list' at position "
                   "%ld -- Metal has no representation for R's NA",
                   i + 1, (long)(j + 1));
        }
      }
    }
  }
  
  /* -- buffer/scalar bookkeeping -------------------------------------------- */
  int buffer_count = 1; /* index 0 reserved for output */
  int scalar_count = 0;
  for (int i = 1; i < total_args; i++) {
    if (XLENGTH(VECTOR_ELT(arg_list, i)) == 1) {
      scalar_count++;
    } else {
      buffer_count++;
    }
  }
  int total_args_for_kernel = (buffer_count - 1) + scalar_count;
  
  void **input_buffers = malloc((size_t)buffer_count * sizeof(*input_buffers));
  void **scalar_values = malloc((size_t)(scalar_count == 0 ? 1 : scalar_count) *
                                sizeof(*scalar_values));
  size_t *scalar_sizes = malloc((size_t)(scalar_count == 0 ? 1 : scalar_count) *
                                sizeof(*scalar_sizes));
  int *is_scalar = malloc((size_t)(total_args_for_kernel == 0 ? 1 : total_args_for_kernel) *
                          sizeof(*is_scalar));
  
  if (input_buffers == NULL || scalar_values == NULL ||
      scalar_sizes == NULL || is_scalar == NULL) {
    metal_runner_cleanup(input_buffers, 0, scalar_values, 0, scalar_sizes, is_scalar);
    Rf_error("failed to allocate argument tracking arrays");
  }
  
  /* -- output buffer --------------------------------------------------------- */
  input_buffers[0] = metal_create_buffer(ctx->device,
                                         output_bytes,
                                         METAL_STORAGE_SHARED);
  if (input_buffers[0] == NULL) {
    metal_runner_cleanup(input_buffers, 0, scalar_values, 0, scalar_sizes, is_scalar);
    Rf_error("failed to create output buffer");
  }
  
  /* -- walk remaining args, staging buffers/scalars ------------------------- */
  int buffer_idx = 1;
  int scalar_idx = 0;
  int arg_position = 0;
  
  for (int i = 1; i < total_args; i++) {
    SEXP arg = VECTOR_ELT(arg_list, i);
    MetalType arg_type = metal_parse_type(CHAR(STRING_ELT(arg_types, i)));
    size_t arg_element_size = metal_get_element_size(arg_type);
    R_xlen_t arg_length = XLENGTH(arg);
    
    if (arg_length == 1) {
      is_scalar[arg_position++] = 1;
      scalar_sizes[scalar_idx] = arg_element_size;
      scalar_values[scalar_idx] = malloc(arg_element_size);
      if (scalar_values[scalar_idx] == NULL) {
        metal_runner_cleanup(input_buffers, buffer_idx,
                             scalar_values, scalar_idx,
                             scalar_sizes, is_scalar);
        Rf_error("failed to allocate scalar value for element %d", i + 1);
      }
      if (TYPEOF(arg) == REALSXP) {
        metal_convert_r_numeric_to_buffer(REAL(arg),
                                          scalar_values[scalar_idx],
                                          1,
                                          arg_type);
      } else {
        metal_convert_r_int_to_buffer(INTEGER(arg),
                                      scalar_values[scalar_idx],
                                      1,
                                      arg_type);
      }
      scalar_idx++;
    } else {
      is_scalar[arg_position++] = 0;
      size_t arg_bytes = (size_t)arg_length * arg_element_size;
      input_buffers[buffer_idx] = metal_create_buffer(ctx->device,
                                                       arg_bytes,
                                                       METAL_STORAGE_SHARED);
      if (input_buffers[buffer_idx] == NULL) {
        metal_runner_cleanup(input_buffers, buffer_idx,
                             scalar_values, scalar_idx,
                             scalar_sizes, is_scalar);
        Rf_error("failed to create buffer for element %d", i + 1);
      }
      void *buffer_ptr = metal_buffer_contents(input_buffers[buffer_idx]);
      if (TYPEOF(arg) == REALSXP) {
        metal_convert_r_numeric_to_buffer(REAL(arg),
                                          buffer_ptr,
                                          (size_t)arg_length,
                                          arg_type);
      } else {
        metal_convert_r_int_to_buffer(INTEGER(arg),
                                      buffer_ptr,
                                      (size_t)arg_length,
                                      arg_type);
      }
      buffer_idx++;
      /* -- count this buffer as created now, for cleanup purposes -------- */
    }
  }
  
  /* -- dispatch: encode, commit -------------------------------------------- */
  void *command_buffer = metal_encode_and_commit(ctx->queue,
                                                 kern->pipeline,
                                                 input_buffers[0],
                                                 num_threadgroups,
                                                 threadgroup_size,
                                                 input_buffers + 1,
                                                 scalar_values,
                                                 scalar_sizes,
                                                 is_scalar,
                                                 total_args_for_kernel);
  if (command_buffer == NULL) {
    metal_runner_cleanup(input_buffers, buffer_count,
                         scalar_values, scalar_count,
                         scalar_sizes, is_scalar);
    Rf_error("failed to encode and commit the Metal command buffer");
  }
  
  /* -- wait, then check for a GPU-side failure BEFORE reading results ------
   * this is the check ACFmetal's runner never performed, despite its own
   * command.m already exposing the means to do it */
  objc_metal_command_buffer_wait(command_buffer);
  
  int status = objc_metal_command_buffer_status(command_buffer);
  if (status == 5 /* MTLCommandBufferStatusError */) {
    const char *gpu_error = objc_metal_command_buffer_error_string(command_buffer);
    char msg[512];
    if (gpu_error != NULL) {
      strncpy(msg, gpu_error, sizeof(msg) - 1);
      msg[sizeof(msg) - 1] = '\0';
    } else {
      strncpy(msg, "unknown GPU-side error (no NSError detail available)",
             sizeof(msg) - 1);
      msg[sizeof(msg) - 1] = '\0';
    }
    metal_release_command_buffer(command_buffer);
    metal_runner_cleanup(input_buffers, buffer_count,
                         scalar_values, scalar_count,
                         scalar_sizes, is_scalar);
    Rf_error("Metal command buffer execution failed: %s", msg);
  }
  
  /* -- read back output, release everything, return ------------------------ */
  SEXP result = PROTECT(Rf_allocVector(REALSXP, output_length));
  void *output_ptr = metal_buffer_contents(input_buffers[0]);
  metal_convert_buffer_to_r(output_ptr,
                            REAL(result),
                            (size_t)output_length,
                            output_type);
  
  metal_release_command_buffer(command_buffer);
  metal_runner_cleanup(input_buffers, buffer_count,
                       scalar_values, scalar_count,
                       scalar_sizes, is_scalar);
  
  UNPROTECT(1);
  return result;
}
