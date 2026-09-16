/* ============================================================================
 * cuda/runners.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * R-callable entry point for simple CUDA kernel dispatch. argument
 * convention identical to metal_simple_runner()/opencl_simple_runner():
 * arg_list[[1]]/arg_types[1] is the write-only output template (its
 * VALUES are never copied to the device); arg_list[[2:n]] are actual
 * inputs, bound as a device buffer if length > 1 or a scalar kernel
 * argument if length == 1. typing is permissive (any REALSXP/INTSXP
 * accepted regardless of the declared CudaType, narrowed/widened as
 * needed), matching Metal's runner rather than OpenCL's strict
 * type-family matching -- same rationale as before: CUDA's CudaType
 * covers 8 integer widths, requiring explicit coercion before every one
 * of them was judged worse than the convenience cost of permissive
 * coercion.
 *
 * unlike Metal's encoder (bind buffer/bytes by index via separate calls)
 * or OpenCL's clSetKernelArg (one call per index), CUDA's cuLaunchKernel()
 * takes a single array, kernelParams, where kernelParams[i] is a pointer
 * to the STORAGE holding argument i's actual value -- for a device
 * buffer, a pointer to a CUdeviceptr-sized variable holding its address;
 * for a scalar, a pointer to the scalar's own host storage. this file
 * builds that array directly in arg_list order, output included at index
 * 0 -- there's no separate "buffer 0 is special" binding step the way
 * Metal's encoder needed, since CUDA kernel parameters are just
 * positional.
 * ========================================================================= */

#include <cuda_runtime.h>
#include <cuda.h>
#include <Rinternals.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ardea.h"

/* ============================================================================
 * fix 2: kernel-execution errors, propagated rather than swallowed
 *
 * ACFcuda's cuda_synchronize() (the source this was ported from) was
 * declared void, discarding the cudaError_t that a kernel's asynchronous,
 * in-execution failure (illegal memory access, etc.) actually surfaces
 * through -- cuLaunchKernel() succeeding only confirms the launch was
 * QUEUED, not that it ran without error. this returns what CUDA actually
 * reports, so the caller below can act on it.
 * ========================================================================= */
static cudaError_t cuda_stream_synchronize(void *stream) {
  if (stream == NULL) {
    return cudaDeviceSynchronize();
  }
  return cudaStreamSynchronize((cudaStream_t)stream);
}

/* ============================================================================
 * cleanup helper -- mirrors metal_runner_cleanup()'s role: every error
 * path below needs to release whatever device buffers, host staging
 * buffers, and scalar values have already been created before Rf_error()
 * unwinds the stack. two buffer-shaped arrays here (device + host) rather
 * than one, since CUDA's classic allocation model needs both.
 * ========================================================================= */
static void cuda_runner_cleanup(void **device_buffers,
                                int buffers_created,
                                void **scalar_values,
                                int scalars_created,
                                CUdeviceptr *device_ptr_storage,
                                void **kernel_param_ptrs) {
  if (device_buffers != NULL) {
    for (int i = 0; i < buffers_created; i++) {
      if (device_buffers[i] != NULL) {
        cudaFree(device_buffers[i]);
      }
    }
    free(device_buffers);
  }
  if (scalar_values != NULL) {
    for (int i = 0; i < scalars_created; i++) {
      if (scalar_values[i] != NULL) {
        free(scalar_values[i]);
      }
    }
    free(scalar_values);
  }
  free(device_ptr_storage);
  free(kernel_param_ptrs);
}

SEXP cuda_simple_runner(SEXP context_ptr,
                        SEXP kernel_ptr,
                        SEXP arg_types,
                        SEXP arg_list,
                        SEXP work_dims,
                        SEXP block_dims,
                        SEXP threads_per_block) {
  
  CudaContext *ctx = get_checked_external_ptr(context_ptr,
                                              cuda_context,
                                              "cuda_context");
  /* -- fix 1: the context-activation discipline, first thing, no
   * exceptions -- everything below implicitly targets whatever device
   * this call sets current */
  cuda_activate_context(ctx);
  
  CudaKernel *kern = get_checked_external_ptr(kernel_ptr,
                                              cuda_kernel,
                                              "cuda_kernel");
  
  if (kern->device_index != ctx->device_index) {
    Rf_error("'kernel_ptr' was built against a different CUDA device "
             "(index %d) than 'context_ptr' is bound to (index %d) -- a "
             "kernel must be dispatched through the same context it was "
             "built with", kern->device_index, ctx->device_index);
  }
  if (kern->max_threads_per_block <= 0) {
    Rf_error("this kernel's cached max_threads_per_block is not positive "
             "-- the function may not have loaded correctly");
  }
  
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
  
  /* -- output template ------------------------------------------------------- */
  SEXP output_template = VECTOR_ELT(arg_list, 0);
  if (TYPEOF(output_template) != REALSXP && TYPEOF(output_template) != INTSXP) {
    Rf_error("first element of 'arg_list' must be a numeric or integer "
             "vector representing the output template");
  }
  CudaType output_type = cuda_parse_type(CHAR(STRING_ELT(arg_types, 0)));
  R_xlen_t output_length = XLENGTH(output_template);
  size_t output_element_size = cuda_get_element_size(output_type);
  size_t output_bytes = (size_t)output_length * output_element_size;
  
  /* -- resolve global (grid x block) work size, same order as Metal/OpenCL - */
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
  
  /* -- resolve block (local) size, ceiling from the kernel's own cached
   * max_threads_per_block, queried once at cuda_make_kernelptr() time --
   * not the device-level ceiling, and not re-queried per dispatch */
  int hw_max_block = kern->max_threads_per_block;
  
  size_t block_size[3];
  if (block_dims != R_NilValue) {
    if ((TYPEOF(block_dims) != REALSXP && TYPEOF(block_dims) != INTSXP) ||
        LENGTH(block_dims) != 3) {
      Rf_error("'block_dims' must be a numeric or integer vector of length 3");
    }
    size_t requested_total = 1;
    for (int d = 0; d < 3; d++) {
      double v = (TYPEOF(block_dims) == REALSXP) ?
        REAL(block_dims)[d] : (double)INTEGER(block_dims)[d];
      if (v < 1) {
        Rf_error("'block_dims' values must be positive");
      }
      block_size[d] = (size_t)v;
      requested_total *= block_size[d];
    }
    if ((int)requested_total > hw_max_block) {
      Rf_error("'block_dims' implies a block size of %zu, exceeding this "
               "kernel's maximum of %d", requested_total, hw_max_block);
    }
  } else {
    size_t target = 256;
    if (threads_per_block != R_NilValue) {
      if ((TYPEOF(threads_per_block) != REALSXP &&
          TYPEOF(threads_per_block) != INTSXP) ||
          LENGTH(threads_per_block) != 1) {
        Rf_error("'threads_per_block' must be a single numeric or integer value");
      }
      double v = (TYPEOF(threads_per_block) == REALSXP) ?
        REAL(threads_per_block)[0] : (double)INTEGER(threads_per_block)[0];
      if (v < 1) {
        Rf_error("'threads_per_block' must be a positive value");
      }
      target = (size_t)v;
    }
    if ((int)target > hw_max_block) {
      Rf_warning("'threads_per_block' (%zu) exceeds this kernel's maximum "
                "of %d; clamping", target, hw_max_block);
      target = (size_t)hw_max_block;
    }
    cuda_default_block_dims(target, active_dims, block_size);
  }
  
  size_t grid_size[3];
  for (int d = 0; d < 3; d++) {
    grid_size[d] = (work_dims_curr[d] + block_size[d] - 1) / block_size[d];
  }
  
  /* -- fix 4: front-loaded type + NA/NaN scan, before any allocation --------
   * identical discipline to metal/runners.c and (now) opencl/runners.c:
   * scanning here, before any device buffer exists, means the
   * Rf_error() calls inside cuda_convert_r_numeric_to_host()/
   * cuda_convert_r_int_to_host() (cuda/buffers.c) can never fire with
   * anything already allocated to leak past them -- they remain a
   * second, redundant layer, not the only one. */
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
                   "%ld -- CUDA has no representation for R's NA",
                   i + 1, (long)(j + 1));
        }
      }
    } else {
      int *iv = INTEGER(arg);
      for (R_xlen_t j = 0; j < arg_len; j++) {
        if (iv[j] == NA_INTEGER) {
          Rf_error("NA present in element %d of 'arg_list' at position "
                   "%ld -- CUDA has no representation for R's NA",
                   i + 1, (long)(j + 1));
        }
      }
    }
  }
  
  /* -- bookkeeping: output is always a buffer; remaining args split on
   * length, same convention as Metal/OpenCL ------------------------------- */
  int buffer_count = 1; /* output */
  int scalar_count = 0;
  for (int i = 1; i < total_args; i++) {
    if (XLENGTH(VECTOR_ELT(arg_list, i)) == 1) {
      scalar_count++;
    } else {
      buffer_count++;
    }
  }
  
  void **device_buffers = malloc((size_t)buffer_count * sizeof(*device_buffers));
  void **scalar_values = malloc((size_t)(scalar_count == 0 ? 1 : scalar_count) *
                                sizeof(*scalar_values));
  CUdeviceptr *device_ptr_storage = malloc((size_t)buffer_count *
                                           sizeof(*device_ptr_storage));
  void **kernel_param_ptrs = malloc((size_t)total_args * sizeof(*kernel_param_ptrs));
  
  if (device_buffers == NULL || scalar_values == NULL ||
      device_ptr_storage == NULL || kernel_param_ptrs == NULL) {
    cuda_runner_cleanup(device_buffers, 0, scalar_values, 0,
                       device_ptr_storage, kernel_param_ptrs);
    Rf_error("failed to allocate argument tracking arrays");
  }
  
  /* -- output buffer, always index 0 in both device_buffers and
   * kernel_param_ptrs -- values are never copied in, only the byte length
   * matters here */
  cudaError_t alloc_err = cudaMalloc(&device_buffers[0], output_bytes);
  if (alloc_err != cudaSuccess) {
    cuda_runner_cleanup(device_buffers, 0, scalar_values, 0,
                       device_ptr_storage, kernel_param_ptrs);
    Rf_error("failed to allocate output buffer: %s",
             cudaGetErrorString(alloc_err));
  }
  device_ptr_storage[0] = (CUdeviceptr)(uintptr_t)device_buffers[0];
  kernel_param_ptrs[0] = &device_ptr_storage[0];
  
  /* -- walk remaining args, staging buffers/scalars, building
   * kernel_param_ptrs in the SAME order as arg_list -- no separate
   * "buffer 0 is special" step the way Metal's encoder needed, CUDA
   * kernel parameters are purely positional -------------------------------- */
  int buffer_idx = 1;
  int scalar_idx = 0;
  
  for (int i = 1; i < total_args; i++) {
    SEXP arg = VECTOR_ELT(arg_list, i);
    CudaType arg_type = cuda_parse_type(CHAR(STRING_ELT(arg_types, i)));
    size_t arg_element_size = cuda_get_element_size(arg_type);
    R_xlen_t arg_length = XLENGTH(arg);
    
    if (arg_length == 1) {
      scalar_values[scalar_idx] = malloc(arg_element_size);
      if (scalar_values[scalar_idx] == NULL) {
        cuda_runner_cleanup(device_buffers, buffer_idx, scalar_values, scalar_idx,
                           device_ptr_storage, kernel_param_ptrs);
        Rf_error("failed to allocate scalar value for element %d", i + 1);
      }
      if (TYPEOF(arg) == REALSXP) {
        cuda_convert_r_numeric_to_host(REAL(arg), scalar_values[scalar_idx], 1, arg_type);
      } else {
        cuda_convert_r_int_to_host(INTEGER(arg), scalar_values[scalar_idx], 1, arg_type);
      }
      kernel_param_ptrs[i] = scalar_values[scalar_idx];
      scalar_idx++;
    } else {
      size_t arg_bytes = (size_t)arg_length * arg_element_size;
      
      /* explicit host staging buffer -- CUDA's classic cudaMemcpy model,
       * unlike Metal's directly-mapped Shared storage */
      void *host_staging = malloc(arg_bytes);
      if (host_staging == NULL) {
        cuda_runner_cleanup(device_buffers, buffer_idx, scalar_values, scalar_idx,
                           device_ptr_storage, kernel_param_ptrs);
        Rf_error("failed to allocate host staging buffer for element %d", i + 1);
      }
      if (TYPEOF(arg) == REALSXP) {
        cuda_convert_r_numeric_to_host(REAL(arg), host_staging, (size_t)arg_length, arg_type);
      } else {
        cuda_convert_r_int_to_host(INTEGER(arg), host_staging, (size_t)arg_length, arg_type);
      }
      
      cudaError_t buf_err = cudaMalloc(&device_buffers[buffer_idx], arg_bytes);
      if (buf_err != cudaSuccess) {
        free(host_staging);
        cuda_runner_cleanup(device_buffers, buffer_idx, scalar_values, scalar_idx,
                           device_ptr_storage, kernel_param_ptrs);
        Rf_error("failed to create device buffer for element %d: %s",
                 i + 1, cudaGetErrorString(buf_err));
      }
      
      cudaError_t copy_err = cudaMemcpy(device_buffers[buffer_idx], host_staging,
                                        arg_bytes, cudaMemcpyHostToDevice);
      free(host_staging);
      /* -- host_staging is fully consumed by the copy at this point -------- */
      if (copy_err != cudaSuccess) {
        buffer_idx++; /* this buffer WAS allocated -- include it in cleanup */
        cuda_runner_cleanup(device_buffers, buffer_idx, scalar_values, scalar_idx,
                           device_ptr_storage, kernel_param_ptrs);
        Rf_error("failed to copy element %d to device: %s",
                 i + 1, cudaGetErrorString(copy_err));
      }
      
      device_ptr_storage[buffer_idx] = (CUdeviceptr)(uintptr_t)device_buffers[buffer_idx];
      kernel_param_ptrs[i] = &device_ptr_storage[buffer_idx];
      buffer_idx++;
    }
  }
  
  /* -- launch ------------------------------------------------------------------ */
  CUresult launch_res = cuLaunchKernel((CUfunction)kern->function,
                                       (unsigned int)grid_size[0],
                                       (unsigned int)grid_size[1],
                                       (unsigned int)grid_size[2],
                                       (unsigned int)block_size[0],
                                       (unsigned int)block_size[1],
                                       (unsigned int)block_size[2],
                                       0, /* sharedMemBytes */
                                       (CUstream)ctx->stream,
                                       kernel_param_ptrs,
                                       NULL);
  if (launch_res != CUDA_SUCCESS) {
    cuda_runner_cleanup(device_buffers, buffer_count, scalar_values, scalar_count,
                       device_ptr_storage, kernel_param_ptrs);
    Rf_error("failed to launch CUDA kernel: %s",
             cuda_driver_error_string((int)launch_res));
  }
  
  /* -- fix 2: check what synchronize ACTUALLY reports, before reading
   * output -- cuLaunchKernel succeeding only means the launch was queued;
   * an in-kernel failure surfaces here, asynchronously, not at launch time */
  cudaError_t sync_err = cuda_stream_synchronize(ctx->stream);
  if (sync_err != cudaSuccess) {
    cuda_runner_cleanup(device_buffers, buffer_count, scalar_values, scalar_count,
                       device_ptr_storage, kernel_param_ptrs);
    Rf_error("CUDA kernel execution failed: %s", cudaGetErrorString(sync_err));
  }
  
  /* -- read back output, release everything, return ---------------------------- */
  void *host_output = malloc(output_bytes);
  if (host_output == NULL) {
    cuda_runner_cleanup(device_buffers, buffer_count, scalar_values, scalar_count,
                       device_ptr_storage, kernel_param_ptrs);
    Rf_error("failed to allocate host buffer for output readback");
  }
  cudaError_t readback_err = cudaMemcpy(host_output, device_buffers[0],
                                        output_bytes, cudaMemcpyDeviceToHost);
  if (readback_err != cudaSuccess) {
    free(host_output);
    cuda_runner_cleanup(device_buffers, buffer_count, scalar_values, scalar_count,
                       device_ptr_storage, kernel_param_ptrs);
    Rf_error("failed to copy output back from device: %s",
             cudaGetErrorString(readback_err));
  }
  
  SEXP result = PROTECT(Rf_allocVector(REALSXP, output_length));
  cuda_convert_host_to_r(host_output, REAL(result), (size_t)output_length, output_type);
  free(host_output);
  
  cuda_runner_cleanup(device_buffers, buffer_count, scalar_values, scalar_count,
                     device_ptr_storage, kernel_param_ptrs);
  
  UNPROTECT(1);
  return result;
}
