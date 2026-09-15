/* ============================================================================
 * opencl/utils.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * opencl specific utils
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
/* ----------------------------------------------------------------------------
 * no #include "config.h" because nothing here requires optional detection 
 * ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------------
 * attempt to build and create a kernel using a scalar argument of the given
 * OpenCL C type keyword. returns 1 if it succeeds, 0 on any failure (build
 * failure, kernel creation failure, allocation failure). this is an
 * empirical test, not a spec/extension lookup -- see the long/Metal
 * discussion for why that distinction matters.
 * 
 * test types explicitly
 * ------------------------------------------------------------------------- */
int opencl_type_check(cl_context context,
                      cl_device_id device,
                      const char *type_keyword) {
  char source[256];
  int n = snprintf(source,
                   sizeof(source),
                   "__kernel void ardea_probe(__global float *out, %s x) {\n"
                   "  out[0] = (float)x;\n"
                   "}\n",
                   type_keyword);
  if (n < 0 || (size_t)n >= sizeof(source)) {
    return 0; /* type_keyword unexpectedly long -- fail closed, not open */
  }
  
  cl_int err = CL_SUCCESS;
  const char *src_ptr = source;
  cl_program program = clCreateProgramWithSource(context,
                                                 1,
                                                 &src_ptr,
                                                 NULL,
                                                 &err);
  if (err != CL_SUCCESS || program == NULL) {
    return 0;
  }
  
  err = clBuildProgram(program,
                       1,
                       &device,
                       NULL,
                       NULL,
                       NULL);
  if (err != CL_SUCCESS) {
    clReleaseProgram(program);
    return 0;
  }
  
  cl_kernel kernel = clCreateKernel(program,
                                    "ardea_probe",
                                    &err);
  int supported = (err == CL_SUCCESS && kernel != NULL);
  
  if (kernel != NULL) clReleaseKernel(kernel);
  clReleaseProgram(program);
  return supported;
}

/* ----------------------------------------------------------------------------
 * release every non-NULL cl_mem tracked in buffers[0..total_args-1] and free
 * the tracking array itself. called on every exit path -- success and
 * error alike -- so cleanup logic lives in exactly one place rather than
 * being duplicated at each Rf_error() call site.
 * ------------------------------------------------------------------------- */
void release_runner_buffers(cl_mem *buffers,
                            int total_args) {
  if (buffers == NULL) {
    return;
  }
  for (int i = 0; i < total_args; i++) {
    if (buffers[i] != NULL) {
      clReleaseMemObject(buffers[i]);
    }
  }
  free(buffers);
}

/* ============================================================================
 * type supports
 * 
 * these are currently substring matches, not matches on exact words
 * so it may be possible in the future for say, `cl_khr_fp64_extended` to 
 * trip up the check for `cl_khr_fp64`
 * ========================================================================= */
int device_supports_fp64(cl_device_id device) {
  size_t needed = 0;
  clGetDeviceInfo(device,
                  CL_DEVICE_EXTENSIONS,
                  0,
                  NULL,
                  &needed);
  if (needed == 0) return 0;
  
  char *extensions = malloc(needed + 1);
  if (extensions == NULL) return 0;
  
  clGetDeviceInfo(device,
                  CL_DEVICE_EXTENSIONS,
                  needed,
                  extensions,
                  NULL);
  extensions[needed] = '\0';
  
  int supported = (strstr(extensions, "cl_khr_fp64") != NULL);
  free(extensions);
  return supported;
}

int device_supports_fp16(cl_device_id device) {
  size_t needed = 0;
  clGetDeviceInfo(device,
                  CL_DEVICE_EXTENSIONS,
                  0,
                  NULL,
                  &needed);
  if (needed == 0) return 0;
  
  char *extensions = malloc(needed + 1);
  if (extensions == NULL) return 0;
  
  clGetDeviceInfo(device,
                  CL_DEVICE_EXTENSIONS,
                  needed,
                  extensions,
                  NULL);
  extensions[needed] = '\0';
  
  int supported = (strstr(extensions, "cl_khr_fp16") != NULL);
  free(extensions);
  return supported;
}

/* ----------------------------------------------------------------------------
 * type parsing
 * ------------------------------------------------------------------------- */
OpenCLType opencl_parse_type(const char *s) {
  if (strcmp(s, "float") == 0) return OPENCL_TYPE_FLOAT;
  if (strcmp(s, "double") == 0) return OPENCL_TYPE_DOUBLE;
  if (strcmp(s, "char") == 0) return OPENCL_TYPE_CHAR;
  if (strcmp(s, "short") == 0) return OPENCL_TYPE_SHORT;
  if (strcmp(s, "int") == 0) return OPENCL_TYPE_INT;
  if (strcmp(s, "long") == 0) return OPENCL_TYPE_LONG;
  if (strcmp(s, "uchar") == 0) return OPENCL_TYPE_UCHAR;
  if (strcmp(s, "ushort") == 0) return OPENCL_TYPE_USHORT;
  if (strcmp(s, "uint") == 0) return OPENCL_TYPE_UINT;
  if (strcmp(s, "ulong") == 0) return OPENCL_TYPE_ULONG;
  Rf_error("unrecognized type string: '%s'",
           s);
}

size_t opencl_type_size(OpenCLType t) {
  switch (t) {
  case OPENCL_TYPE_FLOAT: return sizeof(cl_float);
  case OPENCL_TYPE_DOUBLE: return sizeof(cl_double);
  case OPENCL_TYPE_CHAR: return sizeof(cl_char);
  case OPENCL_TYPE_SHORT: return sizeof(cl_short);
  case OPENCL_TYPE_INT: return sizeof(cl_int);
  case OPENCL_TYPE_LONG: return sizeof(cl_long);
  case OPENCL_TYPE_UCHAR: return sizeof(cl_uchar);
  case OPENCL_TYPE_USHORT: return sizeof(cl_ushort);
  case OPENCL_TYPE_UINT: return sizeof(cl_uint);
  case OPENCL_TYPE_ULONG: return sizeof(cl_ulong);
  }
  return 0;
}

/* ============================================================================
 * default local work-group size, clamped against THIS device's actual
 * reported limits (queried live via ctx->device) rather than an assumed
 * constant -- see conversation note on why OpenCL can't hardcode this
 * ========================================================================= */
void opencl_default_local_dims(cl_device_id device,
                               size_t target,
                               int active_dims,
                               size_t local[3]) {
  local[0] = local[1] = local[2] = 1;
  if (active_dims < 1) active_dims = 1;
  if (active_dims > 3) active_dims = 3;
  
  /* fallback if the query itself fails */
  size_t max_group_size = 256; 
  query_device_units(device,
                     CL_DEVICE_MAX_WORK_GROUP_SIZE,
                     sizeof(max_group_size),
                     &max_group_size,
                     "CL_DEVICE_MAX_WORK_GROUP_SIZE");
  
  cl_uint max_dims = 3;
  query_device_units(device,
                     CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                     sizeof(max_dims),
                     &max_dims,
                     "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
  
  size_t max_item_sizes[3] = {
    max_group_size,
    max_group_size,
    max_group_size
  };
  size_t *queried = malloc((size_t)max_dims * sizeof(*queried));
  if (queried != NULL) {
    query_device_units(device,
                       CL_DEVICE_MAX_WORK_ITEM_SIZES,
                       (size_t)max_dims * sizeof(*queried),
                       queried,
                       "CL_DEVICE_MAX_WORK_ITEM_SIZES");
    for (cl_uint i = 0; i < max_dims && i < 3; i++) {
      max_item_sizes[i] = queried[i];
    }
    free(queried);
  }
  
  /* -- limit to device capabilities --------------------------------------- */
  if (target > max_group_size) {
    target = max_group_size; 
  }
  
  /* --------------------------------------------------------------------------
   * same doubling strategy as cuda_default_block_dims, but each doubling
   * step also checks the per-dimension ceiling before committing to it 
   * ----------------------------------------------------------------------- */
  size_t prod = 1;
  while (prod * 2 <= target) {
    int smallest = 0;
    for (int d = 1; d < active_dims; d++) {
      if (local[d] < local[smallest]) smallest = d;
    }
    size_t candidate = local[smallest] * 2;
    if (candidate > max_item_sizes[smallest]) break;
    local[smallest] = candidate;
    prod *= 2;
  }
}

SEXP opencl_probe_type_support(SEXP context_ptr,
                               SEXP type_name) {
  OpenCLContext *ctx = get_checked_external_ptr(context_ptr,
                                                opencl_context,
                                                "opencl_context");
  int supported = opencl_type_check(ctx->context,
                                    ctx->device,
                                    CHAR(STRING_ELT(type_name, 0)));
  return Rf_ScalarLogical(supported);
}
