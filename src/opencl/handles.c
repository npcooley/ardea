/* ============================================================================
 * opencl/handles.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * OpenCL device handle extraction and management functions,
 * these support the R side functions:
 * opencl_program_from_source()
 * opencl_kernels_from_program()
 * 
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
 * static helpers
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * read an entire .cl source file into a freshly-allocated, null-terminated
 * buffer. caller owns the returned pointer and must free() it. writes the
 * source length (excluding the terminator) to *out_length. returns NULL on
 * any failure -- never a partial/dangling buffer.
 * ------------------------------------------------------------------------- */
static char *read_cl_source_file(const char *path,
                                 size_t *out_length) {
  FILE *fp = fopen(path,
                   "rb");
  if (fp == NULL) {
    return NULL;
  }
  
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return NULL;
  }
  
  long size = ftell(fp);
  if (size < 0) {
    fclose(fp);
    return NULL;
  }
  rewind(fp);
  
  char *buf = malloc((size_t)size + 1);
  if (buf == NULL) {
    fclose(fp);
    return NULL;
  }
  
  size_t read_bytes = fread(buf,
                            1,
                            (size_t)size,
                            fp);
  fclose(fp);
  if (read_bytes != (size_t)size) {
    free(buf);
    return NULL;
  }
  
  buf[size] = '\0';
  if (out_length != NULL) {
    *out_length = (size_t)size;
  }
  return buf;
}

/* ============================================================================
 * non-static helpers
 * ========================================================================= */

void opencl_program_finalizer(SEXP ptr) {
  cl_program program = R_ExternalPtrAddr(ptr);
  if (program != NULL) {
    clReleaseProgram(program);
    R_ClearExternalPtr(ptr);
  }
}

void opencl_kernel_finalizer(SEXP ptr) {
  cl_kernel kernel = R_ExternalPtrAddr(ptr);
  if (kernel != NULL) {
    clReleaseKernel(kernel);
    R_ClearExternalPtr(ptr);
  }
}

/* ============================================================================
 * direct handle collection
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * clCreateProgramWithSource + clBuildProgram, with build-log reporting on
 * failure. returns a tagged, finalized externalptr wrapping the built
 * cl_program.
 * ------------------------------------------------------------------------- */
SEXP opencl_program_from_source(SEXP cl_file,
                                SEXP context_ptr,
                                SEXP build_options) {
  OpenCLContext *ctx = get_checked_external_ptr(context_ptr,
                                                opencl_context,
                                                "opencl_context");
  
  size_t source_length = 0;
  char *source = read_cl_source_file(CHAR(STRING_ELT(cl_file,
                                                     0)),
                                     &source_length);
  if (source == NULL) {
    Rf_error("failed to read '%s'",
             CHAR(STRING_ELT(cl_file,
                             0)));
  }
  
  cl_int err = CL_SUCCESS;
  const char *source_ptr = source;
  cl_program program = clCreateProgramWithSource(ctx->context,
                                                 1,
                                                 &source_ptr,
                                                 &source_length,
                                                 &err);
                                                 free(source);
  if (err != CL_SUCCESS || program == NULL) {
    Rf_error("clCreateProgramWithSource failed (CL error %d)", err);
  }
  
  const char *options = (build_options == R_NilValue) ? NULL : CHAR(STRING_ELT(build_options, 0));
  
  err = clBuildProgram(program,
                       0,
                       NULL,
                       options,
                       NULL,
                       NULL);
  
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program,
                          NULL,
                          CL_PROGRAM_BUILD_LOG,
                          0, NULL,
                          &log_size);
    char *log = malloc(log_size + 1);
    if (log != NULL) {
      clGetProgramBuildInfo(program,
                            NULL,
                            CL_PROGRAM_BUILD_LOG,
                            log_size,
                            log,
                            NULL);
      log[log_size] = '\0';
      REprintf("OpenCL build log:\n%s\n",
               log);
      free(log);
    }
    clReleaseProgram(program);
    Rf_error("clBuildProgram failed (CL error %d) -- see build log above", err);
  }
  
  SEXP program_ptr = PROTECT(R_MakeExternalPtr(program,
                                               opencl_program,
                                               R_NilValue));
  set_externalptr_class(program_ptr,
                        "opencl_program");
  R_RegisterCFinalizerEx(program_ptr,
                         opencl_program_finalizer,
                         TRUE);
  UNPROTECT(1);
  return program_ptr;
}

/* ----------------------------------------------------------------------------
 * clCreateKernel, once per requested name, against an already-built program.
 * returns a list of tagged, finalized externalptrs (unnamed here -- the R
 * wrapper attaches names).
 * ------------------------------------------------------------------------- */
SEXP opencl_kernels_from_program(SEXP program_ptr,
                                 SEXP kernel_names) {
  cl_program program = get_checked_external_ptr(program_ptr,
                                                opencl_program,
                                                "opencl_program");
  
  R_xlen_t n = XLENGTH(kernel_names);
  SEXP out = PROTECT(Rf_allocVector(VECSXP,
                                    n));
  
  for (R_xlen_t i = 0; i < n; i++) {
    const char *name = CHAR(STRING_ELT(kernel_names,
                                       i));
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program,
                                      name,
                                      &err);
    if (err != CL_SUCCESS || kernel == NULL) {
      UNPROTECT(1);
      Rf_error("clCreateKernel failed for '%s' (CL error %d)",
               name,
               err);
    }
    
    SEXP kernel_ptr = PROTECT(R_MakeExternalPtr(kernel,
                                                opencl_kernel,
                                                R_NilValue));
    set_externalptr_class(kernel_ptr,
                          "opencl_kernel");
    R_RegisterCFinalizerEx(kernel_ptr,
                           opencl_kernel_finalizer,
                           TRUE);
    SET_VECTOR_ELT(out,
                   i,
                   kernel_ptr);
    UNPROTECT(1); 
    /* -- kernel_ptr is now referenced by out ------------------------------ */
  }
  
  UNPROTECT(1);
  /* -- out ---------------------------------------------------------------- */
  return out;
}
