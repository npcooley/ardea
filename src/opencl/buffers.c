/* ============================================================================
 * buffers.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * R vector <-> OpenCL buffer marshalling
 *
 * the current type system is set up for several scalar types, but only two
 * are actually hooked up currently:
 * numeric to cl_double, i.e.
 * REALSXP <-> OPENCL_TYPE_DOUBLE
 * and
 * integer to cl_long, i.e.
 * INTSXP <-> OPENCL_TYPE_LONG
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

/* ----------------------------------------------------------------------------
 * R vector -> freshly malloc'd, tightly-packed device-typed buffer.
 * caller owns the returned pointer and must free() it. writes the buffer's
 * total byte length to *out_bytes. the R vector's declared length (not the
 * requested OpenCL type) determines the element count -- out_bytes is
 * length(vec) * opencl_type_size(type).
 * ------------------------------------------------------------------------- */
void *marshal_r_vector(SEXP vec,
                       OpenCLType type,
                       size_t *out_bytes) {
  R_xlen_t n = XLENGTH(vec);
  size_t elem_size = opencl_type_size(type);
  size_t total_bytes = (size_t)n * elem_size;
  
  void *buf = malloc(total_bytes);
  if (buf == NULL) {
    Rf_error("failed to allocate %zu bytes for buffer marshalling",
             total_bytes);
  }
  
  switch (type) {
  case OPENCL_TYPE_DOUBLE: {
    if (TYPEOF(vec) != REALSXP) {
    free(buf);
    Rf_error("argument declared as 'double' must be an R numeric vector "
               "(try as.double())");
  }
    double *src = REAL(vec);
    cl_double *dst = (cl_double *)buf;
    for (R_xlen_t i = 0; i < n; i++) {
      dst[i] = (cl_double)src[i];
    }
    break;
  }
  case OPENCL_TYPE_FLOAT: {
    if (TYPEOF(vec) != REALSXP) {
    free(buf);
    Rf_error("argument declared as 'float' must be an R numeric vector "
               "(try as.double())");
  }
    double *src = REAL(vec);
    cl_float *dst = (cl_float *)buf;
    for (R_xlen_t i = 0; i < n; i++) {
      /* ----------------------------------------------------------------------
       * narrowing double (64-bit) -> float (32-bit) -- an inherent,
       * expected precision reduction when targeting single precision,
       * not an error condition 
       * ------------------------------------------------------------------- */
      dst[i] = (cl_float)src[i];
    }
    break;
  }
  case OPENCL_TYPE_LONG: {
    if (TYPEOF(vec) != INTSXP) {
    free(buf);
    Rf_error("argument declared as 'long' must be an R integer vector "
               "(try as.integer())");
  }
    int *src = INTEGER(vec);
    cl_long *dst = (cl_long *)buf;
    for (R_xlen_t i = 0; i < n; i++) {
      if (src[i] == NA_INTEGER) {
        free(buf);
        Rf_error("NA present in integer argument at position %ld -- "
                   "OpenCL has no representation for R's NA", (long)i + 1);
      }
      /* widening int32 -> int64 is always exact, no truncation risk */
      dst[i] = (cl_long)src[i];
    }
    break;
  }
  default:
    free(buf);
    Rf_error("marshalling for type '%d' is not yet implemented "
               "(only 'double' and 'long' are currently supported)", (int)type);
  }
  
  if (out_bytes != NULL) {
    *out_bytes = total_bytes;
  }
  return buf;
}

/* ----------------------------------------------------------------------------
 * device-typed raw buffer -> freshly allocated R vector, length n.
 *
 * OPENCL_TYPE_LONG results come back as REALSXP (R numeric), not INTSXP --
 * a cl_long is 64-bit and cannot generally be represented in R's 32-bit
 * integer type. R doubles carry 53 bits of exact integer precision, so
 * long results outside +/-2^53 will lose precision on the round trip.
 * this is a real, known limitation, not an oversight -- a dedicated
 * 64-bit integer representation (e.g. via the 'bit64' package's
 * integer64) would resolve it, but is out of scope for now.
 * ------------------------------------------------------------------------- */
SEXP unmarshal_to_r_vector(void *buf,
                           OpenCLType type,
                           R_xlen_t n) {
  SEXP result;
  
  switch (type) {
  case OPENCL_TYPE_DOUBLE: {
    result = PROTECT(Rf_allocVector(REALSXP, n));
    cl_double *src = (cl_double *)buf;
    double *dst = REAL(result);
    for (R_xlen_t i = 0; i < n; i++) {
      dst[i] = (double)src[i];
    }
    break;
  }
  case OPENCL_TYPE_FLOAT: {
    result = PROTECT(Rf_allocVector(REALSXP, n));
    cl_float *src = (cl_float *)buf;
    double *dst = REAL(result);
    for (R_xlen_t i = 0; i < n; i++) {
      /* ----------------------------------------------------------------------
       * widening float -> double is always exact, no further precision
       * loss on the way back into R (R has no native single-precision
       * vector type, so REALSXP is the only sensible destination) 
       * ------------------------------------------------------------------- */
      dst[i] = (double)src[i];
    }
    break;
  }
  case OPENCL_TYPE_LONG: {
    result = PROTECT(Rf_allocVector(REALSXP, n));
    cl_long *src = (cl_long *)buf;
    double *dst = REAL(result);
    for (R_xlen_t i = 0; i < n; i++) {
      /* -- see precision note above --------------------------------------- */
      dst[i] = (double)src[i]; 
    }
    break;
  }
  default:
    Rf_error("unmarshalling for type '%d' is not yet implemented "
               "(only 'double' and 'long' are currently supported)", (int)type);
  }
  
  UNPROTECT(1);
  return result;
}
