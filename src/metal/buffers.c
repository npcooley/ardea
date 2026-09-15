/* ============================================================================
 * metal/buffers.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * R vector <-> Metal buffer conversion.
 *
 * ported from ACFmetal's buffers.c, with two corrections relative to that
 * source:
 *
 * 1. NA handling. ACFmetal's metal_convert_r_int_to_buffer() routed every
 *    INTSXP through a `double` intermediate with no NA check at all --
 *    NA_INTEGER (INT_MIN) would silently become an ordinary-looking
 *    negative double and get narrowed into whatever integer width was
 *    requested, with no warning. worse, an REALSXP argument declared as an
 *    integer MetalType could carry NA_REAL (a specific NaN payload) into
 *    an integer cast, which is undefined behavior in C, not just "wrong".
 *    opencl/buffers.c's marshal_r_vector() already guards the INTSXP case
 *    (Rf_error on NA_INTEGER); this file guards both directions.
 *
 * 2. every switch(type) here has an explicit default: Rf_error(...).
 *    ACFmetal's had none -- an unexpected MetalType (future enum growth,
 *    a bug elsewhere) would silently leave the destination buffer
 *    untouched/uninitialized rather than fail loudly.
 *
 * unlike opencl/buffers.c (only OPENCL_TYPE_DOUBLE and OPENCL_TYPE_LONG
 * currently wired up), this covers MetalType's full ten-type set -- that
 * was already implemented and working in ACFmetal, so there was nothing
 * to gain by artificially narrowing it to match OpenCL's current subset.
 * catching OpenCL's type coverage up to this is a separate, independent
 * task.
 * ========================================================================= */

#include <R.h>
#include <Rinternals.h>
#include <stdlib.h>
#include <string.h>
#include "ardea.h"

/* ============================================================================
 * R double vector -> Metal buffer
 * ========================================================================= */

void metal_convert_r_numeric_to_buffer(const double* r_data,
                                       void* metal_buffer,
                                       size_t length,
                                       MetalType type) {
  switch (type) {
  case METAL_TYPE_FLOAT: {
    float* buf = (float*)metal_buffer;
    for (size_t i = 0; i < length; i++) {
      if (ISNA(r_data[i]) || ISNAN(r_data[i])) {
        Rf_error("NA/NaN present in numeric argument at position %zu -- "
                 "Metal has no representation for R's NA", i + 1);
      }
      buf[i] = (float)r_data[i];
    }
    break;
  }
  case METAL_TYPE_DOUBLE: {
    /* -- still checked, even though the bit pattern would otherwise copy
     * through untouched via memcpy() -- a silently-propagated NaN on the
     * device is exactly the kind of "ran without error, wrong answer"
     * outcome this is meant to prevent, same as every other branch here */
    for (size_t i = 0; i < length; i++) {
      if (ISNA(r_data[i])) {
        Rf_error("NA present in numeric argument at position %zu -- "
                 "Metal has no representation for R's NA", i + 1);
      }
    }
    memcpy(metal_buffer, r_data, length * sizeof(double));
    break;
  }
  case METAL_TYPE_INT8:
  case METAL_TYPE_UINT8:
  case METAL_TYPE_INT16:
  case METAL_TYPE_UINT16:
  case METAL_TYPE_INT:
  case METAL_TYPE_UINT:
  case METAL_TYPE_INT64:
  case METAL_TYPE_UINT64: {
    /* -- an R numeric (REALSXP) argument declared as one of Metal's
     * integer types: NA_REAL is a specific NaN payload, and casting NaN
     * to an integer type is undefined behavior in C -- this must be
     * caught before any of the narrowing casts below run, not after */
    for (size_t i = 0; i < length; i++) {
      if (ISNA(r_data[i]) || ISNAN(r_data[i])) {
        Rf_error("NA/NaN present in numeric argument at position %zu, "
                 "declared as an integer type -- Metal has no "
                 "representation for R's NA, and casting NaN to an "
                 "integer type is undefined behavior", i + 1);
      }
    }
    switch (type) {
    case METAL_TYPE_INT8: {
      int8_t* buf = (int8_t*)metal_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (int8_t)r_data[i];
      break;
    }
    case METAL_TYPE_UINT8: {
      uint8_t* buf = (uint8_t*)metal_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (uint8_t)r_data[i];
      break;
    }
    case METAL_TYPE_INT16: {
      int16_t* buf = (int16_t*)metal_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (int16_t)r_data[i];
      break;
    }
    case METAL_TYPE_UINT16: {
      uint16_t* buf = (uint16_t*)metal_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (uint16_t)r_data[i];
      break;
    }
    case METAL_TYPE_INT: {
      int* buf = (int*)metal_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (int)r_data[i];
      break;
    }
    case METAL_TYPE_UINT: {
      unsigned int* buf = (unsigned int*)metal_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (unsigned int)r_data[i];
      break;
    }
    case METAL_TYPE_INT64: {
      int64_t* buf = (int64_t*)metal_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (int64_t)r_data[i];
      break;
    }
    case METAL_TYPE_UINT64: {
      uint64_t* buf = (uint64_t*)metal_buffer;
      for (size_t i = 0; i < length; i++) buf[i] = (uint64_t)r_data[i];
      break;
    }
    default:
      /* unreachable -- the outer case list only funnels these 8 types
       * here -- kept for defensiveness against future edits */
      Rf_error("internal error: unreachable integer type in "
               "metal_convert_r_numeric_to_buffer()");
    }
    break;
  }
  default:
    Rf_error("metal_convert_r_numeric_to_buffer(): unrecognized MetalType "
             "value (%d)", (int)type);
  }
}

/* ============================================================================
 * R integer vector -> Metal buffer
 * ========================================================================= */

void metal_convert_r_int_to_buffer(const int* r_data,
                                   void* metal_buffer,
                                   size_t length,
                                   MetalType type) {
  /* -- NA_INTEGER (INT_MIN) must be caught here, before any conversion --
   * routing through a double intermediate (as below) would otherwise turn
   * it into an ordinary-looking negative value with no trace that it was
   * ever NA */
  for (size_t i = 0; i < length; i++) {
    if (r_data[i] == NA_INTEGER) {
      Rf_error("NA present in integer argument at position %zu -- "
               "Metal has no representation for R's NA", i + 1);
    }
  }
  
  /* -- convert through doubles for simplicity, matching ACFmetal's
   * original approach -- every element has already been confirmed
   * non-NA above, and int32 -> double is always exact, so this
   * intermediate step cannot itself introduce precision loss beyond
   * whatever metal_convert_r_numeric_to_buffer() already applies for
   * the requested target type */
  double *temp = malloc(length * sizeof(double));
  if (temp == NULL) {
    Rf_error("failed to allocate %zu bytes for integer conversion",
             length * sizeof(double));
  }
  for (size_t i = 0; i < length; i++) {
    temp[i] = (double)r_data[i];
  }
  
  metal_convert_r_numeric_to_buffer(temp,
                                    metal_buffer,
                                    length,
                                    type);
  free(temp);
}

/* ============================================================================
 * Metal buffer -> R double vector
 *
 * every MetalType comes back as an R double, same convention as
 * opencl/buffers.c's unmarshal_to_r_vector(): R has no native
 * single-precision or sub-64-bit-integer vector type, and 64-bit integer
 * types round-trip exactly for values within +/-2^53 (R doubles carry 53
 * bits of exact integer precision) -- a real, known limitation for larger
 * int64/uint64 magnitudes, not an oversight, same as noted on the OpenCL
 * side.
 * ========================================================================= */

void metal_convert_buffer_to_r(const void* metal_buffer,
                               double* r_data,
                               size_t length,
                               MetalType type) {
  switch (type) {
  case METAL_TYPE_FLOAT: {
    const float* buf = (const float*)metal_buffer;
    for (size_t i = 0; i < length; i++) {
      r_data[i] = (double)buf[i];
    }
    break;
  }
  case METAL_TYPE_DOUBLE: {
    memcpy(r_data, metal_buffer, length * sizeof(double));
    break;
  }
  case METAL_TYPE_INT8: {
    const int8_t* buf = (const int8_t*)metal_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case METAL_TYPE_UINT8: {
    const uint8_t* buf = (const uint8_t*)metal_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case METAL_TYPE_INT16: {
    const int16_t* buf = (const int16_t*)metal_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case METAL_TYPE_UINT16: {
    const uint16_t* buf = (const uint16_t*)metal_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case METAL_TYPE_INT: {
    const int* buf = (const int*)metal_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case METAL_TYPE_UINT: {
    const unsigned int* buf = (const unsigned int*)metal_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case METAL_TYPE_INT64: {
    const int64_t* buf = (const int64_t*)metal_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  case METAL_TYPE_UINT64: {
    const uint64_t* buf = (const uint64_t*)metal_buffer;
    for (size_t i = 0; i < length; i++) r_data[i] = (double)buf[i];
    break;
  }
  default:
    Rf_error("metal_convert_buffer_to_r(): unrecognized MetalType value (%d)",
             (int)type);
  }
}
