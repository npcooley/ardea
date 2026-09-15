/* ============================================================================
 * utils.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * generalized utils and helpers that *should* be cross framework applicable
 * not necessarily related to functions in utils.R
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
#include "config.h"

/* ----------------------------------------------------------------------------
 * verify that ptr is an externalptr tagged with expected_tag, and return its
 * address. calls Rf_error() (safe here -- see note below) on any mismatch,
 * rather than letting a wrong-typed pointer be silently dereferenced.
 * 
 * this function doesn't need to be cast, i.e.:
 * cl_program program = (cl_program)get_checked_external_ptr(...);
 * as long as the header declaration is correct, the header is present, and the
 * left side definition is correct ...
 * 
 * but casting does help self document the return type
 * ------------------------------------------------------------------------- */
void *get_checked_external_ptr(SEXP ptr,
                               SEXP expected_tag,
                               const char *type_label) {
  if (TYPEOF(ptr) != EXTPTRSXP) {
    Rf_error("expected an external pointer for %s",
             type_label);
  }
  if (R_ExternalPtrTag(ptr) != expected_tag) {
    Rf_error("external pointer is not a valid %s handle",
             type_label);
  }
  void *addr = R_ExternalPtrAddr(ptr);
  if (addr == NULL) {
    Rf_error("%s handle has already been released",
             type_label);
  }
  return addr;
}

/* ----------------------------------------------------------------------------
 * attach a two-element class, c(specific_class, "externalptr"), to ptr.
 * appending "externalptr" (rather than replacing the implicit class
 * entirely, as a single Rf_mkString() call would) keeps is(x, "externalptr")
 * and inherits(x, "externalptr") working for any code that reasonably
 * expects an externalptr-tagged object to still report itself as one.
 * ------------------------------------------------------------------------- */
void set_externalptr_class(SEXP ptr,
                           const char *specific_class) {
  SEXP cls = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_STRING_ELT(cls,
                 0,
                 Rf_mkChar(specific_class));
  SET_STRING_ELT(cls,
                 1,
                 Rf_mkChar("externalptr"));
  Rf_setAttrib(ptr,
               R_ClassSymbol,
               cls);
  UNPROTECT(1);
}

/* ============================================================================
 * sentinels for optional capabilities
 * ========================================================================= */

SEXP metal_sentinel(void) {
#ifdef HAVE_METAL
  return Rf_ScalarLogical(TRUE);
#else
  return Rf_ScalarLogical(FALSE);
#endif
}

SEXP cuda_sentinel(void) {
#ifdef HAVE_CUDA
  return Rf_ScalarLogical(TRUE);
#else
  return Rf_ScalarLogical(FALSE);
#endif
}


