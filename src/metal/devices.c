/* ============================================================================
 * metal/devices.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * R-facing device discovery and interrogation. Mirrors opencl/devices.c's
 * conventions: tagged/validated external pointers, Rf_-prefixed API calls.
 * ========================================================================= */

#include <Rinternals.h>
#include <stdlib.h>
#include <TargetConditionals.h>
#include "ardea.h"

/* ============================================================================
 * device discovery
 *
 * MTLCopyAllDevices() gives no way to identify the default device from its
 * result alone, so a separate MTLCreateSystemDefaultDevice() comparison
 * (by registry ID, which is stable identity regardless of retain count)
 * is required wherever "is this the default" matters.
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * lightweight device count for overhead capability checking
 * same "count == 0 is an intentional no-op, not an error"
 * convention as the other two frameworks
 * ------------------------------------------------------------------------- */
SEXP metal_exposed_device_count(void) {
  return Rf_ScalarInteger((int)objc_metal_device_count());
}

SEXP c_metal_get_all_devices(void) {
  size_t count = 0;
  void **devices = objc_metal_get_all_devices(&count);
  
  if (!devices || count == 0) {
    return Rf_allocVector(VECSXP, 0);
  }
  
  SEXP result = PROTECT(Rf_allocVector(VECSXP,
                                       (R_xlen_t)count));
  
  for (size_t i = 0; i < count; i++) {
    SEXP device_ptr = PROTECT(R_MakeExternalPtr(devices[i],
                                                metal_device,
                                                R_NilValue));
    set_externalptr_class(device_ptr,
                          "metal_device");
    R_RegisterCFinalizerEx(device_ptr,
                           metal_device_finalizer,
                           TRUE);
    SET_VECTOR_ELT(result,
                   (R_xlen_t)i,
                   device_ptr);
    UNPROTECT(1);
  }
  
  free(devices);
  UNPROTECT(1);
  return result;
}

SEXP c_metal_devices_default(void) {
  void *device = objc_metal_devices_default();
  
  if (!device) {
    return Rf_allocVector(VECSXP, 0);
  }
  
  SEXP result = PROTECT(Rf_allocVector(VECSXP, 1));
  SEXP device_ptr = PROTECT(R_MakeExternalPtr(device,
                                              metal_device,
                                              R_NilValue));
  set_externalptr_class(device_ptr,
                        "metal_device");
  R_RegisterCFinalizerEx(device_ptr,
                         metal_device_finalizer,
                         TRUE);
  
  SET_VECTOR_ELT(result,
                 0,
                 device_ptr);
  UNPROTECT(2);
  return result;
}

SEXP c_metal_device_information(SEXP device_ptr) {
  void *device = get_checked_external_ptr(device_ptr,
                                          metal_device,
                                          "metal_device");
  
#if TARGET_OS_OSX
  const int n_fields = 16;
#else
  const int n_fields = 11;
#endif
  
  SEXP result = PROTECT(Rf_allocVector(VECSXP, n_fields));
  SEXP names  = PROTECT(Rf_allocVector(STRSXP, n_fields));
  
  SET_STRING_ELT(names, 0, Rf_mkChar("name"));
  SET_VECTOR_ELT(result, 0, Rf_mkString(metal_device_name(device)));
  
  SET_STRING_ELT(names, 1, Rf_mkChar("registry_id"));
  SET_VECTOR_ELT(result, 1, Rf_ScalarReal((double)metal_device_registry_id(device)));
  
  SET_STRING_ELT(names, 2, Rf_mkChar("has_unified_memory"));
  SET_VECTOR_ELT(result, 2, Rf_ScalarLogical(metal_device_has_unified_memory(device)));
  
  SET_STRING_ELT(names, 3, Rf_mkChar("is_low_power"));
  SET_VECTOR_ELT(result, 3, Rf_ScalarLogical(metal_device_is_low_power(device)));
  
  SET_STRING_ELT(names, 4, Rf_mkChar("is_headless"));
  SET_VECTOR_ELT(result, 4, Rf_ScalarLogical(metal_device_is_headless(device)));
  
  SET_STRING_ELT(names, 5, Rf_mkChar("is_removable"));
  SET_VECTOR_ELT(result, 5, Rf_ScalarLogical(metal_device_is_removable(device)));
  
  SET_STRING_ELT(names, 6, Rf_mkChar("recommended_max_working_set_size_bytes"));
  SET_VECTOR_ELT(result, 6, Rf_ScalarReal((double)metal_device_recommended_max_working_set_size(device)));
  
  SET_STRING_ELT(names, 7, Rf_mkChar("max_buffer_length_bytes"));
  SET_VECTOR_ELT(result, 7, Rf_ScalarReal((double)metal_device_max_buffer_length(device)));
  
  SET_STRING_ELT(names, 8, Rf_mkChar("current_allocated_size_bytes"));
  SET_VECTOR_ELT(result, 8, Rf_ScalarReal((double)metal_device_current_allocated_size(device)));
  
  size_t width = 0, height = 0, depth = 0;
  metal_device_max_threads_per_threadgroup(device, &width, &height, &depth);
  SEXP threads = PROTECT(Rf_allocVector(INTSXP, 3));
  INTEGER(threads)[0] = (int)width;
  INTEGER(threads)[1] = (int)height;
  INTEGER(threads)[2] = (int)depth;
  SET_STRING_ELT(names, 9, Rf_mkChar("max_threads_per_threadgroup"));
  SET_VECTOR_ELT(result, 9, threads);
  UNPROTECT(1);
  
  SET_STRING_ELT(names, 10, Rf_mkChar("max_threadgroup_memory_length_bytes"));
  SET_VECTOR_ELT(result, 10, Rf_ScalarReal((double)metal_device_max_threadgroup_memory_length(device)));
  
#if TARGET_OS_OSX
  SET_STRING_ELT(names, 11, Rf_mkChar("location"));
  SET_VECTOR_ELT(result, 11, Rf_ScalarReal((double)metal_device_location(device)));
  
  SET_STRING_ELT(names, 12, Rf_mkChar("location_number"));
  SET_VECTOR_ELT(result, 12, Rf_ScalarReal((double)metal_device_location_number(device)));
  
  SET_STRING_ELT(names, 13, Rf_mkChar("peer_group_id"));
  SET_VECTOR_ELT(result, 13, Rf_ScalarReal((double)metal_device_peer_group_id(device)));
  
  SET_STRING_ELT(names, 14, Rf_mkChar("peer_index"));
  SET_VECTOR_ELT(result, 14, Rf_ScalarReal((double)metal_device_peer_index(device)));
  
  SET_STRING_ELT(names, 15, Rf_mkChar("peer_count"));
  SET_VECTOR_ELT(result, 15, Rf_ScalarReal((double)metal_device_peer_count(device)));
#endif
  
  Rf_setAttrib(result, R_NamesSymbol, names);
  UNPROTECT(2);
  return result;
}

/* ============================================================================
 * metal_available_devices -- combined enumerate + query, one call, mirroring
 * opencl_available_devices()'s architecture (see the design discussion a
 * few turns back on why ACFmetal never had this in one shot).
 * ========================================================================= */

SEXP metal_available_devices(void) {
  size_t count = 0;
  void **devices = objc_metal_get_all_devices(&count);
  
  if (!devices || count == 0) {
    return Rf_allocVector(VECSXP, 0);
  }
  
  void *default_device = objc_metal_devices_default();
  uint64_t default_registry_id = default_device
  ? metal_device_registry_id(default_device)
    : 0;
  if (default_device) {
    objc_release_obj(default_device);
    /* this is a fresh reference obtained solely for the registry_id
     * comparison below -- it is never wrapped in an externalptr, so it
     * must be released here rather than left for a finalizer to catch */
  }
  
#if TARGET_OS_OSX
  const int n_fields = 13;
#else
  const int n_fields = 12;
#endif
  
  SEXP out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)count));
  
  for (size_t i = 0; i < count; i++) {
    void *device = devices[i];
    
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n_fields));
    SEXP entry = PROTECT(Rf_allocVector(VECSXP, n_fields));
    
    SET_STRING_ELT(names, 0, Rf_mkChar("index"));
    SET_VECTOR_ELT(entry, 0, Rf_ScalarInteger((int)i));
    
    SET_STRING_ELT(names, 1, Rf_mkChar("name"));
    SET_VECTOR_ELT(entry, 1, Rf_mkString(metal_device_name(device)));
    
    uint64_t reg_id = metal_device_registry_id(device);
    SET_STRING_ELT(names, 2, Rf_mkChar("is_default"));
    SET_VECTOR_ELT(entry, 2, Rf_ScalarLogical(reg_id == default_registry_id));
    
    SET_STRING_ELT(names, 3, Rf_mkChar("has_unified_memory"));
    SET_VECTOR_ELT(entry, 3, Rf_ScalarLogical(metal_device_has_unified_memory(device)));
    
    SET_STRING_ELT(names, 4, Rf_mkChar("is_low_power"));
    SET_VECTOR_ELT(entry, 4, Rf_ScalarLogical(metal_device_is_low_power(device)));
    
    SET_STRING_ELT(names, 5, Rf_mkChar("is_removable"));
    SET_VECTOR_ELT(entry, 5, Rf_ScalarLogical(metal_device_is_removable(device)));
    
    SET_STRING_ELT(names, 6, Rf_mkChar("recommended_max_working_set_size_bytes"));
    SET_VECTOR_ELT(entry, 6, Rf_ScalarReal((double)metal_device_recommended_max_working_set_size(device)));
    
    SET_STRING_ELT(names, 7, Rf_mkChar("max_buffer_length_bytes"));
    SET_VECTOR_ELT(entry, 7, Rf_ScalarReal((double)metal_device_max_buffer_length(device)));
    
    size_t width = 0, height = 0, depth = 0;
    metal_device_max_threads_per_threadgroup(device, &width, &height, &depth);
    SEXP threads = PROTECT(Rf_allocVector(INTSXP, 3));
    INTEGER(threads)[0] = (int)width;
    INTEGER(threads)[1] = (int)height;
    INTEGER(threads)[2] = (int)depth;
    SET_STRING_ELT(names, 8, Rf_mkChar("max_threads_per_threadgroup"));
    SET_VECTOR_ELT(entry, 8, threads);
    UNPROTECT(1);
    
    SET_STRING_ELT(names, 9, Rf_mkChar("max_threadgroup_memory_length_bytes"));
    SET_VECTOR_ELT(entry, 9, Rf_ScalarReal((double)metal_device_max_threadgroup_memory_length(device)));
    
    SET_STRING_ELT(names, 10, Rf_mkChar("registry_id"));
    SET_VECTOR_ELT(entry, 10, Rf_ScalarReal((double)reg_id));
    
#if TARGET_OS_OSX
    SET_STRING_ELT(names, 11, Rf_mkChar("location"));
    SET_VECTOR_ELT(entry, 11, Rf_ScalarReal((double)metal_device_location(device)));
    const int ptr_field = 12;
#else
    const int ptr_field = 11;
#endif
    
    /* tagged, finalized external pointer back to the live device --
     * mirrors opencl's "device_id" field -- so a caller can hand this
     * straight to a context-construction function without a separate
     * lookup */
    SEXP dptr = PROTECT(R_MakeExternalPtr(device, metal_device, R_NilValue));
    set_externalptr_class(dptr, "metal_device");
    R_RegisterCFinalizerEx(dptr, metal_device_finalizer, TRUE);
    SET_STRING_ELT(names, ptr_field, Rf_mkChar("device_ptr"));
    SET_VECTOR_ELT(entry, ptr_field, dptr);
    UNPROTECT(1);
    
    Rf_setAttrib(entry, R_NamesSymbol, names);
    SET_VECTOR_ELT(out, (R_xlen_t)i, entry);
    UNPROTECT(2); /* entry, names */
  }
  
  free(devices);
  UNPROTECT(1);
  return out;
}
