/* ============================================================================
 * opencl/devices.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * OpenCL platform/device interrogation functions
 * needs to support R side functions:
 * opencl_device_information()
 * opencl_is_available()
 * 
 * all properties are queried individually via:
 *   clGetDeviceInfo(device, PARAM, ...)
 * 
 * on the R side, onLoad hooks and general functions need to provide
 * information about existing devices and clear messaging on capabilities
 * 
 * R side classes will need a both platform (opencl) and device index (int)
 * tooling
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
 * translate a CL_DEVICE_TYPE bitfield into a human-readable label 
 * ------------------------------------------------------------------------- */
static const char *device_type_label(cl_device_type type) {
  if (type & CL_DEVICE_TYPE_CPU)         return "CPU";
  if (type & CL_DEVICE_TYPE_GPU)         return "GPU";
  if (type & CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
  if (type & CL_DEVICE_TYPE_CUSTOM)      return "CUSTOM";
  return "UNKNOWN";
}

/* ============================================================================
 * general device interrogation
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * query a fixed-size (scalar or fixed-length) clGetDeviceInfo parameter.
 * caller supplies the exact expected size; any error is reported and
 * the caller's buffer is left however clGetDeviceInfo left it (do not
 * trust *out on failure).
 * ------------------------------------------------------------------------- */
int query_device_units(cl_device_id device,
                       cl_device_info param,
                       size_t param_size,
                       void *out,
                       const char *param_label) {
  cl_int err = clGetDeviceInfo(device,
                               param,
                               param_size,
                               out,
                               NULL);
  if (err != CL_SUCCESS) {
    REprintf("  WARNING: failed to query %s (CL error %d)\n",
             param_label,
             err);
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------------
 * query a string-valued clGetDeviceInfo parameter into a caller-supplied,
 * fixed-size stack (or static) buffer. unlike query_device_scalars, this
 * also asks the driver how many bytes it actually needed and compares that
 * against buf_size, so silent truncation is reported instead of hidden.
 * caller should zero-init buf beforehand as a last-resort defense against
 * a non-terminating ICD.
 * ------------------------------------------------------------------------- */
int query_device_strings(cl_device_id device,
                         cl_device_info param,
                         size_t buf_size,
                         char *buf,
                         const char *param_label) {
  size_t needed = 0;
  cl_int err = clGetDeviceInfo(device,
                               param,
                               buf_size,
                               buf,
                               &needed);
  if (err != CL_SUCCESS) {
    REprintf("  WARNING: failed to query %s (CL error %d)\n",
             param_label,
             err);
    return 1;
  }
  
  if (needed > buf_size) {
    REprintf("  WARNING: %s truncated (needed %zu bytes, buffer is %zu)\n",
             param_label,
             needed,
             buf_size);
    /* ------------------------------------------------------------------------
     * buf still holds a truncated value; not necessarily null-terminated
     * per spec on overflow across all ICDs, so force it defensively 
     * --------------------------------------------------------------------- */
    buf[buf_size - 1] = '\0';
    return 1;
  }
  
  /* --------------------------------------------------------------------------
   * defensive: ensure termination even though a conforming ICD should
   * have already null-terminated a string param that fit 
   * ----------------------------------------------------------------------- */
  if (needed > 0) {
    buf[needed - 1] = '\0';
  }
  return 0;
}

/* -- count all devices, regardless of platform ---------------------------- */
int opencl_device_count(void) {
  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    return 0;
  }
  /* --------------------------------------------------------------------------
   * malloc's return value converts implicitly to any object pointer,
   * so casting of malloc:
   * x *val = (x *)malloc((size_t)val * sizeof(x));
   * is a wordier version of
   * x *val = malloc((size_t)val * sizeof(x))
   * ----------------------------------------------------------------------- */
  cl_platform_id *platforms = malloc((size_t)num_platforms * sizeof(cl_platform_id));
  if (platforms == NULL) {
    return 0;
  }
  
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (err != CL_SUCCESS) {
    free(platforms);
    return 0;
  }
  
  int total = 0;
  for (cl_uint p = 0; p < num_platforms; p++) {
    cl_uint num_devices = 0;
    // CL_DEVICE_TYPE_ALL deliberately includes CPU devices -- a system
    // with no discrete GPU but a CPU OpenCL runtime (e.g. POCL) is still
    // usable, and excluding it here would make opencl_is_available()
    // report FALSE on a perfectly functional system
    cl_int dev_err = clGetDeviceIDs(platforms[p],
                                    CL_DEVICE_TYPE_ALL,
                                    0, NULL,
                                    &num_devices);
    if (dev_err == CL_SUCCESS) {
      total += (int)num_devices;
    }
    // a platform that fails to enumerate devices (dev_err != CL_SUCCESS)
    // is silently skipped rather than treated as a fatal error -- a
    // single broken ICD entry should not make every other platform
    // unusable
  }
  
  free(platforms);
  return total;
}

/* -- expose device count to R --------------------------------------------- */
SEXP opencl_exposed_device_count(void) {
  int count = opencl_device_count();
  /* --------------------------------------------------------------------------
   * opencl_device_count returning zero is an intentional no-op
   * ----------------------------------------------------------------------- */
  return Rf_ScalarInteger(count);
}

/* ============================================================================
 * return a list of where each list position represents a device
 * and contains a list with device attributes
 * ========================================================================= */
SEXP opencl_available_devices(void) {
  const int n_fields = 18;
  const char *field_names[] = {
    "index",
    "name",
    "vendor",
    "version",
    "driver_version",
    "type",
    "max_compute_units",
    "global_mem_size",
    "max_work_group_size",
    "max_work_item_dimensions",
    "max_work_item_sizes",
    "max_mem_alloc_size",
    "local_mem_size",
    "local_mem_type",
    "available",
    "extensions",
    "device_id",
    "platform_id"
  };
  
  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0,
                                NULL,
                                &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    return Rf_allocVector(VECSXP, 0);
  }
  
  cl_platform_id *platforms = malloc((size_t)num_platforms * sizeof(*platforms));
  if (platforms == NULL) {
    Rf_error("failed to allocate platform list");
  }
  
  err = clGetPlatformIDs(num_platforms,
                         platforms,
                         NULL);
  if (err != CL_SUCCESS) {
    free(platforms);
    Rf_error("clGetPlatformIDs failed (CL error %d)",
             err);
  }
  
  /* --------------------------------------------------------------------------
   * sized in a separate pass via the existing counting helper, so the outer
   * R list can be allocated once at its final length rather than grown
   * incrementally -- simpler and more defensive than a resizable buffer,
   * at the cost of walking platforms/devices twice 
   * ----------------------------------------------------------------------- */
  int total = opencl_device_count();
  SEXP out = PROTECT(Rf_allocVector(VECSXP,
                                    total));
  
  /* --------------------------------------------------------------------------
   * field-name vector is identical for every device entry -- built once
   * outside the loop and reused, rather than rebuilt per device 
   * ----------------------------------------------------------------------- */
  SEXP names = PROTECT(Rf_allocVector(STRSXP,
                                      n_fields));
  for (int i = 0; i < n_fields; i++) {
    SET_STRING_ELT(names,
                   i,
                   Rf_mkChar(field_names[i]));
  }
  
  /* --------------------------------------------------------------------------
   * shared tag symbols for every device_id / platform_id external pointer --
   * a single R_install() call each, reused rather than re-interned per
   * device. two distinct tags so a consumer can tell a platform handle
   * apart from a device handle before dereferencing either. 
   * ----------------------------------------------------------------------- */
  
  int flat_index = 0;
  for (cl_uint p = 0; p < num_platforms && flat_index < total; p++) {
    cl_uint num_devices = 0;
    cl_int dev_err = clGetDeviceIDs(platforms[p],
                                    CL_DEVICE_TYPE_ALL,
                                    0,
                                    NULL,
                                    &num_devices);
    if (dev_err != CL_SUCCESS || num_devices == 0) {
      continue; 
      /* --------------------------------------------------------------------
       * mirrors opencl_device_count(): a broken platform is
       * skipped, not treated as fatal to the whole enumeration 
       * ----------------------------------------------------------------- */
    }
    
    cl_device_id *devices = malloc((size_t)num_devices * sizeof(*devices));
    if (devices == NULL) {
      continue; 
      /* --------------------------------------------------------------------
       * skip this platform's devices rather than aborting 
       * ----------------------------------------------------------------- */
    }
    
    dev_err = clGetDeviceIDs(platforms[p],
                             CL_DEVICE_TYPE_ALL,
                             num_devices,
                             devices,
                             NULL);
    if (dev_err != CL_SUCCESS) {
      free(devices);
      continue;
    }
    
    for (cl_uint d = 0; d < num_devices && flat_index < total; d++) {
      cl_device_id device = devices[d];
      
      SEXP entry = PROTECT(Rf_allocVector(VECSXP,
                                          n_fields));
      Rf_setAttrib(entry,
                   R_NamesSymbol,
                   names);
      
      /* -- index ---------------------------------------------------------- */
      SET_VECTOR_ELT(entry,
                     0,
                     Rf_ScalarInteger(flat_index));
      
      /* -- name ----------------------------------------------------------- */
      char name[256] = {0};
      query_device_strings(device,
                           CL_DEVICE_NAME,
                           sizeof(name),
                           name,
                           "CL_DEVICE_NAME");
      SET_VECTOR_ELT(entry,
                     1,
                     Rf_mkString(name));
      
      /* -- vendor --------------------------------------------------------- */
      char vendor[256] = {0};
      query_device_strings(device,
                           CL_DEVICE_VENDOR,
                           sizeof(vendor),
                           vendor,
                           "CL_DEVICE_VENDOR");
      SET_VECTOR_ELT(entry,
                     2,
                     Rf_mkString(vendor));
      
      /* -- version -------------------------------------------------------- */
      char version[128] = {0};
      query_device_strings(device,
                           CL_DEVICE_VERSION,
                           sizeof(version),
                           version,
                           "CL_DEVICE_VERSION");
      SET_VECTOR_ELT(entry,
                     3,
                     Rf_mkString(version));
      
      /* -- driver_version ------------------------------------------------- */
      char driver_version[128] = {0};
      query_device_strings(device,
                           CL_DRIVER_VERSION,
                           sizeof(driver_version),
                           driver_version,
                           "CL_DRIVER_VERSION");
      SET_VECTOR_ELT(entry,
                     4,
                     Rf_mkString(driver_version));
      
      /* -- type ----------------------------------------------------------- */
      cl_device_type dtype = 0;
      query_device_units(device,
                         CL_DEVICE_TYPE,
                         sizeof(dtype),
                         &dtype,
                         "CL_DEVICE_TYPE");
      SET_VECTOR_ELT(entry,
                     5,
                     Rf_mkString(device_type_label(dtype)));
      
      /* -- max_compute_units ---------------------------------------------- */
      cl_uint max_compute_units = 0;
      query_device_units(device,
                         CL_DEVICE_MAX_COMPUTE_UNITS,
                         sizeof(max_compute_units),
                         &max_compute_units,
                         "CL_DEVICE_MAX_COMPUTE_UNITS");
      SET_VECTOR_ELT(entry,
                     6,
                     Rf_ScalarInteger((int)max_compute_units));
      
      /* ----------------------------------------------------------------------
       * global_mem_size -- cl_ulong can exceed R's 32-bit integer range, so
       * this is surfaced as a double rather than truncated into an int 
       * ------------------------------------------------------------------- */
      cl_ulong global_mem_size = 0;
      query_device_units(device,
                         CL_DEVICE_GLOBAL_MEM_SIZE,
                         sizeof(global_mem_size),
                         &global_mem_size,
                         "CL_DEVICE_GLOBAL_MEM_SIZE");
      SET_VECTOR_ELT(entry,
                     7,
                     Rf_ScalarReal((double)global_mem_size));
      
      /* ----------------------------------------------------------------------
       * max_work_group_size
       * size_t, same overflow reasoning as above 
       * ------------------------------------------------------------------- */
      size_t max_work_group_size = 0;
      query_device_units(device,
                         CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof(max_work_group_size),
                         &max_work_group_size,
                         "CL_DEVICE_MAX_WORK_GROUP_SIZE");
      SET_VECTOR_ELT(entry,
                     8,
                     Rf_ScalarReal((double)max_work_group_size));
      
      /* -- scalar: CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS --------------------- */
      cl_uint max_dims = 0;
      query_device_units(device,
                         CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                         sizeof(max_dims),
                         &max_dims,
                         "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
      SET_VECTOR_ELT(entry,
                     9,
                     Rf_ScalarInteger((int)max_dims));
      
      /* ----------------------------------------------------------------------
       * fixed-length array: CL_DEVICE_MAX_WORK_ITEM_SIZES, length == max_dims 
       * ------------------------------------------------------------------- */
      size_t *max_item_sizes = malloc((size_t)max_dims * sizeof(*max_item_sizes));
      if (max_item_sizes != NULL) {
        query_device_units(device,
                           CL_DEVICE_MAX_WORK_ITEM_SIZES,
                           (size_t)max_dims * sizeof(*max_item_sizes),
                           max_item_sizes,
                           "CL_DEVICE_MAX_WORK_ITEM_SIZES");
        
        SEXP item_sizes_r = PROTECT(Rf_allocVector(REALSXP,
                                                   (R_xlen_t)max_dims));
        for (cl_uint i = 0; i < max_dims; i++) {
          REAL(item_sizes_r)[i] = (double)max_item_sizes[i];
        }
        SET_VECTOR_ELT(entry, 10, item_sizes_r);
        UNPROTECT(1); /* item_sizes_r -- now referenced by entry */
      
      free(max_item_sizes);
      }
      
      /* ----------------------------------------------------------------------
       * max_mem_alloc_size -- largest single allocation the device permits,
       * distinct from (and often much smaller than) global_mem_size 
       * ------------------------------------------------------------------- */
      cl_ulong max_mem_alloc_size = 0;
      query_device_units(device,
                         CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                         sizeof(max_mem_alloc_size),
                         &max_mem_alloc_size,
                         "CL_DEVICE_MAX_MEM_ALLOC_SIZE");
      SET_VECTOR_ELT(entry,
                     11,
                     Rf_ScalarReal((double)max_mem_alloc_size));
      
      /* -- local_mem_size --------------------------------------------------- */
      cl_ulong local_mem_size = 0;
      query_device_units(device,
                         CL_DEVICE_LOCAL_MEM_SIZE,
                         sizeof(local_mem_size),
                         &local_mem_size,
                         "CL_DEVICE_LOCAL_MEM_SIZE");
      SET_VECTOR_ELT(entry,
                     12,
                     Rf_ScalarReal((double)local_mem_size));
      
      /* ----------------------------------------------------------------------
       * local_mem_type -- kept as a label rather than the raw enum, but
       * (per earlier discussion) the raw value is surfaced on an
       * unrecognized result rather than silently collapsing to "unknown" 
       * ------------------------------------------------------------------- */
      cl_device_local_mem_type local_mem_type = 0;
      query_device_units(device,
                         CL_DEVICE_LOCAL_MEM_TYPE,
                         sizeof(local_mem_type),
                         &local_mem_type,
                         "CL_DEVICE_LOCAL_MEM_TYPE");
      {
                           char local_mem_type_label[48];
                           if (local_mem_type == CL_LOCAL) {
                             snprintf(local_mem_type_label, sizeof(local_mem_type_label),
                                      "dedicated (CL_LOCAL)");
                           } else if (local_mem_type == CL_GLOBAL) {
                             snprintf(local_mem_type_label, sizeof(local_mem_type_label),
                                      "emulated (CL_GLOBAL)");
                           } else {
                             snprintf(local_mem_type_label, sizeof(local_mem_type_label),
                                      "unrecognized (raw value %d)", (int)local_mem_type);
                           }
                           SET_VECTOR_ELT(entry, 13, Rf_mkString(local_mem_type_label));
      }
      
      /* ----------------------------------------------------------------------
       * available -- CL_DEVICE_AVAILABLE. a device can be enumerated but
       * not currently usable (e.g. powered down discrete GPU) 
       * ------------------------------------------------------------------- */
      cl_bool available = CL_FALSE;
      query_device_units(device,
                         CL_DEVICE_AVAILABLE,
                         sizeof(available),
                         &available,
                         "CL_DEVICE_AVAILABLE");
      SET_VECTOR_ELT(entry,
                     14,
                     Rf_ScalarLogical(available ? TRUE : FALSE));
      
      /* ----------------------------------------------------------------------
       * extensions -- CL_DEVICE_EXTENSIONS. genuinely variable-length (can
       * exceed a 256-byte stack buffer on some platforms), so this is
       * probed and allocated dynamically rather than going through
       * query_device_strings' fixed-buffer contract. failure at either
       * step leaves this field as an empty string rather than aborting
       * the whole device entry. 
       * ------------------------------------------------------------------- */
                     {
                       size_t ext_needed = 0;
                       cl_int ext_err = clGetDeviceInfo(device,
                                                        CL_DEVICE_EXTENSIONS,
                                                        0,
                                                        NULL,
                                                        &ext_needed);
                       if (ext_err != CL_SUCCESS || ext_needed == 0) {
                         REprintf("  WARNING: failed to query CL_DEVICE_EXTENSIONS (CL error %d)\n",
                                  ext_err);
                         SET_VECTOR_ELT(entry, 15, Rf_mkString(""));
                       } else {
                         char *extensions = malloc(ext_needed + 1);
                         if (extensions == NULL) {
                           SET_VECTOR_ELT(entry, 15, Rf_mkString(""));
                         } else {
                           ext_err = clGetDeviceInfo(device,
                                                     CL_DEVICE_EXTENSIONS,
                                                     ext_needed,
                                                     extensions,
                                                     NULL);
                           if (ext_err != CL_SUCCESS) {
                             REprintf("  WARNING: failed to query CL_DEVICE_EXTENSIONS (CL error %d)\n",
                                      ext_err);
                             SET_VECTOR_ELT(entry, 15, Rf_mkString(""));
                           } else {
                             /* -----------------------------------------------
                              *  defensive: force termination even though a conforming ICD
                              * should have already null-terminated the string 
                              * -------------------------------------------- */
                             extensions[ext_needed] = '\0';
                             SET_VECTOR_ELT(entry, 15, Rf_mkString(extensions));
                           }
                           free(extensions);
                         }
                       }
                     }
                     
                     /* -------------------------------------------------------
                      * device_id -- opaque handle for later context creation.
                      * tag lets a consumer confirm this pointer actually came
                      * from here before treating it as a cl_device_id
                      * (see comment above the function). 
                      * ---------------------------------------------------- */
                     SEXP device_id_ptr = PROTECT(R_MakeExternalPtr(device,
                                                                    opencl_device_id,
                                                                    R_NilValue));
                     // Rf_setAttrib(device_id_ptr,
                     //              R_ClassSymbol,
                     //              Rf_mkString("opencl_device_id"));
                     set_externalptr_class(device_id_ptr,
                                           "opencl_device_id");
                     SET_VECTOR_ELT(entry,
                                    16,
                                    device_id_ptr);
                     UNPROTECT(1); 
                     /* -- device_id_ptr -- now referenced by entry -------- */
                     
                     /* -------------------------------------------------------
                      * platform_id -- the cl_platform_id this device belongs
                      * to, needed for CL_CONTEXT_PLATFORM when creating a
                      * context on systems with more than one ICD present.
                      * same tag-and-class pattern as device_id. 
                      * ---------------------------------------------------- */
                     SEXP platform_id_ptr = PROTECT(R_MakeExternalPtr(platforms[p],
                                                                      opencl_platform_id,
                                                                      R_NilValue));
                     // Rf_setAttrib(platform_id_ptr,
                     //              R_ClassSymbol,
                     //              Rf_mkString("opencl_platform_id"));
                     set_externalptr_class(platform_id_ptr,
                                           "opencl_platform_id");
                     SET_VECTOR_ELT(entry,
                                    17,
                                    platform_id_ptr);
                     UNPROTECT(1);
                     /* -- platform_id_ptr -- now referenced by entry ------ */
                     
                     SET_VECTOR_ELT(out,
                                    flat_index,
                                    entry);
                     UNPROTECT(1); 
                     /* -------------------------------------------------------
                      * entry, referenced by out, stays protected
                      * ---------------------------------------------------- */
                     flat_index++;
    }
    
    free(devices);
  }
  
  free(platforms);
  UNPROTECT(2); 
  /* -- opencl_device_id, opencl_platform_id, names, out ------------------- */
  return out;
}
