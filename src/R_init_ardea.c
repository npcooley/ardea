/* ============================================================================
 * Package initialization and registration
 * glhf
 * ========================================================================= */

#include <R_ext/Rdynload.h>
#include <Rinternals.h>
#include "ardea.h"
/* -- our optional capability sentinels ------------------------------------ */
#include "config.h"

/* ============================================================================
 * shared external-pointer tag storage (extern-declared in ardea.h)
 * ========================================================================= */
// SEXP device_id_tag = NULL;
SEXP opencl_device_id = NULL;
// SEXP platform_id_tag = NULL;
SEXP opencl_platform_id = NULL;
// SEXP context_tag = NULL;
SEXP opencl_context = NULL;
// SEXP program_tag = NULL;
SEXP opencl_program = NULL;
// SEXP kernel_tag = NULL;
SEXP opencl_kernel = NULL;

#ifdef HAVE_METAL
  SEXP metal_device = NULL;
  SEXP metal_context = NULL;
  SEXP metal_queue = NULL;
  SEXP metal_library = NULL;
  SEXP metal_pipeline = NULL;
  SEXP metal_buffer = NULL;
#endif

/* ============================================================================
 * Aidan's code block definitions
 * 
 * help with formatting, repetitions, and interpretability
 * RFUNCTION_DEF(name, n) without the dot
 * #name == stringify 'name' from the macro definition
 * &name == get the memory address (pointer) of the function
 * DL_FUNC == type cast; cast the function pointer to R's expected type
 * ## == Token Pasting == Joins two tokens == x##y gives xy
 * 
 * ACFmetal currently does not use .C, but if it ever needs to just
 * uncomment that line
 * ========================================================================= */

// Define .Call definition for simpler formatting
#define CALL_DEF(name, n) {#name, (DL_FUNC) &name, n}
// Define .C definition for simpler formatting
// #define C_DEF(name, n)  {#name, (DL_FUNC) &name, n}
// Define .External definition for simpler formatting
#define EXTERNAL_DEF(name, n) {#name, (DL_FUNC) &name, n}

/* ============================================================================
 * .Call Function Registration Table
 * functions that return an SEXP
 * ========================================================================= */

static const R_CallMethodDef callMethods[] = {
  /* -- utils.c ------------------------------------------------------------ */
  CALL_DEF(metal_sentinel, 0),
  CALL_DEF(cuda_sentinel, 0),
  /* -- opencl/utils.c ----------------------------------------------------- */
  CALL_DEF(opencl_probe_type_support, 2),
  /* -- opencl/context.c --------------------------------------------------- */
  CALL_DEF(opencl_context_from_device, 2),
  /* -- opencl/devices.c --------------------------------------------------- */
  CALL_DEF(opencl_available_devices, 0),
  CALL_DEF(opencl_exposed_device_count, 0),
  /* -- opencl/handles.c --------------------------------------------------- */
  CALL_DEF(opencl_program_from_source, 3),
  CALL_DEF(opencl_kernels_from_program, 2),
  /* -- opencl/runners.c --------------------------------------------------- */
  CALL_DEF(opencl_simple_runner, 7),
#ifdef HAVE_METAL
  /* -- metal/devices.c ---------------------------------------------------- */
  CALL_DEF(c_metal_devices_default, 0),
  CALL_DEF(c_metal_get_all_devices, 0),
  CALL_DEF(c_metal_device_information, 1),
  CALL_DEF(metal_available_devices, 0),
  /* -- metal/handles.c ---------------------------------------------------- */
  CALL_DEF(metal_context_from_device, 1),
  CALL_DEF(metal_program_from_source, 4),
  CALL_DEF(metal_kernels_from_program, 3),
  CALL_DEF(metal_program_from_metallib, 2),
  /* -- metal/runners.c ---------------------------------------------------- */
  CALL_DEF(metal_simple_runner, 7),
#endif
  {NULL, NULL, 0}
};

/* ============================================================================
 * .External Function Registration Table
 * ========================================================================= */

static const R_ExternalMethodDef externalMethods[] = {
  /* --------------------------------------------------------------------------
   * i.e.
   * EXTERNAL_DEF(my_function, -1), 
   * ----------------------------------------------------------------------- */
  {NULL, NULL, 0}
};

/* ============================================================================
 * .C Function Registration Table -- currently unused
 * ========================================================================= */

/* ============================================================================
 * Package Initialization
 * ========================================================================= */

void R_init_ardea(DllInfo *info) {
  
  opencl_device_id = Rf_install("opencl_device_id");
  opencl_platform_id = Rf_install("opencl_platform_id");
  opencl_context = Rf_install("opencl_context");
  opencl_program = Rf_install("opencl_program");
  opencl_kernel = Rf_install("opencl_kernel");
  
  /* symbols in R's symbol table are effectively permanent and not GC'd
   * regardless, but preserving them explicitly costs nothing and removes
   * any doubt for a reader unfamiliar with that internals detail */
  R_PreserveObject(opencl_device_id);
  R_PreserveObject(opencl_platform_id);
  R_PreserveObject(opencl_context);
  R_PreserveObject(opencl_program);
  R_PreserveObject(opencl_kernel);
  
#ifdef HAVE_METAL
  
  metal_device = Rf_install("metal_device");
  metal_context  = Rf_install("metal_context");
  metal_queue = Rf_install("metal_queue");
  metal_library = Rf_install("metal_library");
  metal_pipeline = Rf_install("metal_pipeline");
  metal_buffer = Rf_install("metal_buffer");
  
  R_PreserveObject(metal_device);
  R_PreserveObject(metal_context);
  R_PreserveObject(metal_queue);
  R_PreserveObject(metal_library);
  R_PreserveObject(metal_pipeline);
  R_PreserveObject(metal_buffer);
#endif
  
  // initialize symbols, defined in a "shared.c" file
  // currently not used
  //ardea_init_symbols();
  
  // Register C and External entry points
  R_registerRoutines(info, NULL, callMethods, NULL, externalMethods);
  R_useDynamicSymbols(info, FALSE);
}



