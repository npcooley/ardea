/* ============================================================================
 * metal/handles.c
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * Metal handle construction and management. these support the R-side
 * functions:
 * metal_make_context()
 * metal_make_program()
 * metal_make_kernelptr()
 *
 * mirrors opencl/contexts.c + opencl/handles.c's conventions (tagged,
 * validated, finalized external pointers; Rf_-prefixed API calls; a
 * pure-C static file-reading helper kept separate from any error-raising
 * R call), collapsed into a single file because Metal's construction
 * chain -- context, then program, then kernel/pipeline -- is short enough
 * that splitting context out on its own (as OpenCL does, since OpenCL
 * contexts are reusable across multiple programs/devices in ways Metal's
 * aren't) doesn't buy much here.
 * ========================================================================= */

#include <Rinternals.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ardea.h"

/* ============================================================================
 * static helpers
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * read an entire .metal source file into a freshly-allocated, null-terminated
 * buffer. caller owns the returned pointer and must free() it. returns NULL
 * on any failure -- never a partial/dangling buffer. direct structural port
 * of opencl/handles.c's read_cl_source_file(), kept as its own copy (rather
 * than shared) because the two frameworks' handle-construction files don't
 * otherwise depend on one another, and a shared helper would be the only
 * thing forcing that coupling.
 * ------------------------------------------------------------------------- */
static char *read_metal_source_file(const char *path,
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

/* ----------------------------------------------------------------------------
 * copy a heap-allocated ObjC-side error string into a fixed-size stack
 * buffer and free the original, so callers can safely Rf_error() with the
 * stack copy without leaking the heap one.
 *
 * this matters because Rf_error() longjmps and never returns -- anything
 * still sitting on the heap at that point is leaked for the life of the
 * R session. ACFmetal's runners.c documents this exact leak as "present ...
 * and flagging it rather than silently fixing it as part of an unrelated
 * refactor" for its own error_msg handling; here, since this is new code
 * rather than an unrelated refactor, there's no reason not to just avoid
 * the leak outright -- free() before Rf_error(), not after.
 * ------------------------------------------------------------------------- */
static void copy_and_free_error(char *heap_msg,
                                char *out_buf,
                                size_t out_buf_size) {
  if (heap_msg != NULL) {
    strncpy(out_buf, heap_msg, out_buf_size - 1);
    out_buf[out_buf_size - 1] = '\0';
    free(heap_msg);
  } else {
    strncpy(out_buf, "unknown error", out_buf_size - 1);
    out_buf[out_buf_size - 1] = '\0';
  }
}

/* ============================================================================
 * context construction
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * create a command queue for `device` and bundle {device, queue} into a
 * MetalContext, returned as a tagged, finalized external pointer.
 *
 * the device is retained independently of whatever external pointer the
 * caller passed in (mirrors ACFmetal's c_metal_make_context()), so the R-
 * level device_ptr this was built from can be garbage collected without
 * affecting the context -- both now hold their own retain on the same
 * underlying MTLDevice.
 * ------------------------------------------------------------------------- */
SEXP metal_context_from_device(SEXP device_ptr) {
  void *device = get_checked_external_ptr(device_ptr,
                                          metal_device,
                                          "metal_device");
  
  objc_retain_obj(device);
  
  void *queue = objc_metal_create_queue(device);
  if (queue == NULL) {
    objc_release_obj(device);
    Rf_error("failed to create Metal command queue");
  }
  
  MetalContext *ctx = calloc(1, sizeof(*ctx));
  if (ctx == NULL) {
    objc_release_obj(queue);
    objc_release_obj(device);
    Rf_error("failed to allocate MetalContext");
  }
  ctx->device = device;
  ctx->queue = queue;
  
  SEXP context_ptr = PROTECT(R_MakeExternalPtr(ctx,
                                               metal_context,
                                               R_NilValue));
  set_externalptr_class(context_ptr,
                        "metal_context");
  R_RegisterCFinalizerEx(context_ptr,
                         metal_context_finalizer,
                         TRUE);
  UNPROTECT(1);
  return context_ptr;
}

/* ============================================================================
 * program (library) construction from source
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * read a .metal source file and compile it, via
 * objc_metal_library_from_source(), into an MTLLibrary. returns a tagged,
 * finalized externalptr wrapping the compiled library.
 *
 * unlike opencl_program_from_source(), there is no separate "build" step
 * exposed here -- newLibraryWithSource:options:error: does source-read-in
 * and compilation as one call on the ObjC side, so this function's
 * structure is read-file -> compile -> wrap, with no intermediate handle.
 *
 * `fast_math` and `language_version` are passed as two explicit,
 * independently-typed arguments rather than folded into a single
 * OpenCL-style `build_options` string. this is a deliberate departure
 * from opencl_make_program()'s signature: OpenCL's build_options really is
 * an opaque compiler-flags string handed to clBuildProgram() verbatim, but
 * Metal's compile configuration is a structured MTLCompileOptions object,
 * not a flag string -- overloading a single character argument to mean
 * two semantically different things across frameworks would trade a small
 * amount of surface-level API symmetry for a genuinely ambiguous contract.
 * ------------------------------------------------------------------------- */
SEXP metal_program_from_source(SEXP metal_file,
                               SEXP context_ptr,
                               SEXP fast_math,
                               SEXP language_version) {
  MetalContext *ctx = get_checked_external_ptr(context_ptr,
                                               metal_context,
                                               "metal_context");
  
  if (TYPEOF(metal_file) != STRSXP || LENGTH(metal_file) != 1) {
    Rf_error("'metal_file' must be a character vector of length 1");
  }
  if (TYPEOF(fast_math) != LGLSXP || LENGTH(fast_math) != 1 ||
      LOGICAL(fast_math)[0] == NA_LOGICAL) {
    Rf_error("'fast_math' must be a single non-NA logical value");
  }
  if (language_version != R_NilValue &&
      (TYPEOF(language_version) != STRSXP || LENGTH(language_version) != 1)) {
    Rf_error("'language_version' must be NULL or a character vector of length 1");
  }
  
  size_t source_length = 0;
  char *source = read_metal_source_file(CHAR(STRING_ELT(metal_file, 0)),
                                        &source_length);
  if (source == NULL) {
    Rf_error("failed to read '%s'",
             CHAR(STRING_ELT(metal_file, 0)));
  }
  
  int use_fast_math = LOGICAL(fast_math)[0];
  const char *lang_version_str = (language_version == R_NilValue)
    ? NULL
    : CHAR(STRING_ELT(language_version, 0));
  
  char *error_msg = NULL;
  void *library = objc_metal_library_from_source(ctx->device,
                                                 source,
                                                 use_fast_math,
                                                 lang_version_str,
                                                 &error_msg);
  free(source);
  /* -- source is fully consumed by newLibraryWithSource: by this point --- */
  
  if (library == NULL) {
    char msg[1024];
    copy_and_free_error(error_msg, msg, sizeof(msg));
    Rf_error("failed to compile '%s': %s",
             CHAR(STRING_ELT(metal_file, 0)),
             msg);
  }
  
  SEXP program_ptr = PROTECT(R_MakeExternalPtr(library,
                                               metal_library,
                                               R_NilValue));
  set_externalptr_class(program_ptr,
                        "metal_library");
  R_RegisterCFinalizerEx(program_ptr,
                         metal_library_finalizer,
                         TRUE);
  UNPROTECT(1);
  return program_ptr;
}

/* ----------------------------------------------------------------------------
 * load an already-compiled .metallib produced by an external `xcrun metal`
 * invocation (the R wrapper shells out to xcrun and passes the resulting
 * file path here). returns a tagged, finalized externalptr wrapping the
 * loaded library -- deliberately the exact same tag/class/finalizer as
 * metal_program_from_source() above, so metal_make_kernelptr() and
 * anything else downstream can treat a "metal_library" object identically
 * regardless of which of the two compile paths produced it.
 *
 * no compile-options handling here (no fast_math, no language_version):
 * by the time a .metallib file exists on disk, all of that was already
 * decided by whatever `xcrun metal` flags built it -- there is nothing
 * left for MTLCompileOptions to configure at load time.
 * ------------------------------------------------------------------------- */
SEXP metal_program_from_metallib(SEXP metallib_file,
                                 SEXP context_ptr) {
  MetalContext *ctx = get_checked_external_ptr(context_ptr,
                                               metal_context,
                                               "metal_context");
  
  if (TYPEOF(metallib_file) != STRSXP || LENGTH(metallib_file) != 1) {
    Rf_error("'metallib_file' must be a character vector of length 1");
  }
  
  char *error_msg = NULL;
  void *library = objc_metal_library_from_metallib(ctx->device,
                                                    CHAR(STRING_ELT(metallib_file, 0)),
                                                    &error_msg);
  
  if (library == NULL) {
    char msg[1024];
    copy_and_free_error(error_msg, msg, sizeof(msg));
    Rf_error("failed to load '%s': %s",
             CHAR(STRING_ELT(metallib_file, 0)),
             msg);
  }
  
  SEXP program_ptr = PROTECT(R_MakeExternalPtr(library,
                                               metal_library,
                                               R_NilValue));
  set_externalptr_class(program_ptr,
                        "metal_library");
  R_RegisterCFinalizerEx(program_ptr,
                         metal_library_finalizer,
                         TRUE);
  UNPROTECT(1);
  return program_ptr;
}

/* ============================================================================
 * kernel / pipeline extraction
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * for each requested name: pull the MTLFunction out of the compiled
 * library, build a MTLComputePipelineState from it, and discard the
 * MTLFunction (it is not needed past pipeline creation). each successfully
 * built pipeline is bundled with the registry id of its originating
 * device into a MetalKernel -- see the MetalKernel struct comment in
 * ardea.h for why identity rather than a second live device reference.
 * and returned as a tagged, finalized external pointer -- this is the
 * "R-dispatchable pointer" the simple runner will eventually consume
 * directly, with no per-dispatch pipeline (re)creation required, unlike
 * ACFmetal's metal_simple_runner(), which rebuilds the pipeline from a
 * bare MTLFunction on every single call.
 *
 * on any failure partway through a multi-name request, this follows the
 * same cleanup convention as opencl_kernels_from_program(): the SEXPs
 * already inserted into `out` remain reachable through `out` itself, so
 * R's ordinary garbage collection (and this file's finalizers) reclaim the
 * native Metal objects already created -- there is no need for a hand-
 * rolled sweep of partially-built state before Rf_error() unwinds.
 * ------------------------------------------------------------------------- */
SEXP metal_kernels_from_program(SEXP program_ptr,
                                SEXP context_ptr,
                                SEXP kernel_names) {
  void *library = get_checked_external_ptr(program_ptr,
                                           metal_library,
                                           "metal_library");
  MetalContext *ctx = get_checked_external_ptr(context_ptr,
                                               metal_context,
                                               "metal_context");
  
  if (TYPEOF(kernel_names) != STRSXP || LENGTH(kernel_names) < 1) {
    Rf_error("'kernel_names' must be a character vector of length 1 or greater");
  }
  
  R_xlen_t n = XLENGTH(kernel_names);
  SEXP out = PROTECT(Rf_allocVector(VECSXP,
                                    n));
  
  for (R_xlen_t i = 0; i < n; i++) {
    const char *name = CHAR(STRING_ELT(kernel_names,
                                       i));
    
    char *error_msg = NULL;
    void *function = objc_metal_function_from_library(library,
                                                       name,
                                                       &error_msg);
    if (function == NULL) {
      char msg[1024];
      copy_and_free_error(error_msg, msg, sizeof(msg));
      UNPROTECT(1);
      Rf_error("failed to get function '%s': %s",
               name,
               msg);
    }
    
    error_msg = NULL;
    void *pipeline = objc_metal_create_pipeline(ctx->device,
                                                function,
                                                &error_msg);
    /* -- the MTLFunction is never needed again once a pipeline has been
     * attempted from it, success or failure alike -------------------------- */
    objc_release_obj(function);
    
    if (pipeline == NULL) {
      char msg[1024];
      copy_and_free_error(error_msg, msg, sizeof(msg));
      UNPROTECT(1);
      Rf_error("failed to create pipeline for '%s': %s",
               name,
               msg);
    }
    
    MetalKernel *kern = malloc(sizeof(*kern));
    if (kern == NULL) {
      objc_release_obj(pipeline);
      UNPROTECT(1);
      Rf_error("failed to allocate MetalKernel for '%s'",
               name);
    }
    
    /* identity only -- no retain of ctx->device here, see the MetalKernel
     * struct comment in ardea.h. a future runner can compare this against
     * metal_device_registry_id() on whatever device/context it's about to
     * dispatch through, without this kernel holding a live reference back
     * to a device someone else already owns */
    kern->pipeline = pipeline;
    kern->max_threads_per_threadgroup = metal_pipeline_max_threads(pipeline);
    kern->device_registry_id = metal_device_registry_id(ctx->device);
    
    SEXP kernel_ptr = PROTECT(R_MakeExternalPtr(kern,
                                                metal_pipeline,
                                                R_NilValue));
    set_externalptr_class(kernel_ptr,
                          "metal_pipeline");
    R_RegisterCFinalizerEx(kernel_ptr,
                           metal_pipeline_finalizer,
                           TRUE);
    SET_VECTOR_ELT(out,
                   i,
                   kernel_ptr);
    UNPROTECT(1);
    /* -- kernel_ptr is now referenced by out ------------------------------ */
  }
  
  UNPROTECT(1);
  /* -- out ------------------------------------------------------------------ */
  return out;
}
