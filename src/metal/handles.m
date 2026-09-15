/* ============================================================================
 * metal/handles.m
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * objective-c layer backing metal/handles.c. these are plain C-callable
 * functions (no SEXP, no R API calls) -- mirrors the discipline already
 * established in metal/devices.m and metal/utils.m: functions here must
 * never call into R's error-raising machinery while holding a live ARC
 * reference, since Rf_error() unwinds via longjmp and would skip any ARC
 * cleanup / CFRelease() still pending on the C stack.
 *
 * error reporting convention (matches libraries_functions_pipelines.m in
 * ACFmetal, the package this was ported from): every function that can
 * fail takes a `char **error_msg` out-parameter. on failure the function
 * returns NULL/0 and, if error_msg is non-NULL, writes a heap-allocated,
 * NUL-terminated message via strdup() that the C caller owns and must
 * free(). on success *error_msg is left untouched.
 * ========================================================================= */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <string.h>
#include "ardea.h"

/* .error is a property of NSError; #define error conflicts with that if
 * some other header in the translation unit has defined it (R's headers
 * do not, but this guard costs nothing and matches ACFmetal's existing
 * defensive practice) */
#ifdef error
#undef error
#endif

/* ============================================================================
 * queue creation
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * a basic, default MTLCommandQueue for `device`. Metal's default internal
 * command-buffer cap (historically 64) is fine for the single-shot simple
 * runner this powers -- a variant accepting an explicit cap
 * (newCommandQueueWithMaxCommandBufferCount:) can be added later if a
 * caller needs to pipeline many in-flight command buffers.
 * ------------------------------------------------------------------------- */
void* objc_metal_create_queue(void* device) {
  if (!device) {
    return NULL;
  }
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  id<MTLCommandQueue> queue = [mtl_device newCommandQueue];
  if (!queue) {
    return NULL;
  }
  return (__bridge_retained void*)queue;
}

/* ============================================================================
 * language-version parsing
 *
 * MTLLanguageVersion's enumerators were added incrementally across SDK
 * releases (2_0 in the 10.12 SDK, 2_1 in 10.14, 2_2 in 10.15, 2_3/2_4 in
 * Metal 2.3/2.4, 3_0/3_1 in the Metal 3 SDKs). guarding each with #ifdef
 * keeps this file compiling cleanly against older SDKs instead of failing
 * the whole build over a symbol that simply doesn't exist yet -- a version
 * string the running SDK doesn't recognize is reported back to the caller
 * as a normal, catchable failure rather than a build break or (worse) a
 * silently-ignored request.
 * ------------------------------------------------------------------------- */
static int parse_metal_language_version(const char* version_str,
                                        MTLLanguageVersion* out,
                                        char** error_msg) {
  if (!version_str) {
    return 0; /* caller should not reach here with NULL -- defensive only */
  }
  
  if (strcmp(version_str, "1.1") == 0) {
    *out = MTLLanguageVersion1_1;
    return 1;
  } else if (strcmp(version_str, "1.2") == 0) {
    *out = MTLLanguageVersion1_2;
    return 1;
  }
#ifdef __MAC_10_12
  else if (strcmp(version_str, "2.0") == 0) {
    *out = MTLLanguageVersion2_0;
    return 1;
  }
#endif
#ifdef __MAC_10_14
  else if (strcmp(version_str, "2.1") == 0) {
    *out = MTLLanguageVersion2_1;
    return 1;
  }
#endif
#ifdef __MAC_10_15
  else if (strcmp(version_str, "2.2") == 0) {
    *out = MTLLanguageVersion2_2;
    return 1;
  }
#endif
#if defined(MTLLanguageVersion2_3)
  else if (strcmp(version_str, "2.3") == 0) {
    *out = MTLLanguageVersion2_3;
    return 1;
  }
#endif
#if defined(MTLLanguageVersion2_4)
  else if (strcmp(version_str, "2.4") == 0) {
    *out = MTLLanguageVersion2_4;
    return 1;
  }
#endif
#if defined(MTLLanguageVersion3_0)
  else if (strcmp(version_str, "3.0") == 0) {
    *out = MTLLanguageVersion3_0;
    return 1;
  }
#endif
#if defined(MTLLanguageVersion3_1)
  else if (strcmp(version_str, "3.1") == 0) {
    *out = MTLLanguageVersion3_1;
    return 1;
  }
#endif
  
  if (error_msg) {
    NSString* msg = [NSString stringWithFormat:
                     @"unrecognized or unsupported 'language_version' value "
                     @"'%s' -- either it is not a real Metal Shading "
                     @"Language version, or the SDK this package was built "
                     @"against predates it", version_str];
    *error_msg = strdup([msg UTF8String]);
  }
  return 0;
}

/* ============================================================================
 * library compilation from in-memory source text
 *
 * this is the piece ACFmetal never needed: ACFmetal's
 * metal_load_library()/newLibraryWithURL: loads an already-compiled
 * .metallib produced ahead of time by `xcrun metal` (see
 * metal_functions_to_library() on the R side of ACFmetal). ardea's stated
 * goal is different -- ingest a *source* .metal file directly -- so this
 * uses newLibraryWithSource:options:error:, which invokes the Metal
 * shading-language compiler at runtime against a source string.
 *
 * operational caveat worth knowing up front: newLibraryWithSource: needs
 * the Metal compiler toolchain to be present on the machine actually
 * *running* this code, not just the Metal.framework runtime needed to
 * execute already-compiled shaders. on a bare macOS install without Xcode
 * or the Xcode Command Line Tools, this call can fail (commonly surfaced
 * as an NSError whose domain is MTLLibraryErrorDomain). that failure comes
 * back through `error_msg` like any other failure here -- it is a
 * deployment/environment condition, not a bug in this function.
 * ------------------------------------------------------------------------- */
void* objc_metal_library_from_source(void* device,
                                     const char* source,
                                     int fast_math,
                                     const char* language_version,
                                     char** error_msg) {
  if (!device) {
    if (error_msg) {
      *error_msg = strdup("device pointer is NULL");
    }
    return NULL;
  }
  if (!source) {
    if (error_msg) {
      *error_msg = strdup("source text is NULL");
    }
    return NULL;
  }
  
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  
  NSString* ns_source = [NSString stringWithUTF8String:source];
  if (!ns_source) {
    if (error_msg) {
      *error_msg = strdup("source text is not valid UTF-8");
    }
    return NULL;
  }
  
  MTLCompileOptions* options = [MTLCompileOptions new];
  /* fastMathEnabled is deprecated in newer SDKs in favor of a `mathMode`
   * property, but it remains functional and is the only form available on
   * SDKs this package may still be built against -- keeping it here is a
   * portability choice, not an oversight. */
  #ifdef __MAC_15_0
    options.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
  #else
    options.fastMathEnabled = fast_math ? YES : NO;
  #endif
  
  if (language_version) {
    MTLLanguageVersion parsed_version;
    if (!parse_metal_language_version(language_version,
                                      &parsed_version,
                                      error_msg)) {
      /* error_msg already populated by the parse helper above */
      return NULL;
    }
    options.languageVersion = parsed_version;
  }
  
  NSError* ns_error = nil;
  id<MTLLibrary> library = [mtl_device newLibraryWithSource:ns_source
                                                     options:options
                                                       error:&ns_error];
  
  if (!library) {
    if (error_msg) {
      if (ns_error) {
        *error_msg = strdup([[ns_error localizedDescription] UTF8String]);
      } else {
        *error_msg = strdup("newLibraryWithSource: failed with no NSError "
                            "detail -- possibly a missing Metal compiler "
                            "toolchain on this machine");
      }
    }
    return NULL;
  }
  
  /* newLibraryWithSource: can succeed while still reporting warnings via
   * ns_error (Metal's documented behavior: a non-nil return value AND a
   * non-nil error together mean "compiled, but see these warnings"). that
   * information is currently dropped -- surfacing it would mean adding an
   * output-parameter for a *warning* string distinct from the failure-path
   * error_msg above. flagging this as a known gap rather than silently
   * letting compiler warnings vanish. */
  
  return (__bridge_retained void*)library;
}

/* ----------------------------------------------------------------------------
 * load an already-compiled .metallib from disk, as an alternative to
 * compiling source text at runtime above. this is the loader half of the
 * offline `xcrun metal ... -o file.metallib` compile path (the R side
 * shells out to `xcrun` itself; this just loads the result) -- same role
 * ACFmetal's metal_load_library() played, but returning through the same
 * `metal_library` tag/finalizer as objc_metal_library_from_source() above,
 * so metal_make_program() can hand back an identical type regardless of
 * which compiler produced it.
 *
 * unlike the source-compile path, there is no MTLCompileOptions here --
 * `fast_math`/`languageVersion`/etc. are controlled through `xcrun metal`'s
 * own flags at compile time, before this function is ever called.
 * ------------------------------------------------------------------------- */
void* objc_metal_library_from_metallib(void* device,
                                       const char* path,
                                       char** error_msg) {
  if (!device) {
    if (error_msg) {
      *error_msg = strdup("device pointer is NULL");
    }
    return NULL;
  }
  if (!path) {
    if (error_msg) {
      *error_msg = strdup("metallib path is NULL");
    }
    return NULL;
  }
  
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  
  NSString* ns_path = [NSString stringWithUTF8String:path];
  if (!ns_path) {
    if (error_msg) {
      *error_msg = strdup("metallib path is not valid UTF-8");
    }
    return NULL;
  }
  NSURL* url = [NSURL fileURLWithPath:ns_path];
  
  NSError* ns_error = nil;
  id<MTLLibrary> library = [mtl_device newLibraryWithURL:url error:&ns_error];
  
  if (!library) {
    if (error_msg) {
      if (ns_error) {
        *error_msg = strdup([[ns_error localizedDescription] UTF8String]);
      } else {
        *error_msg = strdup("newLibraryWithURL: failed with no NSError "
                            "detail -- check the file exists and was "
                            "produced by a compatible xcrun/metal toolchain");
      }
    }
    return NULL;
  }
  
  return (__bridge_retained void*)library;
}

/* ============================================================================
 * function + pipeline extraction
 * ========================================================================= */

void* objc_metal_function_from_library(void* library,
                                       const char* name,
                                       char** error_msg) {
  if (!library) {
    if (error_msg) {
      *error_msg = strdup("library pointer is NULL");
    }
    return NULL;
  }
  if (!name) {
    if (error_msg) {
      *error_msg = strdup("function name is NULL");
    }
    return NULL;
  }
  
  id<MTLLibrary> mtl_library = (__bridge id<MTLLibrary>)library;
  
  NSString* ns_name = [NSString stringWithUTF8String:name];
  if (!ns_name) {
    if (error_msg) {
      *error_msg = strdup("function name is not valid UTF-8");
    }
    return NULL;
  }
  
  id<MTLFunction> function = [mtl_library newFunctionWithName:ns_name];
  
  if (!function) {
    if (error_msg) {
      /* Metal gives no NSError here -- construct a useful message from
       * the requested name, same approach ACFmetal already uses */
      NSString* msg = [NSString stringWithFormat:
                       @"function '%s' not found in library -- check the "
                       @"kernel name matches the 'kernel void' declaration "
                       @"in the .metal source", name];
      *error_msg = strdup([msg UTF8String]);
    }
    return NULL;
  }
  
  return (__bridge_retained void*)function;
}

void* objc_metal_create_pipeline(void* device,
                                 void* function,
                                 char** error_msg) {
  if (!device) {
    if (error_msg) {
      *error_msg = strdup("device pointer is NULL");
    }
    return NULL;
  }
  if (!function) {
    if (error_msg) {
      *error_msg = strdup("function pointer is NULL");
    }
    return NULL;
  }
  
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  id<MTLFunction> mtl_function = (__bridge id<MTLFunction>)function;
  
  NSError* ns_error = nil;
  id<MTLComputePipelineState> pipeline =
    [mtl_device newComputePipelineStateWithFunction:mtl_function
                                              error:&ns_error];
  
  if (!pipeline) {
    if (error_msg) {
      if (ns_error) {
        *error_msg = strdup([[ns_error localizedDescription] UTF8String]);
      } else {
        *error_msg = strdup("newComputePipelineStateWithFunction: failed "
                            "with no NSError detail");
      }
    }
    return NULL;
  }
  
  return (__bridge_retained void*)pipeline;
}

/* ============================================================================
 * pipeline introspection
 *
 * this is the number the eventual metal_simple_runner needs for clamping
 * threadgroup size, and it is NOT the same number as the per-*device*
 * `max_threads_per_threadgroup` field metal_device_information() already
 * reports (see metal/devices.m). the device-level number is a hardware
 * ceiling; maxTotalThreadsPerThreadgroup is specific to *this compiled
 * pipeline* and can be lower depending on the kernel's register and
 * threadgroup-memory usage. ACFmetal's runner queries this per-dispatch
 * (see metal_max_threads_per_threadgroup() in its
 * libraries_functions_pipelines.m); exposing it here lets ardea instead
 * snapshot it once, at metal_make_kernelptr() time, and cache it in the
 * MetalKernel struct for reuse across many dispatches of the same pipeline
 * (alongside a device_registry_id for identity checks -- see the
 * MetalKernel struct comment in ardea.h).
 * ------------------------------------------------------------------------- */
size_t metal_pipeline_max_threads(void* pipeline) {
  if (!pipeline) {
    return 0;
  }
  id<MTLComputePipelineState> mtl_pipeline =
    (__bridge id<MTLComputePipelineState>)pipeline;
  return (size_t)[mtl_pipeline maxTotalThreadsPerThreadgroup];
}
