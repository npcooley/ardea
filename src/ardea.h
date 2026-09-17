/* ============================================================================
 * ardea.h
 *
 * Public interface for ardea C functions.
 * author: nicholas cooley
 * maintainer: nicholas cooley
 * 
 * currently under construction, so naming convention are loose
 * 
 * rough naming convention prototypes:
 * C-side:
 * <framework>_exposed_device_count:
 *  return an integer value of framework compliant devices
 * <framework>_framework_available_devices
 *  return a list of framework compliant devices with properties
 * <framework>_program_from_source
 *  for given framework, read in a file and convert it to a program with 1
 *  or more kernel functions
 * <framework>_kernels_from_program
 *  extract kernel pointers from a program for context construction
 * R-side:
 * to be added ...
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * start header guard 
 * shouty case is the convention here
 * ------------------------------------------------------------------------- */
#ifndef ARDEA_H
#define ARDEA_H

#include <Rinternals.h>
/* ----------------------------------------------------------------------------
 * homebrew opencl carries both a cl.h and opencl.h header file with an
 * OpenCL directory symlinked to the CL dir, so the apple check is ... fine
 * but homebrew's opencl.h and cl.h headers aren't equivalent, so there's
 * more going on here than i currently understand
 * AND
 * even though homebrew's opencl headers look like they *should* work,
 * they fail to find the device anyway, so
 * ¯\_(ツ)_/¯
 * ------------------------------------------------------------------------- */
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif
#include <R_ext/Visibility.h>
#include <stdint.h>
#include <stddef.h>
/* -- our optional capability sentinels ------------------------------------ */
#include "config.h"


/* ============================================================================
 * utils.c
 * ========================================================================= */

void *get_checked_external_ptr(SEXP ptr,
                               SEXP expected_tag,
                               const char *type_label);
void set_externalptr_class(SEXP ptr,
                           const char *specific_class);
SEXP metal_sentinel(void);
SEXP cuda_sentinel(void);
SEXP opencl_sentinel(void);

#ifdef HAVE_OPENCL
/* ============================================================================
 * shared external-pointer tag symbols
 * created once via Rf_install() in R_init_ardea() at package load time.
 * every framework file validates external pointers against these same
 * symbols, rather than each file re-interning its own copy.
 * ========================================================================= */
extern SEXP opencl_device_id;
extern SEXP opencl_platform_id;
extern SEXP opencl_context;
extern SEXP opencl_program;
extern SEXP opencl_kernel;

/* ============================================================================
 * opencl structs
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * it is reasonably defensive to re-query the work group size and dimensions
 * when we're invoking simple runner calls and might be the correct-ish
 * strategy generally.
 * looking forward, being able to cache them in a context object so that they
 * can be recalled many times over is also reasonable, so for now we split
 * the difference, set the struct up for future capabilities, but maintain
 * a query function that supervises these inputs for the simple runner
 * ------------------------------------------------------------------------- */
typedef struct {
  cl_context context;
  cl_command_queue queue;
  cl_device_id device;
  /* --------------------------------------------------------------------------
   * currently reserved for future caching optimizations
   * ----------------------------------------------------------------------- */
  size_t max_work_group_size;
  cl_uint max_work_item_dimensions;
  size_t max_work_item_sizes[3];
} OpenCLContext;

typedef enum {
  OPENCL_TYPE_FLOAT = 1,
  OPENCL_TYPE_DOUBLE,
  OPENCL_TYPE_CHAR,
  OPENCL_TYPE_SHORT,
  OPENCL_TYPE_INT,
  OPENCL_TYPE_LONG,
  OPENCL_TYPE_UCHAR,
  OPENCL_TYPE_USHORT,
  OPENCL_TYPE_UINT,
  OPENCL_TYPE_ULONG
} OpenCLType;

/* ============================================================================
 * opencl/buffers.c
 * ========================================================================= */
void *marshal_r_vector(SEXP vec, OpenCLType type, size_t *out_bytes);
SEXP unmarshal_to_r_vector(void *buf, OpenCLType type, R_xlen_t n);

/* ============================================================================
 * opencl/contexts.c
 * ========================================================================= */
void opencl_context_finalizer(SEXP ptr);
SEXP opencl_context_from_device(SEXP device_ptr,
                                SEXP platform_ptr);

/* ============================================================================
 * opencl/devices.c
 * ========================================================================= */
int opencl_device_count(void);
int query_device_strings(cl_device_id device,
                         cl_device_info param,
                         size_t buf_size,
                         char *buf,
                         const char *param_label);
int query_device_units(cl_device_id device,
                       cl_device_info param,
                       size_t param_size,
                       void *out,
                       const char *param_label);
SEXP opencl_exposed_device_count(void);
SEXP opencl_available_devices(void);

/* ============================================================================
 * opencl/handles.c
 * ========================================================================= */
void opencl_program_finalizer(SEXP ptr);
void opencl_kernel_finalizer(SEXP ptr);
SEXP opencl_program_from_source(SEXP cl_file,
                                SEXP context_ptr,
                                SEXP build_options);
SEXP opencl_kernels_from_program(SEXP program_ptr,
                                 SEXP kernel_names);

/* ============================================================================
 * opencl/runners.c
 * ========================================================================= */
SEXP opencl_simple_runner(SEXP context_ptr,
                          SEXP kernel_ptr,
                          SEXP arg_types,
                          SEXP arg_list,
                          SEXP work_dims,
                          SEXP local_dims,
                          SEXP target_local_size);

/* ============================================================================
 * opencl/utils.c
 * ========================================================================= */
SEXP opencl_probe_type_support(SEXP context_ptr,
                               SEXP type_name);
void opencl_default_local_dims(cl_device_id device,
                               size_t target,
                               int active_dims,
                               size_t local[3]);
void release_runner_buffers(cl_mem *buffers,
                            int total_args);
OpenCLType opencl_parse_type(const char *s);
size_t opencl_type_size(OpenCLType t);
int device_supports_fp64(cl_device_id device);
int device_supports_fp16(cl_device_id device);

/* -- end opencl definitions ----------------------------------------------- */
#endif

#ifdef HAVE_METAL
/* ============================================================================
 * shared external-pointer tag symbols
 * created once via Rf_install() in R_init_ardea() at package load time.
 * every framework file validates external pointers against these same
 * symbols, rather than each file re-interning its own copy.
 * ========================================================================= */

extern SEXP metal_device;
extern SEXP metal_context;
extern SEXP metal_queue;
extern SEXP metal_library;
extern SEXP metal_pipeline;
extern SEXP metal_buffer;

/* ============================================================================
 * types, constants, and their helpers
 * ========================================================================= */

// type definitions, these can appear in various 
typedef enum {
  // METAL_TYPE_HALF = 0,      // "half" - 16-bit float
  METAL_TYPE_FLOAT = 1,     // "single" - 32-bit float
  METAL_TYPE_DOUBLE = 2,    // "double" - 64-bit float
  METAL_TYPE_INT8 = 3,      // "byte" - 8-bit signed int
  METAL_TYPE_INT16 = 4,     // "short" - 16-bit signed int
  METAL_TYPE_INT = 5,       // "integer" - 32-bit signed int
  METAL_TYPE_INT64 = 6,     // "long" - 64-bit signed int
  METAL_TYPE_UINT8 = 7,     // "ubyte" - 8-bit unsigned int
  METAL_TYPE_UINT16 = 8,    // "ushort" - 16-bit unsigned int
  METAL_TYPE_UINT = 9,      // "unsigned" - 32-bit unsigned int
  METAL_TYPE_UINT64 = 10    // "ulong" - 64-bit unsigned int
} MetalType;

// metal storage modes
typedef enum {
  METAL_STORAGE_SHARED = 0,   // unified CPU/GPU memory
  METAL_STORAGE_MANAGED = 1,  // managed memory (synced)
  METAL_STORAGE_PRIVATE = 2   // GPU-only memory
} MetalStorageMode;

// container for contexts, just pointers for the queue and the device, these
// get passed around together
typedef struct {
  void* device;
  void* queue;
} MetalContext;

/* ----------------------------------------------------------------------------
 * with metal we compile kernel functions *for devices* as opposed to opencl
 * where we compile kernel functions *agnostic of devices*
 * 
 * using a slot to identity check the device is a lightweight choice
 * ------------------------------------------------------------------------- */
typedef struct {
  void* pipeline;
  size_t max_threads_per_threadgroup;
  uint64_t device_registry_id;
} MetalKernel;

/* -- metal/devices.c ------------------------------------------------------ */
SEXP metal_available_devices(void);
SEXP c_metal_get_all_devices(void);
SEXP c_metal_devices_default(void);
SEXP c_metal_device_information(SEXP device_ptr);
SEXP metal_exposed_device_count(void);

/* -- metal/devices.m ------------------------------------------------------ */
// general device interrogation
void* objc_metal_devices_default(void);
void** objc_metal_get_all_devices(size_t* count);
// general device information
const char* metal_device_name(void* device);
uint64_t metal_device_registry_id(void* device);
int metal_device_has_unified_memory(void* device);
int metal_device_is_low_power(void* device);
int metal_device_is_headless(void* device);
int metal_device_is_removable(void* device);
size_t objc_metal_device_count(void);
// memory limits
uint64_t metal_device_recommended_max_working_set_size(void* device);
uint64_t metal_device_max_buffer_length(void* device);
uint64_t metal_device_current_allocated_size(void* device);
// threading limits
void metal_device_max_threads_per_threadgroup(void* device,
                                              size_t* width,
                                              size_t* height,
                                              size_t* depth);
size_t metal_device_max_threadgroup_memory_length(void* device);
// performance monitoring
void metal_device_sample_timestamps(void* device,
                                    uint64_t* cpu_timestamp,
                                    uint64_t* gpu_timestamp);
int metal_device_supports_counter_sampling(void* device,
                                           int sampling_point);
// physical slot information
int metal_device_location(void* device);
uint64_t metal_device_location_number(void* device);
uint64_t metal_device_peer_group_id(void* device);
uint32_t metal_device_peer_index(void* device);
uint32_t metal_device_peer_count(void* device);

/* -- metal/handles.c ---------------------------------------------------------
 * powers the R-side functions:
 * metal_make_context()
 * metal_make_program()
 * metal_make_kernelptr()
 * ------------------------------------------------------------------------- */
SEXP metal_context_from_device(SEXP device_ptr);
SEXP metal_program_from_source(SEXP metal_file,
                               SEXP context_ptr,
                               SEXP fast_math,
                               SEXP language_version);
SEXP metal_program_from_metallib(SEXP metallib_file,
                                 SEXP context_ptr);
SEXP metal_kernels_from_program(SEXP program_ptr,
                                SEXP context_ptr,
                                SEXP kernel_names);

/* -- metal/handles.m ------------------------------------------------------ */
void* objc_metal_create_queue(void* device);
void* objc_metal_library_from_source(void* device,
                                     const char* source,
                                     int fast_math,
                                     const char* language_version,
                                     char** error_msg);
void* objc_metal_library_from_metallib(void* device,
                                       const char* path,
                                       char** error_msg);
void* objc_metal_function_from_library(void* library,
                                       const char* name,
                                       char** error_msg);
void* objc_metal_create_pipeline(void* device,
                                 void* function,
                                 char** error_msg);
size_t metal_pipeline_max_threads(void* pipeline);

/* -- metal/buffers.c -------------------------------------------------------
 * R vector <-> Metal buffer conversion. unlike opencl/buffers.c (only
 * double/long currently wired up), this covers MetalType's full set --
 * ported from ACFmetal's already-working implementation, so there was
 * nothing to gain by artificially narrowing it to match OpenCL's current
 * subset. NA handling was NOT present in the ACFmetal source this was
 * ported from and has been added here (see metal/buffers.c's header
 * comment for why that mattered).
 * ------------------------------------------------------------------------- */
void metal_convert_r_numeric_to_buffer(const double* r_data,
                                       void* metal_buffer,
                                       size_t length,
                                       MetalType type);
void metal_convert_r_int_to_buffer(const int* r_data,
                                   void* metal_buffer,
                                   size_t length,
                                   MetalType type);
void metal_convert_buffer_to_r(const void* metal_buffer,
                               double* r_data,
                               size_t length,
                               MetalType type);

/* -- metal/buffers.m ------------------------------------------------------ */
void* metal_create_buffer(void* device,
                          size_t length,
                          int storage_mode);
void* metal_buffer_contents(void* buffer);
size_t metal_buffer_length(void* buffer);

/* -- metal/runners.c ---------------------------------------------------------
 * powers the R-side function: (to be wired into simple_wrapper())
 * metal_simple_runner()
 * ------------------------------------------------------------------------- */
SEXP metal_simple_runner(SEXP context_ptr,
                         SEXP kernel_ptr,
                         SEXP arg_types,
                         SEXP arg_list,
                         SEXP work_dims,
                         SEXP threadgroup_dims,
                         SEXP threads_per_threadgroup);

/* -- metal/runners.m ---------------------------------------------------------
 * command buffer lifecycle: create -> encode (in metal_encode_and_commit) ->
 * commit -> wait -> check status/error -> read results -> release.
 * ------------------------------------------------------------------------- */
void* objc_metal_command_buffer_create(void* queue);
void objc_metal_command_buffer_commit(void* command_buffer);
void objc_metal_command_buffer_wait(void* command_buffer);
int objc_metal_command_buffer_status(void* command_buffer);
const char* objc_metal_command_buffer_error_string(void* command_buffer);
void* metal_encode_and_commit(void* queue,
                              void* pipeline,
                              void* output_buffer,
                              size_t num_threadgroups[3],
                              size_t threads_per_threadgroup[3],
                              void** arg_buffers,
                              void** arg_scalars,
                              size_t* arg_scalar_sizes,
                              int* is_scalar,
                              int num_args);

/* -- metal/utils.c -------------------------------------------------------- */
void objc_inclusive_finalizer(SEXP ptr);
void metal_device_finalizer(SEXP device_exp);
void metal_context_finalizer(SEXP context_exp);
void metal_command_queue_finalizer(SEXP queue_exp);
void metal_library_finalizer(SEXP library_exp);
void metal_pipeline_finalizer(SEXP pipeline_exp);
void metal_buffer_finalizer(SEXP buffer_exp);
MetalType metal_parse_type(const char* type_str);
size_t metal_get_element_size(MetalType type);
const char* metal_type_name(MetalType type);
void metal_default_threadgroup_dims(size_t target,
                                    int active_dims,
                                    size_t threadgroup[3]);

/* -- metal/utils.m -------------------------------------------------------- */
void objc_release_obj(void* obj);
void objc_retain_obj(void* obj);
void metal_release_buffer(void* buffer);
void metal_release_command_buffer(void* command_buffer);
void metal_release_command_queue(void* command_queue);
void metal_release_device(void* device);
void metal_release_library(void* library);
void metal_release_function(void* function);
void metal_release_pipeline(void* pipeline);

/* -- end metal definitions ------------------------------------------------ */
#endif

#ifdef HAVE_CUDA

/* ============================================================================
 * shared external-pointer tag symbols
 * ========================================================================= */
extern SEXP cuda_context;
extern SEXP cuda_module;
extern SEXP cuda_kernel;

/* ============================================================================
 * types, constants, and their helpers
 * ========================================================================= */

typedef enum {
  CUDA_TYPE_FLOAT = 1,
  CUDA_TYPE_DOUBLE = 2,
  CUDA_TYPE_INT8 = 3,
  CUDA_TYPE_INT16 = 4,
  CUDA_TYPE_INT = 5,
  CUDA_TYPE_INT64 = 6,
  CUDA_TYPE_UINT8 = 7,
  CUDA_TYPE_UINT16 = 8,
  CUDA_TYPE_UINT = 9,
  CUDA_TYPE_UINT64 = 10
} CudaType;

/* ----------------------------------------------------------------------------
 * NOT an object the way OpenCLContext/MetalContext are -- CUDA identifies
 * "which device is active" via a current CONTEXT on the calling thread,
 * not a handle you pass around the way cl_context/MTLDevice are. this
 * struct holds an EXPLICITLY retained primary_context (via
 * cuDevicePrimaryCtxRetain(), released via cuDevicePrimaryCtxRelease() in
 * the finalizer) rather than relying on whatever ambient context
 * cudaSetDevice() happens to leave current -- Runtime-API-only context
 * management (just calling cudaSetDevice() with nothing explicitly
 * retained) does not guarantee the underlying primary context survives
 * across separate .Call() boundaries the way a held reference does; a
 * Driver API call (cuModuleLoad, cuLaunchKernel) made after that context
 * has been torn down fails with CUDA_ERROR_CONTEXT_IS_DESTROYED. every
 * function that touches a CudaContext must call cuda_activate_context()
 * on it FIRST, as a hard rule -- see cuda/utils.c's cuda_activate_context()
 * for why this can't be enforced by the type system and has to be a
 * discipline instead.
 * ------------------------------------------------------------------------- */
typedef struct {
  int device_index;
  /* -- cudaStream_t, or NULL for the default stream ----------------------- */
  void *stream;
  /* -- CUcontext, explicitly retained -- see above ------------------------ */
  void *primary_context;
} CudaContext;

/* ----------------------------------------------------------------------------
 * CUfunction/CUmodule have no reference counting at all -- unlike
 * MetalKernel (which independently retains its device via ARC) or
 * cl_kernel (refcounted by clRetainKernel/clReleaseKernel), a CUfunction
 * is a raw handle into its owning CUmodule with no protection: unload
 * the module and every function from it is instantly dangling. this
 * struct therefore does NOT try to keep the module alive itself -- that
 * job is done at the R level (see cuda_kernels_from_module() in
 * cuda/handles.c), by anchoring the parent program SEXP as an attribute
 * on the kernel's own externalptr, so R's garbage collector can't reclaim
 * the module while any kernel built from it is still reachable.
 *
 * max_threads_per_block is snapshotted once, at kernel-creation time, via
 * cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) -- the
 * direct CUDA analog of MetalKernel.max_threads_per_threadgroup, and for
 * the same reason: a per-FUNCTION ceiling, not a per-device one, queried
 * once rather than on every dispatch.
 * ------------------------------------------------------------------------- */
typedef struct {
  /* -- CUfunction --------------------------------------------------------- */
  void *function;
  int max_threads_per_block;
  int device_index;    
 /* ---------------------------------------------------------------------------
  * cross-check against the dispatching context,
  * mirroring MetalKernel.device_registry_id's role 
  * ------------------------------------------------------------------------ */
} CudaKernel;

/* -- cuda/devices.c ------------------------------------------------------- */
int cuda_device_count(void);
SEXP cuda_exposed_device_count(void);
SEXP cuda_available_devices(void);

/* -- cuda/handles.c ----------------------------------------------------------
 * powers the R-side functions:
 * cuda_make_context()
 * cuda_make_program()
 * cuda_make_kernelptr()
 * ------------------------------------------------------------------------- */
SEXP cuda_context_from_device(SEXP device_index,
                              SEXP use_default_stream);
SEXP cuda_program_from_ptx(SEXP ptx_file,
                           SEXP context_ptr);
SEXP cuda_kernels_from_module(SEXP program_ptr,
                              SEXP context_ptr,
                              SEXP kernel_names);

/* -- cuda/buffers.c ----------------------------------------------------------
 * host-side R vector <-> CUDA staging-buffer conversion. same shape and
 * same NA-guarding discipline as metal/buffers.c -- see that file's header
 * comment for why the NA/NaN checks matter. this operates on plain host
 * memory (the intermediate staging buffer cudaMemcpy needs), not device
 * memory directly -- unlike Metal's Shared-storage buffers, CUDA's classic
 * allocation model needs an explicit host-side buffer before the explicit
 * copy to device.
 * ------------------------------------------------------------------------- */
void cuda_convert_r_numeric_to_host(const double *r_data,
                                    void *host_buffer,
                                    size_t length,
                                    CudaType type);
void cuda_convert_r_int_to_host(const int *r_data,
                                void *host_buffer,
                                size_t length,
                                CudaType type);
void cuda_convert_host_to_r(const void *host_buffer,
                            double *r_data,
                            size_t length,
                            CudaType type);

/* -- cuda/runners.c ----------------------------------------------------------
 * powers the R-side function: (wired into simple_wrapper())
 * cuda_simple_runner()
 * ------------------------------------------------------------------------- */
SEXP cuda_simple_runner(SEXP context_ptr,
                        SEXP kernel_ptr,
                        SEXP arg_types,
                        SEXP arg_list,
                        SEXP work_dims,
                        SEXP block_dims,
                        SEXP threads_per_block);

/* -- cuda/utils.c --------------------------------------------------------- */
void cuda_ensure_driver_init(void);
void cuda_activate_context(CudaContext *ctx);
void cuda_context_finalizer(SEXP context_exp);
void cuda_module_finalizer(SEXP module_exp);
void cuda_kernel_finalizer(SEXP kernel_exp);
CudaType cuda_parse_type(const char *type_str);
size_t cuda_get_element_size(CudaType type);
const char *cuda_type_name(CudaType type);
void cuda_default_block_dims(size_t target,
                             int active_dims,
                             size_t block[3]);
const char *cuda_driver_error_string(int cu_result);

/* -- end cuda definitions ------------------------------------------------- */
#endif

/* -- end header guard ----------------------------------------------------- */
#endif
