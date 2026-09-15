/* ============================================================================
 * metal/buffers.m
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * Objective-C interface to Metal buffer creation and access. direct port
 * of ACFmetal's buffers.m -- nothing here needed correction, the issues
 * found during review were all on the C side (metal/buffers.c) and the
 * dispatch side (metal/runners.m).
 *
 * release is handled by the existing generic metal_release_buffer() in
 * metal/utils.m (CFRelease) -- no separate release function needed here.
 *
 * storage mode: metal_simple_runner() always requests METAL_STORAGE_SHARED
 * for every buffer it creates. Shared storage is coherent CPU/GPU memory
 * on every Mac GPU (unified-memory Apple Silicon and discrete-memory Intel
 * Macs alike) -- unlike Managed storage, it never needs an explicit
 * didModifyRange:/synchronize call, which is what keeps the runner's
 * buffer handling simple. on a discrete GPU this may be slower than
 * Private storage plus an explicit blit copy would be (Shared memory is
 * host-resident, so the GPU accesses it over the bus rather than from
 * local VRAM) -- a correctness-first, portable choice over a
 * performance-optimal one for a "simple runner", not an oversight.
 * ========================================================================= */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <string.h>
#include "ardea.h"

#ifdef error
#undef error
#endif

/* ----------------------------------------------------------------------------
 * create an MTLBuffer of `length` bytes under the requested storage mode.
 * `storage_mode` is one of the MetalStorageMode enum values (ardea.h);
 * any unrecognized value falls back to Shared, matching ACFmetal's
 * original behavior -- this one default-falls-through deliberately, unlike
 * the buffers.c conversion switches, since an out-of-range storage mode
 * here has an obviously safe fallback (Shared), where an out-of-range
 * MetalType in a conversion switch does not (there's no "safe" element
 * size or conversion to fall back to).
 * ------------------------------------------------------------------------- */
void* metal_create_buffer(void* device,
                          size_t length,
                          int storage_mode) {
  if (!device) {
    return NULL;
  }
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  
  MTLResourceOptions options;
  switch (storage_mode) {
  case METAL_STORAGE_SHARED:
    options = MTLResourceStorageModeShared;
    break;
  case METAL_STORAGE_MANAGED:
    options = MTLResourceStorageModeManaged;
    break;
  case METAL_STORAGE_PRIVATE:
    options = MTLResourceStorageModePrivate;
    break;
  default:
    options = MTLResourceStorageModeShared;
  }
  
  id<MTLBuffer> buffer = [mtl_device newBufferWithLength:length options:options];
  if (!buffer) {
    return NULL;
  }
  
  return (__bridge_retained void*)buffer;
}

/* ----------------------------------------------------------------------------
 * CPU-visible pointer to a Shared- or Managed-storage buffer's contents.
 * returns NULL (rather than a garbage/inaccessible pointer) for
 * Private-storage buffers, per Metal's own documented behavior for
 * -contents on that storage mode -- metal_simple_runner() never requests
 * Private storage, so this case shouldn't be hit through this package's
 * own code paths, but the function itself makes no such assumption.
 * ------------------------------------------------------------------------- */
void* metal_buffer_contents(void* buffer) {
  if (!buffer) {
    return NULL;
  }
  id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer;
  return [mtl_buffer contents];
}

size_t metal_buffer_length(void* buffer) {
  if (!buffer) {
    return 0;
  }
  id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)buffer;
  return [mtl_buffer length];
}
