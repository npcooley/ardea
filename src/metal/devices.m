/* ============================================================================
 * metal/devices.m
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * device enumeration and per-device property queries. plain C-callable
 * (no SEXP here) -- see metal/devices.c for the R-facing layer.
 * ========================================================================= */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <TargetConditionals.h>

#include "ardea.h"

void* objc_metal_devices_default(void) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    return NULL;
  }
  return (__bridge_retained void*)device;
}

void** objc_metal_get_all_devices(size_t* count) {
  if (!count) {
    return NULL;
  }
  
#if TARGET_OS_OSX
  NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
#else
  id<MTLDevice> single = MTLCreateSystemDefaultDevice();
  NSArray<id<MTLDevice>>* devices = single ? @[single] : @[];
#endif
  
  if (!devices || [devices count] == 0) {
    *count = 0;
    return NULL;
  }
  
  *count = [devices count];
  void** device_array = (void**)malloc(*count * sizeof(void*));
  if (!device_array) {
    *count = 0;
    return NULL;
  }
  for (size_t i = 0; i < *count; i++) {
    device_array[i] = (__bridge_retained void*)devices[i];
  }
  return device_array;
}

const char* metal_device_name(void* device) {
  if (!device) return NULL;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return [mtl_device.name UTF8String];
}

uint64_t metal_device_registry_id(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return mtl_device.registryID;
}

int metal_device_has_unified_memory(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return mtl_device.hasUnifiedMemory ? 1 : 0;
}

int metal_device_is_low_power(void* device) {
  if (!device) return 0;
#if TARGET_OS_OSX
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return mtl_device.isLowPower ? 1 : 0;
#else
  return 0;
#endif
}

int metal_device_is_headless(void* device) {
  if (!device) return 0;
#if TARGET_OS_OSX
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return mtl_device.isHeadless ? 1 : 0;
#else
  return 0;
#endif
}

int metal_device_is_removable(void* device) {
  if (!device) return 0;
#if TARGET_OS_OSX
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return mtl_device.isRemovable ? 1 : 0;
#else
  return 0;
#endif
}

uint64_t metal_device_recommended_max_working_set_size(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return mtl_device.recommendedMaxWorkingSetSize;
}

uint64_t metal_device_max_buffer_length(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return mtl_device.maxBufferLength;
}

uint64_t metal_device_current_allocated_size(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return mtl_device.currentAllocatedSize;
}

void metal_device_max_threads_per_threadgroup(void* device,
                                              size_t* width,
                                              size_t* height,
                                              size_t* depth) {
  if (!device || !width || !height || !depth) {
    if (width)  *width  = 0;
    if (height) *height = 0;
    if (depth)  *depth  = 0;
    return;
  }
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  MTLSize max_threads = mtl_device.maxThreadsPerThreadgroup;
  *width  = max_threads.width;
  *height = max_threads.height;
  *depth  = max_threads.depth;
}

size_t metal_device_max_threadgroup_memory_length(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return mtl_device.maxThreadgroupMemoryLength;
}

void metal_device_sample_timestamps(void* device,
                                    uint64_t* cpu_timestamp,
                                    uint64_t* gpu_timestamp) {
  if (!device || !cpu_timestamp || !gpu_timestamp) {
    if (cpu_timestamp) *cpu_timestamp = 0;
    if (gpu_timestamp) *gpu_timestamp = 0;
    return;
  }
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  MTLTimestamp cpu_ts = 0;
  MTLTimestamp gpu_ts = 0;
  [mtl_device sampleTimestamps:&cpu_ts gpuTimestamp:&gpu_ts];
  *cpu_timestamp = (uint64_t)cpu_ts;
  *gpu_timestamp = (uint64_t)gpu_ts;
}

int metal_device_supports_counter_sampling(void* device, int sampling_point) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return [mtl_device supportsCounterSampling:(MTLCounterSamplingPoint)sampling_point] ? 1 : 0;
}

#if TARGET_OS_OSX
int metal_device_location(void* device) {
  if (!device) return -1;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return (int)mtl_device.location;
}

uint64_t metal_device_location_number(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return (uint64_t)mtl_device.locationNumber;
}

uint64_t metal_device_peer_group_id(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return (uint64_t)mtl_device.peerGroupID;
}

uint32_t metal_device_peer_index(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return (uint32_t)mtl_device.peerIndex;
}

uint32_t metal_device_peer_count(void* device) {
  if (!device) return 0;
  id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;
  return (uint32_t)mtl_device.peerCount;
}

/* -- TARGET_OS_OSX -------------------------------------------------------- */
#endif 
