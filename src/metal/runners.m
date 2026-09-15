/* ============================================================================
 * metal/runners.m
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * command buffer creation, encoding, dispatch, and completion-checking.
 * ported from ACFmetal's command.m, narrowed to what metal_simple_runner()
 * (metal/runners.c) actually needs -- ACFmetal's queue-creation functions
 * (objc_metal_create_queue() and friends) are not duplicated here, since
 * ardea already has that in metal/handles.m from the context-construction
 * work. see the note in ardea.h / the prior design discussion for why this
 * lives in runners.m/.c rather than a separate command.c/.m or contexts.c/.m
 * file: opencl/runners.c already holds OpenCL's entire dispatch sequence
 * (clEnqueueWriteBuffer/clEnqueueNDRangeKernel/clEnqueueReadBuffer) with no
 * separate "commands" file, so this is the direct Metal analog of that,
 * not a new architectural category.
 *
 * the per-dispatch submission chain:
 *
 *   queue (persistent, from MetalContext)
 *     -> commandBuffer (transient, one per dispatch)
 *       -> computeCommandEncoder (transient, encodes kernel + args)
 *         -> endEncoding
 *       -> commit
 *     -> waitUntilCompleted
 *     -> [status/error check -- see metal_encode_and_commit()'s caller in
 *         runners.c; this file only exposes the means to check, the
 *         decision to error on it lives on the C side]
 *
 * the single most important correction relative to ACFmetal here isn't in
 * this file at all -- it's that runners.c actually CALLS
 * objc_metal_command_buffer_status()/error_string() after waiting, where
 * ACFmetal's metal_simple_runner() never did, despite this exact
 * functionality already existing and being commented as intended for
 * that purpose in ACFmetal's own command.m.
 * ========================================================================= */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "ardea.h"

#ifdef error
#undef error
#endif

/* ============================================================================
 * command buffer lifecycle
 * ========================================================================= */

/* ----------------------------------------------------------------------------
 * NOTES:
 * command buffers are transient -- created from the queue, encoded into,
 * committed, waited on, and discarded. they are not meant to be held
 * across dispatches. Metal auto-releases these internally, but this is
 * retained here (via __bridge_retained) to protect it across the C/ObjC
 * boundary, same as every other object crossing that boundary in this
 * package -- must be matched with objc_release_obj() (metal/utils.m) once
 * the C side is done with it.
 * ------------------------------------------------------------------------- */
void* objc_metal_command_buffer_create(void* queue) {
  if (!queue) {
    return NULL;
  }
  id<MTLCommandQueue> mtl_queue = (__bridge id<MTLCommandQueue>)queue;
  id<MTLCommandBuffer> command_buffer = [mtl_queue commandBuffer];
  if (!command_buffer) {
    return NULL;
  }
  return (__bridge_retained void*)command_buffer;
}

/* commit the command buffer for execution. sequence: endEncoding -> commit -> wait */
void objc_metal_command_buffer_commit(void* command_buffer) {
  if (!command_buffer) {
    return;
  }
  id<MTLCommandBuffer> mtl_buffer = (__bridge id<MTLCommandBuffer>)command_buffer;
  [mtl_buffer commit];
}

/* block the calling thread until execution completes. sequence: commit -> wait -> read results */
void objc_metal_command_buffer_wait(void* command_buffer) {
  if (!command_buffer) {
    return;
  }
  id<MTLCommandBuffer> mtl_buffer = (__bridge id<MTLCommandBuffer>)command_buffer;
  [mtl_buffer waitUntilCompleted];
}

/* ----------------------------------------------------------------------------
 * MTLCommandBufferStatus values:
 *   0 == not enqueued   1 == enqueued   2 == committed
 *   3 == scheduled      4 == completed  5 == error
 *   -1 (non-standard, this function's own convention) == invalid buffer
 *
 * this is what lets a GPU-side failure (out-of-bounds write, unsupported
 * instruction, etc.) be caught BEFORE the output buffer is read, rather
 * than silently handing back whatever garbage happens to be sitting in
 * device memory. see metal_simple_runner() in runners.c for where this
 * actually gets checked.
 * ------------------------------------------------------------------------- */
int objc_metal_command_buffer_status(void* command_buffer) {
  if (!command_buffer) {
    return -1;
  }
  id<MTLCommandBuffer> mtl_buffer = (__bridge id<MTLCommandBuffer>)command_buffer;
  return (int)mtl_buffer.status;
}

/* ----------------------------------------------------------------------------
 * NULL for both an invalid buffer AND the absence of an error (a non-nil
 * return here does not by itself mean failure occurred -- check
 * objc_metal_command_buffer_status() for that; this only supplies detail
 * once a failure is already known). the returned C string points into
 * memory owned by NSError -- copy it (e.g. via strdup() or straight into
 * an Rf_error() format string, which itself copies) before the command
 * buffer is released.
 * ------------------------------------------------------------------------- */
const char* objc_metal_command_buffer_error_string(void* command_buffer) {
  if (!command_buffer) {
    return NULL;
  }
  id<MTLCommandBuffer> mtl_buffer = (__bridge id<MTLCommandBuffer>)command_buffer;
  if (mtl_buffer.error) {
    return [[mtl_buffer.error localizedDescription] UTF8String];
  }
  return NULL;
}

/* ============================================================================
 * encode + dispatch + commit, in one shot
 *
 * runners.c has already resolved buffers vs. scalars, built every input
 * buffer, and converted every scalar value into a raw byte blob by the
 * time this is called -- this function's only job is encoding: bind the
 * pipeline, bind the output buffer at index 0, bind every remaining
 * argument at its index (buffer via setBuffer:, scalar via setBytes:),
 * dispatch the threadgroups, end encoding, and commit.
 *
 * this does NOT wait for completion and does NOT check status/error --
 * that's runners.c's job, after this returns, so the C side (which is
 * what actually knows how to Rf_error()) is the one deciding what to do
 * about a GPU-side failure, not this ObjC layer.
 * ========================================================================= */
void* metal_encode_and_commit(void* queue,
                              void* pipeline,
                              void* output_buffer,
                              size_t num_threadgroups[3],
                              size_t threads_per_threadgroup[3],
                              void** arg_buffers,
                              void** arg_scalars,
                              size_t* arg_scalar_sizes,
                              int* is_scalar,
                              int num_args) {
  
  if (!queue || !pipeline || !output_buffer) {
    return NULL;
  }
  
  id<MTLCommandQueue> mtl_queue = (__bridge id<MTLCommandQueue>)queue;
  id<MTLComputePipelineState> mtl_pipeline = (__bridge id<MTLComputePipelineState>)pipeline;
  id<MTLBuffer> mtl_output = (__bridge id<MTLBuffer>)output_buffer;
  
  id<MTLCommandBuffer> commandBuffer = [mtl_queue commandBuffer];
  if (!commandBuffer) {
    return NULL;
  }
  
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!encoder) {
    return NULL;
  }
  
  [encoder setComputePipelineState:mtl_pipeline];
  [encoder setBuffer:mtl_output offset:0 atIndex:0];
  
  /* remaining arguments, indices 1..num_args (0 is the output bound above) */
  int buffer_idx = 0;
  int scalar_idx = 0;
  
  for (int i = 0; i < num_args; i++) {
    int metal_index = i + 1;
    
    if (is_scalar[i]) {
      [encoder setBytes:arg_scalars[scalar_idx]
                 length:arg_scalar_sizes[scalar_idx]
                atIndex:metal_index];
      scalar_idx++;
    } else {
      id<MTLBuffer> arg_buffer = (__bridge id<MTLBuffer>)arg_buffers[buffer_idx];
      [encoder setBuffer:arg_buffer offset:0 atIndex:metal_index];
      buffer_idx++;
    }
  }
  
  MTLSize threadgroupSize = MTLSizeMake(threads_per_threadgroup[0],
                                        threads_per_threadgroup[1],
                                        threads_per_threadgroup[2]);
  MTLSize threadgroupCount = MTLSizeMake(num_threadgroups[0],
                                         num_threadgroups[1],
                                         num_threadgroups[2]);
  
  [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
  [encoder endEncoding];
  [commandBuffer commit];
  
  return (__bridge_retained void*)commandBuffer;
}
