/* ============================================================================
 * metal/utils.m
 * author: nicholas cooley
 * maintainer: nicholas cooley
 *
 * generic Objective-C retain/release helpers for Metal objects.
 * these are plain C-callable functions (no SEXP, no R API calls) --
 * see the R/C/ObjC boundary discussion: functions here must never call
 * into R's error-raising machinery while holding a live ARC reference.
 * ========================================================================= */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "ardea.h"

/* ============================================================================
 * generic retain / release -- must be matched
 * ========================================================================= */

void objc_release_obj(void* obj) {
  if (obj) {
    CFRelease(obj);
  }
}

void objc_retain_obj(void* obj) {
  if (obj) {
    CFRetain(obj);
  }
}

/* ============================================================================
 * per-kind release functions -- used by the finalizers in utils.c
 * ========================================================================= */

void metal_release_buffer(void* buffer) {
  if (buffer) {
    CFRelease(buffer);
  }
}

void metal_release_command_buffer(void* command_buffer) {
  if (command_buffer) {
    CFRelease(command_buffer);
  }
}

void metal_release_command_queue(void* queue) {
  if (queue) {
    CFRelease(queue);
  }
}

void metal_release_device(void* device) {
  if (device) {
    CFRelease(device);
  }
}

void metal_release_library(void* library) {
  if (library) {
    CFRelease(library);
  }
}

void metal_release_function(void* function) {
  if (function) {
    CFRelease(function);
  }
}

void metal_release_pipeline(void* pipeline) {
  if (pipeline) {
    CFRelease(pipeline);
  }
}
