###### -- create dispatch-ready kernel pointers from a compiled library -------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# take in a compiled metal_library, the metal_context it should be paired
# with, and a vector of kernel function names declared in the source; return
# a named list of "metal_pipeline" external pointers, one per name
#
# unlike opencl_make_kernelptr() (which hands back a cl_kernel -- an object
# that still needs binding to a device-specific execution context at dispatch
# time), each returned pointer here already bundles the compiled
# MTLComputePipelineState together with the device it was built against, so
# it is dispatch-ready as-is: no separate pipeline-creation step remains for
# the eventual metal_simple_runner to perform per-call

###### -- FUNCTION ------------------------------------------------------------

metal_make_kernelptr <- function(program,
                                 context,
                                 kernel_names) {
  
  if (!metal_is_available()) {
    stop("Metal support was not compiled for this build.")
  }
  
  if (!is(object = program,
          class2 = "metal_library")) {
    stop("'program' must be a 'metal_library' object created by 'metal_make_program()'")
  }
  if (!is(object = context,
          class2 = "metal_context")) {
    stop("'context' must be a 'metal_context' object created by 'metal_make_context()'")
  }
  if (length(kernel_names) < 1 || !is.character(kernel_names)) {
    stop("'kernel_names' must be a character vector of length 1 or greater")
  }
  
  res <- .Call("metal_kernels_from_program",
               program,
               context,
               kernel_names,
               PACKAGE = "ardea")
  names(res) <- kernel_names
  return(res)
}
