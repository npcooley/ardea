###### -- create dispatch-ready kernel pointers from a CUDA module ------------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# unlike metal_make_kernelptr(), the C side of this also anchors the
# parent 'program' object as an attribute on each returned kernel pointer
# -- CUfunction has no reference counting at all, so unlike MetalKernel
# (which independently retains its device), a CudaKernel has nothing
# native to lean on to stay valid if its module is unloaded. that anchor
# is what keeps 'program' (and therefore its module) alive for as long as
# any kernel built from it is still reachable, even if the caller doesn't
# keep their own reference to 'program' around

###### -- FUNCTION ------------------------------------------------------------

cuda_make_kernelptr <- function(program,
                                context,
                                kernel_names) {
  
  if (!cuda_is_available()) {
    stop("CUDA support was not compiled for this build")
  }
  if (!is(object = program,
          class2 = "cuda_module")) {
    stop("'program' must be a 'cuda_module' object created by 'cuda_make_program()'")
  }
  if (!is(object = context,
          class2 = "cuda_context")) {
    stop("'context' must be a 'cuda_context' object created by 'cuda_make_context()'")
  }
  if (length(kernel_names) < 1 || !is.character(kernel_names)) {
    stop("'kernel_names' must be a character vector of length 1 or greater")
  }
  
  res <- .Call("cuda_kernels_from_module",
               program,
               context,
               kernel_names,
               PACKAGE = "ardea")
  names(res) <- kernel_names
  return(res)
}
