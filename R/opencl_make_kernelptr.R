###### -- create a kernel ptr from an intermediate ptr ------------------------

###### -- NOTES ---------------------------------------------------------------
# notes?
# take in a ptr intermediate and a vector of characters specifying
# kernel function names within the program
# returns a list of ptrs!

###### -- FUNCTION ------------------------------------------------------------

opencl_make_kernelptr <- function(program,
                                  kernel_names) {
  if (!is(object = program,
          class2 = "externalptr")) {
    stop("'program' must be an externalptr object")
  }
  if (length(kernel_names) < 1 || !is.character(kernel_names)) {
    stop("'kernel_names' must be a character vector of length 1 or greater")
  }
  
  res <- .Call("opencl_kernels_from_program",
               program,
               kernel_names,
               PACKAGE = "ardea")
  names(res) <- kernel_names
  return(res)
}
