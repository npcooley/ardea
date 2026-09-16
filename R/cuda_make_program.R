###### -- compile a .cu source file into a loaded CUDA module -----------------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# unlike metal_make_program(), there is no in-process runtime-compile path
# here -- no NVRTC integration yet, so this always shells out to 'nvcc'
# to produce a .ptx file, then loads it. same shell-invocation discipline
# established for metal_make_program()'s xcrun path: every path-bearing
# argument is individually shQuote()'d and passed as a separate element of
# system2()'s 'args' vector (not paste()'d into one string), the exit
# status is checked explicitly, and the expected output file's existence
# is verified before ever calling into C.
#
# 'nvcc_flags' follows the same convention as metal_make_program()'s
# 'xcrun_flags': each element is one shell word, so a flag taking a value
# ("-I", "/path/to/headers") is two elements, not one space-joined string

###### -- FUNCTION ------------------------------------------------------------

cuda_make_program <- function(cuda_file,
                              context,
                              ptx_file = NULL,
                              nvcc_flags = NULL) {
  
  if (!cuda_is_available()) {
    stop("CUDA support was not compiled for this build")
  }
  if (length(cuda_file) != 1 || !is.character(cuda_file)) {
    stop("'cuda_file' must be a character vector of length 1")
  }
  if (!file.exists(cuda_file)) {
    stop("'cuda_file' must exist")
  }
  if (!is(object = context,
          class2 = "cuda_context")) {
    stop("'context' must be a 'cuda_context' object created by 'cuda_make_context()'")
  }
  if (is.null(ptx_file)) {
    ptx_file <- tempfile(fileext = ".ptx")
  } else if (length(ptx_file) != 1 || !is.character(ptx_file)) {
    stop("'ptx_file' must be NULL or a character vector of length 1")
  }
  if (!is.null(nvcc_flags)) {
    if (!is.character(nvcc_flags) || anyNA(nvcc_flags)) {
      stop("'nvcc_flags' must be NULL or a character vector with no NA elements")
    }
    if (any(!nzchar(nvcc_flags))) {
      stop("'nvcc_flags' must not contain empty-string elements -- ",
           "omit an unused flag entirely rather than passing \"\"")
    }
    if (any(nvcc_flags == "-o")) {
      stop("'nvcc_flags' must not include '-o' -- this function already ",
           "supplies it from 'ptx_file'")
    }
  }
  
  # nvcc [nvcc_flags] <cuda_file> -ptx -o <ptx_file>
  if (is.null(nvcc_flags)) {
    nvcc_args <- c(shQuote(cuda_file),
                   "-ptx",
                   "-o", shQuote(ptx_file))
  } else {
    nvcc_args <- c(shQuote(nvcc_flags),
                   shQuote(cuda_file),
                   "-ptx",
                   "-o", shQuote(ptx_file))
  }
  # nvcc_args <- c(if (!is.null(nvcc_flags)) shQuote(nvcc_flags),
  #               shQuote(cuda_file),
  #               "-ptx",
  #               "-o", shQuote(ptx_file))
  
  exit_status <- system2(command = "nvcc",
                         args = nvcc_args,
                         wait = TRUE)
  if (!identical(exit_status, 0L) && !identical(exit_status, 0)) {
    stop("'nvcc' failed to compile '",
         cuda_file,
         "' (exit status ",
         exit_status,
         ")")
  }
  if (!file.exists(ptx_file)) {
    stop("'nvcc' reported success but '",
         ptx_file,
         "' was not created")
  }
  
  res <- .Call("cuda_program_from_ptx",
               ptx_file,
               context,
               PACKAGE = "ardea")
  return(res)
}
