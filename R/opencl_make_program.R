###### -- create a ptr intermediate from a dot cl file ------------------------

###### -- NOTES ---------------------------------------------------------------
# notes?
# take in a file, a context ptr and optional build options

###### -- FUNCTION ------------------------------------------------------------

opencl_make_program <- function(cl_file,
                                context,
                                build_options = NULL) {
  if (length(cl_file) != 1 || !is.character(cl_file)) {
    stop("'cl_file' must be a character vector of length 1")
  }
  if (!file.exists(cl_file)) {
    stop("'cl_file' must exist")
  }
  if (!is(object = context,
          class2 = "externalptr")) {
    stop("'context' must be an externalptr object")
  }
  if (!is.null(build_options) &&
      (length(build_options) != 1 || !is.character(build_options))) {
    stop("'build_options' must be NULL or a character vector of length 1")
  }
  
  res <- .Call("opencl_program_from_source",
               cl_file,
               context,
               build_options,
               PACKAGE = "ardea")
  
  return(res)
}
