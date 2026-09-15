###### -- utility functions for ardea -----------------------------------------
# functions that take very few arguments, or required very simple help files
# live here

###### -- NOTES ---------------------------------------------------------------


###### -- FUNCTIONS -----------------------------------------------------------

opencl_is_available <- function() {
  res <- .Call("opencl_exposed_device_count",
               PACKAGE = "ardea")
  if (res > 0) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}

metal_is_available <- function() {
  res <- .Call("metal_sentinel",
               PACKAGE = "ardea")
  return(res)
}

cuda_is_available <- function() {
  res <- .Call("cuda_sentinel",
               PACKAGE = "ardea")
  return(res)
}

###### -- OTHER ---------------------------------------------------------------
