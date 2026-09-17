###### -- utility functions for ardea -----------------------------------------
# functions that take very few arguments, or required very simple help files
# live here

###### -- NOTES ---------------------------------------------------------------


###### -- Sentinels -----------------------------------------------------------

opencl_is_available <- function() {
  res <- .Call("opencl_sentinel",
               PACKAGE = "ardea")
  return(res)
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

###### -- Device presence -----------------------------------------------------

opencl_devices_exist <- function() {
  if (opencl_is_available()) {
    res <- .Call("opencl_exposed_device_count",
                 PACKAGE = "ardea")
    if (res > 0) {
      return(TRUE)
    } else {
      return(FALSE)
    }
  } else {
    return(FALSE)
  }
}

# i need to build a bare bones exposed device count to replace this query ...
metal_devices_exist <- function() {
  if (metal_is_available()) {
    res <- .Call("metal_exposed_device_count",
                 PACKAGE = "ardea")
    if (res > 0) {
      return(TRUE)
    } else {
      FALSE
    }
  } else {
    return(FALSE)
  }
}

cuda_devices_exist <- function() {
  if (cuda_is_available()) {
    res <- .Call("cuda_exposed_device_count",
                 PACKAGE = "ardea")
    if (res > 0) {
      return(TRUE)
    } else {
      return(FALSE)
    }
  } else {
    return(FALSE)
  }
}

###### -- OTHER ---------------------------------------------------------------
