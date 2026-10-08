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

###### -- check for the offline metal compiler --------------------------------

# three stage check for the system side metal compiler
# correct os
# xcrun exists
# xcrun has the right toolchain

metal_compiler_is_available <- function() {
  # only valid to check when we have the right OS
  sys_info <- Sys.info()
  if (is.null(sys_info) ||
      !identical(unname(sys_info[["sysname"]]),
                 "Darwin")) {
    return(FALSE)
  }
  xcrun_path <- Sys.which("xcrun")
  if (length(xcrun_path) != 1L ||
      !nzchar(xcrun_path)) {
    return(FALSE)
  }
  # run the actual tool rather than only locating it: some Xcode versions
  # ship a stub 'metal' that exists but fails when the toolchain is missing
  status <- tryCatch(suppressWarnings(system2(command = xcrun_path,
                                              args = c("-sdk",
                                                       "macosx",
                                                       "metal",
                                                       "--version"),
                                              stdout = FALSE,
                                              stderr = FALSE)),
                     error = function(e) {
                       NA_integer_
                     })
  res <- length(status) == 1L &&
    !is.na(status) &&
    identical(as.integer(status),
              0L)
  return(res)
}

###### -- OTHER ---------------------------------------------------------------
