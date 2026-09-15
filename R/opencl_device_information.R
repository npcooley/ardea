###### -- return a list containing info on available openCL devives -----------
# this function doesn't need any arguments

###### -- NOTES ---------------------------------------------------------------
# return a list where each position is itself a named list containing pertinent
# information about an openCL device that was successfully discovered

###### -- FUNCTION ------------------------------------------------------------

opencl_device_information <- function() {
  
  # check whether there are devices to query, if not return NULL
  if (opencl_is_available()) {
    res <- .Call("opencl_available_devices",
                 PACKAGE = "ardea")
    for (d1 in seq_along(res)) {
      res[[d1]]$framework <- "opencl"
      class(res[[d1]]) <- c("alternative_device",
                            "list")
    }
    if (length(res) == 0) {
      return(NULL)
    } else {
      class(res) <- c("device_list",
                      "list")
      return(res)
    }
  } else {
    return(NULL)
  }
}
