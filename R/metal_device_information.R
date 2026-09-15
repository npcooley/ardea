###### -- return a list containing info on available metal devives ------------
# this function doesn't need any arguments

###### -- NOTES ---------------------------------------------------------------
# return a list where each position is itself a named list containing pertinent
# information about an openCL device that was successfully discovered

###### -- FUNCTION ------------------------------------------------------------

metal_device_information <- function() {
  
  if (metal_is_available()) {
    res <- .Call("metal_available_devices",
                 PACKAGE = "ardea")
    for (d1 in seq_along(res)) {
      res[[d1]]$framework <- "metal"
      res[[d1]]$type <- "GPU"
      res[[d1]]$max_compute_units <- NA
      class(res[[d1]]) <- c("alternative_device",
                            "list")
    }
    return(res)
  } else {
    return(NULL)
  }
}
