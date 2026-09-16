###### -- return a list containing info on available CUDA devices -------------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# same general shape as with the rest of the framework discovery tools
# with one exception:
# we do not return a device ptr field because there doesn't seem to be an
# analogous concept to that in CUDA, CUDA instead relies on an ordinal
# device integer

###### -- FUNCTION ------------------------------------------------------------

cuda_device_information <- function() {
  
  if (cuda_is_available()) {
    res <- .Call("cuda_available_devices",
                 PACKAGE = "ardea")
    if (length(res) > 0) {
      for (d1 in seq_along(res)) {
        res[[d1]]$framework <- "cuda"
        res[[d1]]$type <- "GPU"
        res[[d1]]$max_compute_units <- res[[d1]]$multiprocessor_count
        class(res[[d1]]) <- c("alternative_device",
                              "list")
      }
      return(res)
    } else {
      return(NULL)
    }
  } else {
    return(NULL)
  }
}
