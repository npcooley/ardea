###### -- create a CUDA context for a device -----------------------------------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# unlike metal_make_context()/opencl_make_context(), there is no opaque
# device handle to wrap here -- CUDA identifies devices by a plain integer
# ordinal (0..cudaGetDeviceCount()-1). 'device$device_index' fills the same
# structural role 'device$device_ptr' fills for the other two frameworks,
# just as an already-validated integer rather than an external pointer,
# since that's genuinely all CUDA needs. taking an 'alternative_device'
# object here (rather than a bare integer) is a deliberate choice to keep
# the calling convention consistent across all three frameworks -- the
# index is re-validated against a live device count on the C side
# regardless, since nothing guarantees this object isn't stale.
#
# 'use_default_stream = TRUE' (the default) uses CUDA's implicit default
# stream, fine for the single-shot simple runner this powers. setting it
# FALSE creates and owns an explicit stream instead, released when the
# context is garbage collected.

###### -- FUNCTION ------------------------------------------------------------

cuda_make_context <- function(device,
                              use_default_stream = TRUE) {
  
  if (!cuda_is_available()) {
    stop("CUDA support was not compiled for this build")
  }
  if (!is(object = device,
          class2 = "alternative_device")) {
    stop("'device' must be a single device entry, e.g. cuda_device_information()[[i]]")
  }
  if (!identical(device$framework, "cuda")) {
    stop("'device' must be a CUDA device (framework == \"cuda\")")
  }
  if (is.null(device$device_index) ||
      length(device$device_index) != 1 ||
      !is.numeric(device$device_index) ||
      is.na(device$device_index)) {
    stop("'device' does not contain a valid 'device_index' value")
  }
  if (length(use_default_stream) != 1 ||
      !is.logical(use_default_stream) ||
      is.na(use_default_stream)) {
    stop("'use_default_stream' must be a single non-NA logical value")
  }
  
  res <- .Call("cuda_context_from_device",
               as.integer(device$device_index),
               use_default_stream,
               PACKAGE = "ardea")
  return(res)
}
