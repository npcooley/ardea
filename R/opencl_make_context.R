###### -- create an OpenCL context for a device -------------------------------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# given a single device, construct a cl_context scoped to that
# device's platform, and return it as an R external pointer with class
# "opencl_context"
#
# only a single device per context is currently supported. OpenCL itself
# permits multiple devices in one context (unlike CUDA or Metal, both of
# which are one-device-per-context by construction)
# 
# future multidevice context will probably be something like:
# opencl_make_multidevice_context()

###### -- FUNCTION ------------------------------------------------------------

opencl_make_context <- function(device) {
  if (!is(object = device,
          class2 = "alternative_device")) {
    stop("'device' must be a single device entry, e.g. opencl_device_information()[[i]]")
  }
  if (!identical(device$framework, "opencl")) {
    stop("'device' must be an OpenCL device (framework == \"opencl\")")
  }
  if (is.null(device$device_id) ||
      !is(object = device$device_id,
          class2 = "externalptr")) {
    stop("'device' does not contain a valid 'device_id' external pointer")
  }
  if (is.null(device$platform_id) ||
      !is(object = device$platform_id,
          class2 = "externalptr")) {
    stop("'device' does not contain a valid 'platform_id' external pointer")
  }
  
  res <- .Call("opencl_context_from_device",
               device$device_id,
               device$platform_id,
               PACKAGE = "ardea")
  return(res)
}
