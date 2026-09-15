###### -- create a Metal context for a device ---------------------------------
# author: nicholas cooley
# maintainer: nicholas cooley

###### -- NOTES ---------------------------------------------------------------
# given a single device entry returned from metal_device_information(),
# construct a command queue for that device and return {device, queue}
# bundled together as an R external pointer with class "metal_context"
#
# mirrors opencl_make_context()'s contract: a single device in, a single
# tagged external pointer out. Metal is structurally one-device-per-context
# already (unlike OpenCL), so there's no multi-device variant to anticipate
# here the way there eventually may be for a hypothetical
# 'opencl_make_multidevice_context()' function

###### -- FUNCTION ------------------------------------------------------------

metal_make_context <- function(device) {
  
  if (!metal_is_available()) {
    stop("Metal support was not compiled for this build.")
  }
  
  if (!is(object = device,
          class2 = "alternative_device")) {
    stop("'device' must be a single device entry, e.g. metal_device_information()[[i]]")
  }
  if (!identical(device$framework, "metal")) {
    stop("'device' must be a Metal device (framework == \"metal\")")
  }
  if (is.null(device$device_ptr) ||
      !is(object = device$device_ptr,
          class2 = "externalptr")) {
    stop("'device' does not contain a valid 'device_ptr' external pointer")
  }
  
  res <- .Call("metal_context_from_device",
               device$device_ptr,
               PACKAGE = "ardea")
  return(res)
}
