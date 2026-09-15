\name{metal_make_context}
\alias{metal_make_context}
\title{Create a Metal Context for a Device}
\description{
  Construct a Metal context -- a command queue paired with a specific device -- given a device entry returned by \code{\link{metal_device_information}}.
}
\usage{
  metal_make_context(device)
}
\arguments{
  \item{device}{
    alternative_device: a single device entry from the list created by \code{\link{metal_device_information}}.
    Specifically required device properties currently include:
    \itemize{
      \item \code{framework}, must be \code{metal} in this case
      \item \code{device_ptr}, a ptr for the device
    }
  }
}
\details{
  A command queue is created for the supplied device under the hood, and bundled together with the device into a single object so callers don't need to manage two separate pointers.
  The device is retained independently of the \code{device_ptr} external pointer passed in, so that pointer may be garbage collected without affecting the returned context.

  Metal contexts are structurally scoped to a single device -- unlike OpenCL, where a context can span multiple devices.

  The returned context is reference-counted by the underlying Metal runtime and is released automatically when the R external pointer is garbage collected.
}
\value{
  An external pointer of class \code{"metal_context"} wrapping the created device/queue pair.
  This is consumed by other \code{ardea} functions that require a context, such as \code{\link{metal_make_program}} and \code{\link{metal_make_kernelptr}}.
}
\examples{
if (metal_is_available()) {
  devices <- metal_device_information()
  ctx <- metal_make_context(devices[[1]])
}
}
\seealso{
  \code{\link{metal_device_information}},
  \code{\link{metal_make_program}},
  \code{\link{metal_make_kernelptr}}
}
