\name{opencl_make_context}
\alias{opencl_make_context}
\title{Create an OpenCL Context for a Device}
\description{
  Construct an OpenCL context scoped to a single device, given a device entry returned by \code{\link{opencl_device_information}}.
}
\usage{
  opencl_make_context(device)
}
\arguments{
  \item{device}{
    alternative_device: a single device entry from the list created by \code{\link{opencl_device_information}}.
    Specifically required device properties currently include:
    \itemize{
      \item \code{framework}, must be \code{opencl} in this case
      \item \code{device_id}, a ptr for the device
      \item \code{platform_id}, a ptr for the platform
    }
  }
}
\details{
  Context is created under the hood via \code{clCreateContext}, explicitly scoped to the supplied device entry's device and platform.
  This avoids ambiguity on systems with more than one OpenCL platform (ICD) installed, where omitting the platform property can produce implementation-defined behavior.

  Only a single device is included in the context created by this function.
  OpenCL itself permits a context to span multiple devices unlike CUDA or Metal, both of which are structurally limited to one device per context.
  That capability is not currently exposed in this package.

  The returned context is reference-counted by the OpenCL runtime and should be released automatically (\code{clReleaseContext}) when the R external pointer is garbage collected.
}
\value{
  An external pointer of class \code{"opencl_context"} wrapping the created OpenCL context.
  This is consumed by other \code{ardea} functions that require a context, such as \code{\link{opencl_make_program}}.
}
\examples{
if (opencl_is_available()) {
  devices <- opencl_device_information()
  ctx <- opencl_make_context(devices[[1]])
}
}
\seealso{
  \code{\link{opencl_device_information}},
  \code{\link{opencl_make_program}}
}
