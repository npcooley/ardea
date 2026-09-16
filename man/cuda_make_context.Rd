\name{cuda_make_context}
\alias{cuda_make_context}
\title{Create a CUDA Context for a Device}
\description{
  Construct a CUDA context for a specific device, given a device entry returned by \code{\link{cuda_device_information}}.
}
\usage{
  cuda_make_context(device,
                    use_default_stream = TRUE)
}
\arguments{
  \item{device}{
    alternative_device: a single device entry from the list created by \code{\link{cuda_device_information}}.
    Specifically required device properties currently include:
    \itemize{
      \item \code{framework}, must be \code{cuda} in this case
      \item \code{device_index}, an integer ordinal for the device (\code{0..cudaGetDeviceCount()-1})
    }
  }
  \item{use_default_stream}{
    logical: a single non-NA value. If \code{TRUE} (the default), CUDA's implicit default stream is used. If \code{FALSE}, an explicit stream is created and owned by the returned context, released when it is garbage collected.
  }
}
\details{
  Unlike \code{\link{metal_make_context}}/\code{\link{opencl_make_context}}, there is no external device pointer because CUDA identifies devices by a plain integer ordinal. \code{device$device_index} fills the same structural role \code{device$device_ptr} fills for the other two frameworks.
  Taking an \code{alternative_device} object here (rather than a bare integer) is a deliberate choice to keep the calling convention consistent across all supported frameworks; the index is re-validated against a live device count on the C side regardless, since nothing guarantees the supplied object isn't stale (e.g. built in an earlier session, or a device reconfigured since).

  CUDA has no object that "is" a context the way \code{cl_context}/\code{MTLDevice} do -- \code{cudaSetDevice()} sets which GPU is current for the calling thread, and every subsequent CUDA call to a given context implicitly targets whatever was last set.
  The object returned here is a thin wrapper around that device index (and, optionally, a stream); every \code{ardea} function that consumes it re-activates the correct device internally before doing any work.
}
\value{
  An external pointer of class \code{"cuda_context"}.
  This is consumed by other \code{ardea} functions that require a context, such as \code{\link{cuda_make_program}} and \code{\link{cuda_make_kernelptr}}.
}
\examples{
if (cuda_is_available()) {
  devices <- cuda_device_information()
  ctx <- cuda_make_context(devices[[1]])
}
}
\seealso{
  \code{\link{cuda_device_information}},
  \code{\link{cuda_make_program}},
  \code{\link{cuda_make_kernelptr}}
}
