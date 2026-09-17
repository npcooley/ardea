\name{cuda_devices_exist}
\alias{cuda_devices_exist}
\title{Check For Metal Devices}
\description{
  Report whether at least one CUDA compliant device is available on the current system.
}
\usage{
  cuda_devices_exist()
}
\value{
  A single logical value: \code{TRUE} if one or more CUDA devices were found, \code{FALSE} if no devices were discovered.
}
\details{
  This function is a thin wrapper for the C function \code{cuda_exposed_device_count}.
}
\examples{
  if (cuda_is_available()) {
    cuda_devices_exist()
  }
}
\seealso{
  \code{\link{cuda_is_available}}
  \code{\link{cuda_device_information}}
}
