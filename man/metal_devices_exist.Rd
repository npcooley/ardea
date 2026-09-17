\name{metal_devices_exist}
\alias{metal_devices_exist}
\title{Check For Metal Devices}
\description{
  Report whether at least one Metal compliant device is available on the current system.
}
\usage{
  metal_devices_exist()
}
\value{
  A single logical value: \code{TRUE} if one or more Metal devices were found, \code{FALSE} if no devices were discovered.
}
\details{
  This function is a thin wrapper for the C function \code{exposed_metal_device_count}.
}
\examples{
  if (metal_is_available()) {
    metal_devices_exist()
  }
}
\seealso{
  \code{\link{metal_is_available}}
  \code{\link{metal_device_information}}
}
