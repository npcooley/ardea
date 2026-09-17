\name{opencl_is_available}
\alias{opencl_is_available}
\title{Check For OpenCL Compilation Success}
\description{
  Report whether OpenCL compilation probes were successful.
}
\usage{
  opencl_is_available()
}
\value{
  A single logical value: \code{TRUE} if compilation of OpenCL capability probes were successful.
  If this package installed successfully \code{FALSE} indicates the compilation probe for OpenCL was not successful.
}
\details{
  This function is an R wrapper for \code{ardea}'s C function \code{opencl_sentinel}.
}
\examples{
  opencl_is_available()
}
\seealso{
  \code{\link{opencl_device_information}}
  \code{\link{opencl_devices_exist}}
}
