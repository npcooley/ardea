\name{opencl_devices_exist}
\alias{opencl_devices_exist}
\title{Check For OpenCL Compliant Devices}
\description{
  Report whether at least one OpenCL compliant device is available on the current system, across all detected platforms.
}
\usage{
  opencl_devices_exist()
}
\value{
  A single logical value: \code{TRUE} if one or more OpenCL devices were found on any platform, \code{FALSE} if no devices were discovered.
  If this package installed successfully, \code{FALSE} should indicate that there are no OpenCL compliant *devices* present.
  As successful package installation requires working OpenCL headers and shared libraries.
}
\details{
  This function is an R wrapper for \code{ardea}'s C function \code{exposed_device_count}.
  If \code{exposed_device_count} returns a value greater than zero, this function returns \code{TRUE}.
  Device enumeration should hypothetically include openCL compliant CPUs enabled by POCL-like runtimes, but this has not been tested.
  This means systems with OpenCL compliant CPUs, but no GPUs will return \code{TRUE}.
  
  Platforms that fail to return valid information are skipped, meaning broken or misconfigured ICDs will remove a device from availability.
}
\examples{
  if (opencl_is_available()) {
    opencl_devices_exist()
  }
}
\seealso{
  \code{\link{opencl_is_available}}
  \code{\link{opencl_device_information}}
}
