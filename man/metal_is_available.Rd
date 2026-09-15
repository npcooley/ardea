\name{metal_is_available}
\alias{metal_is_available}
\title{Check Whether Metal Support Was Compiled Into This Build}
\description{
  Report whether \code{ardea} was compiled with Metal support on this system.
}
\usage{
  metal_is_available()
}
\value{
  Logical: \code{TRUE} if Metal support was detected and compiled in at installation time, \code{FALSE} otherwise.
  \code{FALSE} is the expected result on any non-macOS platform, and is also expected on a macOS system where Metal detection failed during installation.
}
\details{
  This function is an R wrapper for \code{ardea}'s C function \code{metal_sentinel}.
  Unlike \code{\link{opencl_is_available}}, which performs a live enumeration of OpenCL devices on the current system, \code{metal_is_available} reflects a compile-time decision made once, during package installation.

  A \code{TRUE} result means the functions in \code{ardea} that depend on Metal were compiled and are callable; it does not by itself guarantee that a Metal-compliant GPU is currently attached, only that one was detected at install time.
  A \code{FALSE} result means underlying functions supporting interaction with Metal were not compiled into the local build of the package.
}
\examples{
  metal_is_available()
}
\seealso{
  \code{\link{opencl_is_available}},
  \code{\link{cuda_is_available}}
}
