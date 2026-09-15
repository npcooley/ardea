\name{cuda_is_available}
\alias{cuda_is_available}
\title{Check Whether CUDA Support Was Compiled Into This Build}
\description{
  Report whether \code{ardea} was compiled with CUDA support on this system.
}
\usage{
  cuda_is_available()
}
\value{
  Logical: \code{TRUE} if CUDA support was detected and compiled in at installation time, \code{FALSE} otherwise.
  CUDA detection is not yet implemented in \code{ardea}'s \code{configure} script, so this function currently returns \code{FALSE} unconditionally, on every platform.
}
\details{
  This function is an R wrapper for \code{ardea}'s C function \code{cuda_sentinel}.
  Like \code{\link{metal_is_available}}, and unlike \code{\link{opencl_is_available}}, this reflects a compile-time decision made once, during package installation.

  A \code{TRUE} result would mean the functions in \code{ardea} that depend on CUDA were compiled and are callable; it would not by itself guarantee that a CUDA-compliant GPU is currently attached.
  Until CUDA detection is implemented, this function exists so that code checking for CUDA support has a stable interface to call, without needing to change once CUDA support is added.
}
\examples{
  cuda_is_available()
}
\seealso{
  \code{\link{opencl_is_available}},
  \code{\link{metal_is_available}}
}
