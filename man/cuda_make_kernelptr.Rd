\name{cuda_make_kernelptr}
\alias{cuda_make_kernelptr}
\title{Create Dispatch-Ready Kernel Pointers from a CUDA Module}
\description{
  Given a loaded CUDA module and the context it should be paired with, extract each named kernel function and return them as a named list of external pointers.
}
\usage{
  cuda_make_kernelptr(program,
                      context,
                      kernel_names)
}
\arguments{
  \item{program}{
    cuda_module: a module created by \code{\link{cuda_make_program}}.
  }
  \item{context}{
    cuda_context: the context created by \code{\link{cuda_make_context}}, used to determine which device the kernel(s) should be associated with. Should be the same context (or at least the same underlying device) that \code{program} was compiled/loaded with.
  }
  \item{kernel_names}{
    character: a vector of length 1 or greater, naming \code{__global__} functions declared in the \code{.cu} source \code{program} was compiled from.
  }
}
\details{
  Unlike \code{\link{metal_make_kernelptr}}, the underlying \code{CUfunction} handle has no reference counting at all -- unloading its parent module instantly invalidates every function derived from it, with no protection from the CUDA API itself.
  To guard against this, each kernel pointer returned here carries its parent \code{program} object as an attribute, which keeps \code{program} (and therefore its module) reachable, and thus un-collectable, for as long as any kernel built from it is still reachable -- even if the caller does not keep their own separate reference to \code{program}.

  Each kernel's maximum threads-per-block is queried once, at creation time, and cached -- this is a per-\emph{function} limit, not the per-device limit \code{\link{cuda_device_information}} reports, and is not re-queried on every dispatch.

  The returned list is named using \code{kernel_names}, in the order supplied.
}
\value{
  A named list of external pointers of class \code{"cuda_kernel"}, one per element of \code{kernel_names}.
}
\examples{
if (cuda_is_available()) {
  devices <- cuda_device_information()
  ctx <- cuda_make_context(devices[[1]])
  # mod <- cuda_make_program("kernel.cu", ctx)
  # kernels <- cuda_make_kernelptr(mod, ctx, c("my_kernel"))
}
}
\seealso{
  \code{\link{cuda_make_context}},
  \code{\link{cuda_make_program}}
}
