\name{metal_make_kernelptr}
\alias{metal_make_kernelptr}
\title{Create Dispatch-Ready Kernel Pointers from a Metal Library}
\description{
  Given a compiled Metal library and the context it should be paired with, build a compute pipeline for each named kernel function and return them as a named list of external pointers.
}
\usage{
  metal_make_kernelptr(program,
                       context,
                       kernel_names)
}
\arguments{
  \item{program}{
    metal_library: a library created by \code{\link{metal_make_program}}.
  }
  \item{context}{
    metal_context: the context created by \code{\link{metal_make_context}}, used to determine which device the pipeline(s) should be compiled against. Should be the same context (or at least the same underlying device) that \code{program} was compiled with.
  }
  \item{kernel_names}{
    character: a vector of length 1 or greater, naming \code{kernel void} functions declared in the \code{.metal} source \code{program} was compiled from.
  }
}
\details{
  For each name in \code{kernel_names}, the corresponding function is extracted from \code{program} and used to build a compute pipeline state. The intermediate function object is discarded immediately after the pipeline is built -- it is not retained or returned.

  Unlike \code{\link{opencl_make_kernelptr}}, which returns a handle (\code{cl_kernel}) that still needs binding to a device-specific execution context at dispatch time, each pointer returned here already bundles the compiled pipeline together with the identity of the device it was built against. It is dispatch-ready as returned: no separate pipeline-creation step remains to be performed per-call.

  Building a compute pipeline is a relatively expensive operation. This function is intended to be called once per kernel and the result reused across many dispatches, rather than rebuilding a pipeline on every call.

  The returned list is named using \code{kernel_names}, in the order supplied.
}
\value{
  A named list of external pointers of class \code{"metal_pipeline"}, one per element of \code{kernel_names}.
}
\examples{
metal_mm_naive <- '
kernel void metal_mm_naive(device float* output [[buffer(0)]],
                           constant uint& M [[buffer(1)]],
                           constant uint& K [[buffer(2)]],
                           constant uint& N [[buffer(3)]],
                           device const float* A [[buffer(4)]],
                           device const float* B [[buffer(5)]],
                           uint2 id [[thread_position_in_grid]])
{
    uint row = id.x;
    uint col = id.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint k = 0; k < K; k++) {
            sum += A[row + k * M] * B[k + col * K];
        }
        output[row + col * M] = sum;
    }
}
'
tmp_metal <- tempfile(fileext = ".metal")
tmp_metallib <- tempfile(fileext = ".metallib")
writeLines(text = metal_mm_naive,
           con = tmp_metal)

if (metal_is_available() &
    metal_devices_exist()) {
  devices <- metal_device_information()
  ctx <- metal_make_context(devices[[1]])
  lib <- metal_make_program(tmp_metal, ctx)
  kernels <- metal_make_kernelptr(lib, ctx, c("metal_mm_naive"))
}
}
\seealso{
  \code{\link{metal_make_context}},
  \code{\link{metal_make_program}}
}
