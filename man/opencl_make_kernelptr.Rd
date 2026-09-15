\name{opencl_make_kernelptr}
\alias{opencl_make_kernelptr}
\title{Extract Kernel Pointers from an OpenCL Program}
\description{
  Given a built OpenCL program and one or more kernel function names, extract callable kernel handles for each name.
}
\usage{
opencl_make_kernelptr(program,
                      kernel_names)
}
\arguments{
  \item{program}{
    An external pointer, created by \code{\link{opencl_make_program}}.
    Must represent a successfully built program.
  }
  \item{kernel_names}{
    A character vector of length 1 or greater, giving the name of each \code{__kernel}-qualified function to extract from \code{program}.
  }
}
\value{
  A named list, with one element per entry in \code{kernel_names}, in the same order.
  Each element is an external pointer wrapping the corresponding compiled kernel.
  List position names are inferred from the supplied \code{kernel_names}.
}
\details{
  A single OpenCL program can define multiple kernel functions; this function may be called once to extract several of them at once, rather than requiring one call per kernel.

  \code{clCreateKernel} is called once per name in \code{kernel_names}.
  Kernel function names that cannot be resolved will cause the function to error and return a message identifying the offending name.

  Each returned kernel handle is reference-counted by the OpenCL runtime and should be released automatically (\code{clReleaseKernel}) when its R external pointer is garbage collected.
}
\examples{
if (opencl_is_available()) {
  kernel_source <- "
    __kernel void add_one(__global float *data) {
    int gid = get_global_id(0);
    data[gid] = data[gid] + 1.0f;
  }
  "
  tmp01 <- tempfile(fileext = ".cl")
  writeLines(kernel_source, tmp01)

  devices <- opencl_device_information()
  ctx <- opencl_make_context(devices[[1]])
  
  program <- opencl_make_program(tmp01, ctx)
  kernels <- opencl_make_kernelptr(program, "add_one")
  unlink(tmp01)
}
}
\seealso{
  \code{\link{opencl_make_program}},
  \code{\link{opencl_make_context}}
}
