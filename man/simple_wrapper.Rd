\name{simple_wrapper}
\alias{simple_wrapper}
\title{Perform Simple Kernel Function Dispatch for a Selected Framework}
\description{
  Dispatches a single compiled kernel function call to the selected GPU framework's underlying runner, using a common set.
  Currently only \code{"opencl"} is implemented; \code{"metal"} and \code{"cuda"} will be absorbed from the \code{ACFmetal} and \code{ACFcuda} packages respectively.
}
\usage{
simple_wrapper(framework = c("opencl",
                             "metal",
                             "cuda"),
              context_ptr,
              kernel_ptr,
              arg_types,
              arg_list,
              problem_dims,
              group_dims,
              workers_per = NULL)
}
\arguments{
  \item{framework}{
    Character: One of \code{"opencl"}, \code{"metal"}, or \code{"cuda"}, partial matching via \code{\link{match.arg}} is supported.
    Only \code{"opencl"} is currently implemented; the other two raise an informative error.
  }
  \item{context_ptr}{
    Externalptr: Classed to match the option passed to \code{framework}.
  }
  \item{kernel_ptr}{
    Externalptr: Classed to match the option passed to \code{framework}.
  }
  \item{arg_types}{
    Character: vector the same length as \code{arg_list}, naming the device-side type of each corresponding argument (e.g. \code{"float"}, \code{"double"}, \code{"int"}, \code{"long"}).
    The first element describes the output.
  }
  \item{arg_list}{
    List: A list the same length as \code{arg_types}.
    The first element is the output template: its length and declared type (per \code{arg_types[1]}) define the result buffer.
    Remaining elements are the input arguments -- length-1 elements are bound as scalar kernel arguments, longer elements are staged into device buffers.
  }
  \item{problem_dims}{
    Numeric: A vector of length 3 giving the logical (global) problem size in up to three dimensions -- e.g. \code{c(nrow, ncol, 1)} for a 2D problem.
    Unused trailing dimensions should be set to \code{1}.
  }
  \item{group_dims}{
    Numeric: A vector of length 3 giving the local work-group size per dimension, or \code{NULL} to let the underlying runner derive a default from the device's reported limits and \code{workers_per}.
  }
  \item{workers_per}{
    Numeric: An optional single value giving the target number of work-items per work-group, used only when \code{group_dims} is \code{NULL} to derive a default work-group size.
    Defaults to \code{NULL} (the underlying the device's default is queries when applicable).
  }
}
\value{
  A numeric or integer vector (matching the type declared in \code{arg_types[1]}) of length \code{length(arg_list[[1]])}, containing the dispatched kernel's output.
}
\details{
  This function serves as a consistent dispatch interface across frameworks for simple use cases.
  
  \tabular{lc}{
    \strong{Argument} \tab \strong{OpenCL constructor} \cr
    \code{context_ptr} \tab \code{\link{opencl_make_context}} \cr
    \code{kernel_ptr} \tab \code{\link{opencl_make_kernelptr}} \cr
  }
  
  \tabular{lccc}{
    \strong{Argument} \tab \strong{OpenCL concept} \tab \strong{Metal concept} \tab \strong{CUDA concept} \cr
    \code{problem_dims} \tab \code{global size} \tab \strong{x} \tab \strong{y} \cr
    \code{group_dims} \tab \code{work-group size} \tab \strong{x} \tab \strong{y} \cr
    \code{workers_per} \tab \code{threads per threadgroup} \tab \strong{x} \tab \strong{y} \cr
  }
  
  Validation of device availability and argument compatibility should all occur before dispatch.
}
\examples{

  var00 <- vector(mode = "numeric",
                  length = 2500)
  var01 <- matrix(rnorm(2500),
                  nrow = 50,
                  ncol = 50)
  var02 <- matrix(rnorm(2500),
                  nrow = 50,
                  ncol = 50)
  dim01 <- nrow(var01)
  dim02 <- ncol(var01)
  # dim03 <- nrow(var02)
  dim04 <- ncol(var02)
  
  arg_list <- list(var00,
                   dim01,
                   dim02,
                   dim04,
                   as.vector(var01),
                   as.vector(var02))
  
  res01 <- var01 %*% var02
  
  if (opencl_is_available()) {
  
    opencl_arg_types <- c("float", # output(0)
                          "long",  # M (scalar)
                          "long",  # K (scalar)
                          "long",  # N (scalar)
                          "float", # A
                          "float") # B
    
    opencl_mm_naive <- '
__kernel void opencl_mm_naive(__global float* output,
                              const long M,
                              const long K,
                              const long N,
                              __global const float* A,
                              __global const float* B)
{
    long row = get_global_id(0);
    long col = get_global_id(1);

    if (row < M && col < N) {
        float sum = 0.0f;
        for (long k = 0; k < K; k++) {
            sum += A[row + k * M] * B[k + col * K];
        }
        output[row + col * M] = sum;
    }
}
'
    tmp01 <- tempfile(fileext = ".cl")
    writeLines(opencl_mm_naive, tmp01)

    cl_dvcs <- opencl_device_information()
    cl_ctx <- opencl_make_context(device = cl_dvcs[[1]])
    cl_program <- opencl_make_program(cl_file = tmp01,
                                      context = cl_ctx)
    cl_knl <- opencl_make_kernelptr(program = cl_program,
                                    kernel_names = "opencl_mm_naive")

    res02 <- simple_wrapper(framework = "opencl",
                            context_ptr = cl_ctx,
                            kernel_ptr = cl_knl$opencl_mm_naive,
                            arg_types = opencl_arg_types,
                            arg_list = arg_list,
                            problem_dims = as.integer(c(dim01, dim04, 1L)),
                            group_dims = NULL,
                            workers_per = NULL)

    unlink(tmp01)
  }
  if (metal_is_available()) {
    metal_arg_types = c("float", # buffer(0): output
                        "uint", # buffer(1): M (scalar)
                        "uint", # buffer(2): K (scalar)
                        "uint", # buffer(3): N (scalar)
                        "float", # buffer(4): A
                        "float") # buffer(5): B
  
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

    tmp02 <- tempfile(fileext = ".metal")
    writeLines(text = metal_mm_naive,
               con = tmp02)

    mtl_dvcs <- metal_device_information()
    mtl_ctx <- metal_make_context(device = mtl_dvcs[[1]])

    tmp03 <- tempfile(fileext = ".metallib")
    mtl_program <- metal_make_program(metal_file = tmp02,
                                      metallib_file = tmp03,
                                      context = mtl_ctx)
    
    mtl_knl <- metal_make_kernelptr(program = mtl_program,
                                    context = mtl_ctx,
                                    kernel_names = "metal_mm_naive")
    res03 <- simple_wrapper(framework = "metal",
                            context_ptr = mtl_ctx,
                            kernel_ptr = mtl_knl$metal_mm_naive,
                            arg_types = metal_arg_types,
                            arg_list = arg_list,
                            problem_dims = as.integer(c(dim01, dim04, 1L)),
                            group_dims = NULL,
                            workers_per = NULL)
    unlink(tmp02)
    unlink(tmp03)
  }
}
\seealso{
  \code{\link{opencl_make_context}},
  \code{\link{opencl_make_program}},
  \code{\link{opencl_make_kernelptr}}
}
