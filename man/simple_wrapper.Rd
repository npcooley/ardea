\name{simple_wrapper}
\alias{simple_wrapper}
\title{Perform Simple Kernel Function Dispatch for a Selected Framework}
\description{
  Dispatch a single compiled kernel function call to the selected GPU framework's underlying runner, using a common set of arguments.
  Currently supports \code{"opencl"}, \code{"metal"}, and \code{"cuda"}.
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
  
  \tabular{lccc}{
    \strong{Argument} \tab \strong{OpenCL constructor} \tab \strong{CUDA constructor} \tab \strong{Metal constructor}\cr
    \code{context_ptr} \tab \code{\link{opencl_make_context}} \tab \code{\link{cuda_make_context}} \tab \code{\link{metal_make_context}} \cr
    \code{kernel_ptr} \tab \code{\link{opencl_make_kernelptr}} \tab \code{\link{cuda_make_kernelptr}} \tab \code{\link{metal_make_kernelptr}} \cr
  }
  
  \tabular{lccc}{
    \strong{Argument} \tab \strong{OpenCL concept} \tab \strong{Metal concept} \tab \strong{CUDA concept} \cr
    \code{problem_dims} \tab \code{global size} \tab \strong{threads per grid} \tab \strong{grid size (total threads)} \cr
    \code{group_dims} \tab \code{work-group size} \tab \strong{threadgroup size} \tab \strong{block size} \cr
    \code{workers_per} \tab \code{threads per threadgroup} \tab \strong{threads per threadgroup} \tab \strong{threads per block} \cr
  }
  
  Validation of device availability and argument compatibility should all occur before dispatch.
}
\examples{
# write out a character vector to a tempfile for opencl, metal, and cuda
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
tmp_cl <- tempfile(fileext = ".cl")
writeLines(text = opencl_mm_naive,
           con = tmp_cl)

cuda_mm_naive <- '
extern "C" __global__ void cuda_mm_naive(float* output,
                                         const long long M,
                                         const long long K,
                                         const long long N,
                                         const float* A,
                                         const float* B)
{
    long long row = blockIdx.x * blockDim.x + threadIdx.x;
    long long col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (long long k = 0; k < K; k++) {
            sum += A[row + k * M] * B[k + col * K];
        }
        output[row + col * M] = sum;
    }
}
'
tmp_cu <- tempfile(fileext = ".cu")
tmp_ptx <- tempfile(fileext = ".ptx")
writeLines(text = cuda_mm_naive,
           con = tmp_cu)

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

# generic inputs, regardless of framework
var00 <- vector(mode = "numeric",
                length = 250000)
var01 <- matrix(rnorm(250000),
                nrow = 500,
                ncol = 500)
var02 <- matrix(rnorm(250000),
                nrow = 500,
                ncol = 500)
dim01 <- nrow(var01)
dim02 <- ncol(var01)
# dim03 <- nrow(var02)
dim04 <- ncol(var02)

arg_list <- list(var00, # our expected output vector, we've just filled it with zeros
                 dim01,
                 dim02,
                 dim04,
                 as.vector(var01),
                 as.vector(var02))

opencl_arg_types <- c("float",
                      "long",
                      "long",
                      "long",
                      "float",
                      "float")
metal_arg_types <- c("float",
                     "uint",
                     "uint",
                     "uint",
                     "float",
                     "float")
cuda_arg_types <- c("float",
                    "long",
                    "long",
                    "long",
                    "float",
                    "float")

print("Builtin implementation:")
system.time(res01 <- var01 \%*\% var02)

if (opencl_is_available() &
    opencl_devices_exist()) {
  print("OpenCL is available on this system!")
  cl_dvcs <- opencl_device_information()
  cl_ctx <- opencl_make_context(device = cl_dvcs[[1]])
  cl_program <- opencl_make_program(cl_file = tmp_cl,
                                    context = cl_ctx)
  cl_knl <- opencl_make_kernelptr(program = cl_program,
                                  kernel_names = "opencl_mm_naive")
  print("OpenCL implementation:")
  print(system.time(res02 <- simple_wrapper(framework = "opencl",
                                            context_ptr = cl_ctx,
                                            kernel_ptr = cl_knl$opencl_mm_naive,
                                            arg_types = opencl_arg_types,
                                            arg_list = arg_list,
                                            problem_dims = as.integer(c(dim01,
                                                                        dim04,
                                                                        1L)),
                                            group_dims = NULL,
                                            workers_per = NULL)))
  plot(as.vector(res01),
       res02,
       pch = 46,
       xlab = "builtin",
       ylab = "OpenCL")
}
if (cuda_is_available() &
    cuda_devices_exist()) {
  print("CUDA is available on this system!")
  cu_dvcs <- cuda_device_information()
  cu_ctx <- cuda_make_context(device = cu_dvcs[[1]])
  cu_program <- cuda_make_program(cuda_file = tmp_cu,
                                  context = cu_ctx,
                                  ptx_file = tmp_ptx)
  cu_knl <- cuda_make_kernelptr(program = cu_program,
                                context = cu_ctx,
                                kernel_names = "cuda_mm_naive")
  print("CUDA implementation:")
  print(system.time(res03 <- simple_wrapper(framework = "cuda",
                                            context_ptr = cu_ctx,
                                            kernel_ptr = cu_knl$cuda_mm_naive,
                                            arg_types = cuda_arg_types,
                                            arg_list = arg_list,
                                            problem_dims = as.integer(c(dim01,
                                                                        dim04,
                                                                        1L)),
                                            group_dims = NULL,
                                            workers_per = NULL)))
  plot(as.vector(res01),
       res03,
       pch = 46,
       xlab = "builtin",
       ylab = "CUDA")
}
if (metal_is_available() &
    metal_devices_exist()) {
  print("Metal is available on this system!")
  mtl_dvcs <- metal_device_information()
  mtl_ctx <- metal_make_context(device = mtl_dvcs[[1]])
  mtl_program <- metal_make_program(metal_file = tmp_metal,
                                    metallib_file = tmp_metallib,
                                    context = mtl_ctx)
  mtl_knl <- metal_make_kernelptr(program = mtl_program,
                                  context = mtl_ctx,
                                  kernel_names = "metal_mm_naive")
  print("Metal implementation:")
  print(system.time(res03 <- simple_wrapper(framework = "metal",
                                            context_ptr = mtl_ctx,
                                            kernel_ptr = mtl_knl$metal_mm_naive,
                                            arg_types = metal_arg_types,
                                            arg_list = arg_list,
                                            problem_dims = as.integer(c(dim01,
                                                                        dim04,
                                                                        1L)),
                                            group_dims = NULL,
                                            workers_per = NULL)))
  plot(as.vector(res01),
       res03,
       pch = 46,
       xlab = "builtin",
       ylab = "metal")
}
}
\seealso{
  \code{\link{opencl_make_context}},
  \code{\link{opencl_make_program}},
  \code{\link{opencl_make_kernelptr}},
  \code{\link{cuda_make_context}},
  \code{\link{cuda_make_program}},
  \code{\link{cuda_make_kernelptr}},
  \code{\link{metal_make_context}},
  \code{\link{metal_make_program}},
  \code{\link{metal_make_kernelptr}}
}
