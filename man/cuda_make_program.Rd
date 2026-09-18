\name{cuda_make_program}
\alias{cuda_make_program}
\title{Compile a CUDA Source File into a Loaded Module}
\description{
  Compile a \code{.cu} source file to PTX via \code{nvcc}, and load the result against a \code{\link{cuda_make_context}} context, returning it as an external pointer ready to be passed to \code{\link{cuda_make_kernelptr}}.
}
\usage{
  cuda_make_program(cuda_file,
                    context,
                    ptx_file = NULL,
                    nvcc_flags = NULL)
}
\arguments{
  \item{cuda_file}{
    character: a length-1 path to a \code{.cu} source file. Must exist.
  }
  \item{context}{
    cuda_context: a context created by \code{\link{cuda_make_context}}.
  }
  \item{ptx_file}{
    character or NULL: a length-1 output path for the compiled \code{.ptx} file. If NULL (the default), a temporary file is created via \code{tempfile(fileext = ".ptx")}.
  }
  \item{nvcc_flags}{
    character or NULL: additional flags passed through to \code{nvcc}. Each element is treated as a single shell word, matching \code{system2()}'s own \code{args} convention -- a flag that takes a value, e.g. \code{-I}, must be supplied as two separate elements (\code{c("-I", "/path/to/headers")}), not one space-joined string. May not include \code{"-o"}, since this function already supplies that flag from \code{ptx_file}.
  }
}
\details{
  Unlike \code{\link{metal_make_program}}, there is no in-process runtime-compile path here -- no NVRTC integration currently exists, so this always shells out to \code{nvcc} to produce a \code{.ptx} file (\code{nvcc <cuda_file> -ptx -o <ptx_file>}), then loads it via \code{cuModuleLoad()}.
  This requires that \code{nvcc} is resolvable in the \code{PATH} of the R session (or otherwise locatable via \code{Sys.which("nvcc")}), which in turn requires a working CUDA toolkit install on the machine running R -- a stricter requirement than a working CUDA driver alone.

  Arguments are passed to \code{system2()} as individually \code{shQuote()}'d elements of a vector, not a single pasted string -- \code{system2()} still joins them with spaces internally before the shell sees the line, so \code{shQuote()} on each path-bearing element is what actually protects against spaces or shell metacharacters, not the vector structure by itself.
  The \code{nvcc} exit status is checked explicitly, and the expected \code{.ptx} file's existence is verified before this function ever calls into C.
}
\value{
  An external pointer of class \code{"cuda_module"} wrapping the loaded module.
}
\examples{
if (cuda_is_available() &
    cuda_devices_exist()) {
  devices <- cuda_device_information()
  ctx <- cuda_make_context(devices[[1]])

  # mod <- cuda_make_program("kernel.cu", ctx)

  # with an extra include path
  # mod2 <- cuda_make_program("kernel.cu",
  #                           ctx,
  #                           ptx_file = tempfile(fileext = ".ptx"),
  #                           nvcc_flags = c("-I", "/path/to/headers"))
}
}
\seealso{
  \code{\link{cuda_make_context}},
  \code{\link{cuda_make_kernelptr}}
}
