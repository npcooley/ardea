\name{opencl_make_program}
\alias{opencl_make_program}
\title{Compile an OpenCL Program from Source}
\description{
  Reads a \code{.cl} source file, compiles it against a given OpenCL context, and returns the built program.
}
\usage{
opencl_make_program(cl_file,
                    context,
                    build_options = NULL)
}
\arguments{
  \item{cl_file}{
    Character: vector of length 1 giving the path to a \code{.cl} OpenCL C source file.
    Must exist.
  }
  \item{context}{
    An external pointer, as returned by \code{\link{opencl_make_context}}.
    The program is compiled for the device(s) associated with this context.
  }
  \item{build_options}{
    Character: An optional vector of length 1 containing OpenCL C compiler options to be passed to \code{clBuildProgram}.
    Defaults to \code{NULL} signifying default options, see Details.
  }
}
\value{
  An external pointer wrapping the built OpenCL program.
  This is consumed by \code{\link{opencl_make_kernelptr}} to extract callable kernel handles.
}
\details{
  Source is read from \code{cl_file} and compiled via \code{clCreateProgramWithSource} followed by \code{clBuildProgram}.
  Unlike CUDA and Metal, OpenCL's compiler is part of the runtime API so we are not invoking \code{system2} to communicate with an external executable.

  If compilation fails, the OpenCL build log should be printed via a warning message before an error is raised, so the underlying OpenCL C compiler error (missing semicolon, unknown type, etc.) is visible to the user rather than only a generic OpenCL error code.

  \code{build_options} corresponds directly to the \code{options} argument of \code{clBuildProgram}; see \strong{"What build_options is for"} below for its purpose and common values.
}
\section{What build_options is for}{
  \code{build_options} is a single string of space-separated flags passed to the OpenCL C compiler at build time.
  This is analogous to flags that would be passed to \code{gcc}, \code{clang}, \code{nvcc}, or \code{xcrun}.
  It lets a user influence how the \code{.cl} source is compiled without editing the source file itself.
  Common categories:

  \itemize{
    \item{
      \strong{Preprocessor defines}, e.g. \code{"-DBLOCK_SIZE=256"}.
      Sets a preprocessor macro used in the kernel source, letting the same \code{.cl} file be recompiled with different constants (tile sizes, loop bounds, feature toggles) without maintaining multiple source files.
    }
    \item{
      \strong{Include paths}, e.g. \code{"-I/path/to/headers"}. Needed if the OpenCL file includes non-standard headers.
    }
    \item{
      \strong{Language version selection}, e.g. \code{"-cl-std=CL2.0"}.
      Set a specific OpenCL C language version, if the source relies on features not present in the device's default version.
    }
    \item{
      \strong{Optimization and numerical-precision trade-offs}, e.g. \code{"-cl-fast-relaxed-math"} or \code{"-cl-mad-enable"}.
      Relax IEEE-754 conformance in exchange for faster generated code.
      Appropriate when a kernel's numerical tolerance allows it, but not a safe default for every workload.
    }
    \item{
      \strong{Warnings-as-errors}, e.g. \code{"-Werror"}.
      Catch compiler warnings that would otherwise be silently ignored.
    }
  }

  Multiple flags can be combined in the one string, since \code{clBuildProgram} expects a single options string, e.g.:
  \preformatted{
    opencl_make_program("kernels.cl", ctx,
                        build_options = "-DBLOCK_SIZE=256 -cl-fast-relaxed-math")
  }

  Most users compiling straightforward kernels can leave \code{build_options} as \code{NULL}.
  It exists for cases where compile-time configuration or a specific numerical/language mode is needed, not as a required argument.
}
\examples{
if (opencl_is_available() &
    opencl_devices_exist()) {
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
  unlink(tmp01)
}
}
\seealso{
  \code{\link{opencl_make_context}},
  \code{\link{opencl_make_kernelptr}}
}