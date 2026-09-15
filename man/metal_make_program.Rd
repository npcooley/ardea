\name{metal_make_program}
\alias{metal_make_program}
\title{Compile a Metal Source File into a Library}
\description{
  Compile a \code{.metal} source file against a \code{\link{metal_make_context}} context, and return the resulting library as an external pointer, ready to be passed to \code{\link{metal_make_kernelptr}}.
}
\usage{
  metal_make_program(metal_file,
                     context,
                     metallib_file = NULL,
                     xcrun_flags = NULL,
                     fast_math = FALSE,
                     language_version = NULL)
}
\arguments{
  \item{metal_file}{
    character: a length-1 path to a \code{.metal} source file. Must exist.
  }
  \item{context}{
    metal_context: a context created by \code{\link{metal_make_context}}.
  }
  \item{metallib_file}{
    character or NULL: if supplied, a length-1 output path for a compiled \code{.metallib}. Selects the \code{xcrun}-based compilation path described in Details, rather than the default runtime API path.
  }
  \item{xcrun_flags}{
    character or NULL: additional flags passed through to \code{xcrun metal} on the \code{metallib_file} compilation path. Ignored (with a warning if non-NULL) when \code{metallib_file} is NULL. Each element is treated as a single shell word, matching \code{system2()}'s own \code{args} convention -- a flag that takes a value, e.g. \code{-D}, must be supplied as two separate elements (\code{c("-D", "TILE_SIZE=32")}), not one space-joined string. May not include \code{"-o"}, since this function already supplies that flag from \code{metallib_file}.
  }
  \item{fast_math}{
    logical: a single non-NA value. Enables Metal's fast-math compiler option on the runtime API compilation path. Ignored (with a warning if non-default) when \code{metallib_file} is supplied.
  }
  \item{language_version}{
    character or NULL: a length-1 Metal Shading Language version string (e.g. \code{"2.4"}, \code{"3.0"}), or NULL to use the platform default. Applies to the runtime API compilation path only; ignored (with a warning if non-NULL) when \code{metallib_file} is supplied. Availability of specific version strings depends on the SDK this package was built against.
  }
}
\details{
  This function supports two compilation paths, selected by whether \code{metallib_file} is supplied:

  \strong{Runtime API path} (\code{metallib_file = NULL}, the default): source is compiled in-process via Metal's \code{newLibraryWithSource:options:error:}, using \code{fast_math} and \code{language_version} to configure the compile options. This requires the Metal compiler toolchain (normally installed alongside Xcode or the Xcode Command Line Tools) to be present on the machine running R -- a stricter requirement than simply having a Mac with a supported GPU. Because compilation is from an in-memory source string with no filesystem search-path context, \code{#include} of separate header files is not reliably supported on this path.

  \strong{xcrun path} (\code{metallib_file} supplied): \code{metal_file} is compiled to a \code{.metallib} at \code{metallib_file} by shelling out to \code{xcrun -sdk macosx metal}, and the compiled library is then loaded from disk. This path does not share the runtime path's \code{#include} limitation, since \code{xcrun} compiles a real file with real filesystem context. Use \code{xcrun_flags} to reach \code{xcrun metal}'s own compiler flags (include paths, preprocessor defines, and so on) on this path; \code{fast_math}/\code{language_version} have no effect here.

  Both paths return an object of the same class -- which compiler produced the library is not visible to \code{\link{metal_make_kernelptr}} or anything downstream of it.
}
\value{
  An external pointer of class \code{"metal_library"} wrapping the compiled library.
}
\examples{
if (metal_is_available()) {
  devices <- metal_device_information()
  ctx <- metal_make_context(devices[[1]])

  # runtime API compilation
  # lib <- metal_make_program("kernel.metal", ctx)

  # xcrun compilation, with an extra include path
  # lib2 <- metal_make_program("kernel.metal",
  #                            ctx,
  #                            metallib_file = tempfile(fileext = ".metallib"))
}
}
\seealso{
  \code{\link{metal_make_context}},
  \code{\link{metal_make_kernelptr}}
}
