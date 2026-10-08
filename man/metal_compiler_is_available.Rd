\name{metal_compiler_is_available}
\alias{metal_compiler_is_available}
\title{Check Whether the Offline Metal Compiler Is Installed}
\description{
  Report whether the offline Metal compiler (\code{xcrun metal}) can be run on this system.
  This compiler is required only when \code{\link{metal_make_program}} is called with a non-\code{NULL} \code{metallib_file}.
}
\usage{
  metal_compiler_is_available()
}
\value{
  A single logical value: \code{TRUE} if \code{xcrun -sdk macosx metal --version} ran and exited with status \code{0}, \code{FALSE} otherwise.
  \code{FALSE} is always returned on non-macOS platforms, and when \code{xcrun} cannot be found on the \code{PATH}.
}
\details{
  \code{ardea} offers two ways to compile Metal source with \code{\link{metal_make_program}}:
  \itemize{
    \item With \code{metallib_file = NULL} (the default), source is compiled at runtime through the Metal API.
      This uses the shader compiler built into macOS and does not need any developer tools.
    \item With \code{metallib_file} supplied, source is compiled ahead of time by \code{xcrun metal} into a \code{.metallib} file.
      The \code{metal} compiler ships with full Xcode only.
      It is not part of the Xcode Command Line Tools, and recent Xcode versions provide it as a separate Metal Toolchain component that can be installed with \code{xcodebuild -downloadComponent MetalToolchain}.
  }

  Because of this, a system can return \code{TRUE} from both \code{\link{metal_is_available}} and \code{\link{metal_devices_exist}} while still returning \code{FALSE} here.
  Call this function before supplying \code{metallib_file} to \code{\link{metal_make_program}}.

  The check runs the compiler rather than only searching for it, because some Xcode versions include a \code{metal} placeholder that exists on disk but fails when the Metal Toolchain is not installed.
  The compiler's output is suppressed.
  Unlike \code{\link{metal_is_available}}, which reflects a decision made when the package was installed, this check is performed live each time the function is called.
}
\examples{
  metal_compiler_is_available()

  if (metal_is_available() &&
      metal_devices_exist() &&
      metal_compiler_is_available()) {
    print("ahead-of-time Metal compilation via 'metallib_file' is supported")
  }
}
\seealso{
  \code{\link{metal_make_program}},
  \code{\link{metal_is_available}},
  \code{\link{metal_devices_exist}}
}
