\name{supported_types_by_context}
\alias{supported_types_by_context}
\title{Check Supported Kernel Argument Types for a Context}
\description{
  Given a compute context and its associated framework, empirically probe which scalar C types can be used as kernel arguments on the device(s) backing that context.
}
\usage{
supported_types_by_context(framework = c("opencl",
                                         "metal",
                                         "cuda"),
                           context_ptr)
}
\arguments{
  \item{framework}{
    Character: vector of length 1, one of \code{"opencl"}, \code{"metal"}, or \code{"cuda"}, indicating which framework \code{context_ptr} belongs to.
    Partial matching via \code{\link{match.arg}} is supported; \code{"metal"} and \code{"cuda"} are not yet implemented, see Details.
  }
  \item{context_ptr}{
    externalptr: An external pointer to a compute context, as returned by \code{\link{opencl_make_context}} for the \code{"opencl"} framework.
    Its class must match \code{framework}, e.g. an \code{"opencl_context"} when \code{framework = "opencl"}.
  }
}
\value{
  A named logical vector, one entry per type keyword checked for the given framework.
  For \code{framework = "opencl"} the types are \code{c("float", "double", "char", "short", "int", "long", "uchar", "ushort", "uint", "ulong")}.
  Recorded as \code{TRUE} if a kernel using that type as a scalar argument compiles and builds successfully on the context's device(s), \code{FALSE} otherwise.
}
\details{
  This function a framework-specific list of type keywords, attempting to build a small single-argument kernel using that type via the C-level \code{opencl_probe_type_support} function.
  This is an empirical test rather than a spec or extension lookup: a type reported as unsupported may still be listed in a device's extension string, and vice versa. 
  \code{FALSE} reported types should be treated as a compile-time fact rather than a statement about the underlying hardware.

  Before probing, \code{context_ptr}'s class is checked against \code{framework} (\code{"opencl_context"}, \code{"metal_context"}, or \code{"cuda_context"} respectively).
  A mismatch, e.g. passing an OpenCL context while \code{framework = "metal"}, raises an error rather than silently probing the wrong API.

  \code{framework = "metal"} and \code{framework = "cuda"} are accepted by \code{\link{match.arg}} but currently raise \code{"not yet supported"} errors once past the class check; support for these frameworks is planned but not yet implemented.
}
\examples{
if (opencl_is_available()) {
  devices <- opencl_device_information()
  ctx <- opencl_make_context(devices[[1]])

  supported_types_by_context(framework = "opencl",
                             context_ptr = ctx)
}
}
\seealso{
  \code{\link{opencl_make_context}},
  \code{\link{opencl_make_program}}
}
