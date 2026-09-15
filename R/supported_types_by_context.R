###### -- check type supports -------------------------------------------------
# ¯\_(ツ)_/¯

###### -- NOTES ---------------------------------------------------------------
# given a context, and a framework, detect the supported types

###### -- FUNCTION ------------------------------------------------------------

supported_types_by_context <- function(framework = c("opencl",
                                                     "metal",
                                                     "cuda"),
                                       context_ptr) {
  
  framework <- match.arg(framework)
  
  if ((framework == "opencl" &
       !is(object = context_ptr,
           class2 = "opencl_context")) |
      (framework == "metal" &
       !is(object = context_ptr,
           class2 = "metal_context")) |
      (framework == "cuda" &
       !is(object = context_ptr,
           class2 = "cuda_context"))) {
    stop("context pointer class must be aligned with the selected framework")
  }
  
  types_to_check <- switch(framework,
                           opencl = c("float",
                                      "double",
                                      "char",
                                      "short",
                                      "int",
                                      "long",
                                      "uchar",
                                      "ushort",
                                      "uint",
                                      "ulong"),
                           metal = stop("not yet supported"),
                           cuda = stop("not yet supported"))
  
  res <- switch(framework,
                opencl = vapply(X = types_to_check,
                                FUN = function(x) {
                                  .Call("opencl_probe_type_support",
                                        context_ptr,
                                        x,
                                        PACKAGE = "ardea")
                                },
                                USE.NAMES = TRUE,
                                FUN.VALUE = vector(mode = "logical",
                                                   length = 1)),
                metal = stop("not yet supported"),
                cuda = stop("not yet supported"))
  
  return(res)
}

